import requests
from openai import OpenAI
import os

from langgraph.graph import StateGraph
from typing import TypedDict
from dotenv import load_dotenv
from recommender.simulated_ncf import SimulatedNCF

recommender = SimulatedNCF()
DEBUG = False

load_dotenv()

# =========================
# 🔹 API KEYS
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

print("=== SISTEMA MULTI-AGENTE ===")

# =========================
# 🔹 AGENTE 1: QUERY
# =========================

def query_agent(user_input):
    prompt = f"""
    Scrivi UNA SOLA query di ricerca Google.
    NON spiegare nulla.
    SOLO la query.

    Input: {user_input}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    query = response.choices[0].message.content.strip()
    query = query.split("\n")[0]

    if ":" in query:
        query = query.split(":")[-1]

    return query.replace('"', '').strip()

# =========================
# 🔹 AGENTE INTENTO
# =========================

def intent_agent(user_input, history):

    prompt = f"""
    Cronologia conversazione:
    {history}

    Nuova richiesta:
    {user_input}

    Se la nuova richiesta dipende dal contesto precedente,
    usa la cronologia per capire l’intento corretto.

    Classifica la richiesta utente in UNA categoria.

    Categorie:

    - WEB_SEARCH
    → richieste di ricerca sul web

    - BOTH
    → richieste educative gestite tramite il sistema di raccomandazione e, se necessario, approfondite tramite ricerca web
    
    - CHAT
    → semplici saluti o conversazione generica

    Esempi:

    "ciao"
    → CHAT

    "trova giochi linguistici"
    → BOTH

    "attività linguistiche bambini 3-6 anni"
    → BOTH

    "giochi educativi per scuola infanzia"
    → BOTH

    "attività per sviluppare il linguaggio"
    → BOTH

    "risorse educative per bambini"
    → BOTH

    "ricerca siti educativi"
    → WEB_SEARCH

    IMPORTANTE:

    Richieste contenenti linguaggi di programmazione
    o termini informatici come:

    Python
    Java
    C++
    Programming
    Coding
    Software Development
    Computer Science

    devono essere classificate come BOTH.

    IMPORTANTE:
    Le richieste educative per bambini,
    giochi didattici,
    attività linguistiche,
    risorse scolastiche
    devono essere classificate come BOTH.

    Rispondi SOLO con:
    WEB_SEARCH
    BOTH
    CHAT

    Conversazione:
    Utente: attività linguistiche bambini
    Utente: fammi qualcosa in inglese
    → BOTH
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    intent = response.choices[0].message.content.strip()
    
    valid_intents = [
        "WEB_SEARCH",
        "BOTH",
        "CHAT"
    ]

    if intent not in valid_intents:
        intent = "CHAT"

    return intent

# =========================
# 🔹 AGENTE 2: SEARCH WEB
# =========================

def search_agent(query):
    url = "https://serpapi.com/search.json"

    params = {
        "q": query,
        "api_key": SERP_API_KEY
    }

    res = requests.get(url, params=params)
    data = res.json()

    if "organic_results" not in data:

        return []

    results = []

    for r in data.get("organic_results", []):
        results.append({
        "type": "link",
        "title": r.get("title", ""),
        "link": r.get("link", ""),
        "snippet": r.get("snippet", "")
    })

    # filtro semantico
    filtered = []
    for r in results:
        text = (r["title"] + " " + r["snippet"]).lower()

        if any(k in text for k in ["bambin", "linguaggio", "gioco", "infanzia"]):
            filtered.append(r)

    if len(filtered) == 0:
        return results[:5]

    return filtered[:5]

    # =========================
    # 🔹 filtro per risposte non pertinenti
    # =========================

def filter_recommendations_llm(user_query, recommendations):

    prompt = f"""
    Richiesta utente:
    {user_query}

    Risorse candidate:
    {recommendations}

    Mantieni SOLO le risorse chiaramente pertinenti.

    Elimina:
    - risultati fuori tema
    - risultati vagamente correlati
    - risultati che condividono solo una parola

    Restituisci SOLO gli indici da mantenere.

    Esempio:

    [0,2]

    oppure

    [1]

    oppure

    []
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    text = response.choices[0].message.content.strip()

    try:
        indices = eval(text)

        return [
            recommendations[i]
            for i in indices
            if i < len(recommendations)
        ]

    except:
        return recommendations[:1]

    # =========================
    # 🔹 recommendation NCF - Marta
    # =========================

def recommendation_agent(query):

    recs = recommender.recommend(
        query=query,
        top_k=3
    )

    recs = filter_recommendations_llm(
        query,
        recs
    )

    return recs

# =========================
# 🔹 AGENTE 4: CRITICO
# =========================

def critic_agent(user_query, web_results, intent, recommendations):

    prompt = f"""

    Richiesta utente:
    {user_query}

    Hai questi risultati WEB:
    {web_results}

    Intent:
    {intent}

    Risorse recuperate dal sistema di raccomandazione:
    {recommendations}

    REGOLE:
    - usa SOLO risultati forniti
    - NON inventare informazioni
    - valuta criticamente la pertinenza dei risultati
    - NON assumere che tutte le risorse siano utili
    - se una risorsa non è chiaramente pertinente alla richiesta, dillo esplicitamente
    - NON scrivere lunghi riassunti dei libri o delle risorse
    - concentrati sulla relazione tra richiesta utente e risultati trovati

    IMPORTANTE:
    Se la richiesta riguarda informatica,
    programmazione o computer science,
    libri tecnici della categoria Computers
    devono essere considerati altamente pertinenti.

    Se il titolo della risorsa contiene parole uguali o molto simili
    alla richiesta dell'utente (es. Python, Java, Computer,
    Software, Programming, Coding, Informatica),
    la pertinenza deve essere considerata ALTA.

    Non classificare come BASSA una risorsa che contiene
    direttamente l'argomento richiesto nel titolo.

    Esempio:
    Richiesta: Python
    Titolo: Learning Python
    → Pertinenza ALTA

    Richiesta: Java
    Titolo: Java Tutorial
    → Pertinenza ALTA

    OUTPUT:

    Per ogni risorsa:
    - titolo
    - livello di pertinenza (Alta / Media / Bassa)
    - breve motivazione (1 frase)

    Se nessuna risorsa è realmente pertinente,
    spiega che sarebbe utile effettuare una ricerca web.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# =========================
# 🔹 LANGGRAPH STATE
# =========================

class AgentState(TypedDict):

    user_input: str

    conversation_history: list

    intent: str

    query: str

    web_results: list

    final_output: str

    recommendations: list

    recommendations_found: bool

    best_recommendation_score: float

# =========================
# 🔹 NODI (AGENTI)
# =========================
def intent_node(state: AgentState):
      return {
        "intent": intent_agent(
            state["user_input"],
            state.get("conversation_history", [])
        )
}

def query_node(state: AgentState):

    if state["intent"] == "CHAT":
        return {"query": ""}

    return {
        "query": query_agent(state["user_input"])
    }

def web_node(state: AgentState):
    return {"web_results": search_agent(state["query"])}

def recommendation_node(state):

    if state["intent"] != "BOTH":

        return {
            "recommendations": [],
            "recommendations_found": False,
            "best_recommendation_score": 0
        }

    recs = recommendation_agent(
        state["user_input"]
    )

    best_score = max(
        [r["score"] for r in recs],
        default=0
    )

    return {
        "recommendations": recs,
        "recommendations_found": len(recs) > 0,
        "best_recommendation_score": best_score
    }

def critic_node(state: AgentState):

    if state["intent"] == "CHAT":

        return {
            "final_output": "Ciao! Come posso aiutarti?"
        }
    
    if (
        state["intent"] == "BOTH"
        and state.get("recommendations_found", False)
        and state.get("best_recommendation_score", 0) < 20
    ):

        return {
            "final_output":
            """
Non ho trovato contenuti sufficientemente pertinenti nel sistema di raccomandazione.

Vuoi che effettui una ricerca web per trovare contenuti più specifici?
            """
        }

    response = critic_agent(
        state["user_input"],
        state.get("web_results", []),
        state["intent"],
        state.get("recommendations", [])
    )

    # Se sono state trovate raccomandazioni NCF
    if state.get("recommendations_found", False):

        response += """

Vuoi che effettui anche una ricerca sul web per approfondire?
    """

    return {
        "final_output": response
    }

# =========================
# 🔹 ROUTER DINAMICO
# =========================

def route_intent(state: AgentState):

    intent = state["intent"]

    if DEBUG:
        print("\n>>> ROUTING:", intent)

    if intent == "CHAT":
        return "critic"

    elif intent == "WEB_SEARCH":
        return "query"

    elif intent == "BOTH":
        return "recommendation"

    return "critic"

# =========================
# 🔹 COSTRUZIONE GRAFO
# =========================

graph = StateGraph(AgentState)

graph.add_node("intent", intent_node)
graph.add_node("query", query_node)
graph.add_node("web", web_node)
graph.add_node("recommendation", recommendation_node)
graph.add_node("critic", critic_node)

graph.set_entry_point("intent")

# routing dinamico
graph.add_conditional_edges(
    "intent",
    route_intent
)

# FLOW BOTH / WEB
graph.add_edge("query", "web")

# dopo web
graph.add_edge("web", "critic")

# recommendation
graph.add_conditional_edges(
    "recommendation",
    lambda state:
        "critic"
        if state["recommendations_found"]
        else "query"
)

app = graph.compile()

# =========================
# 🔹 Which agent?
# =========================

def build_agents_used(result):

    agents = []

    if result.get("recommendations"):
        agents.append("NCF")

    agents.append("CRITIC")

    return agents

# =========================
# 🔹 RUN
# =========================

def run_agent(user_input, history):

    result = app.invoke({
        "user_input": user_input,
        "conversation_history": history
    })

    print(result.keys())
    print(result.get("recommendations"))

    history.append({
        "user": user_input,
        "assistant": result["final_output"]
    })

    needs_web_search = (
        "Vuoi che effettui anche una ricerca sul web"
        in result["final_output"]
    )

    return {
        "response": result["final_output"],
        "needs_web_search": needs_web_search,
        "recommendations": result.get("recommendations", []),
        "agents_used": build_agents_used(result)
    }


def run_web_search(user_query):

    query = query_agent(user_query)

    web_results = search_agent(query)

    response = critic_agent(
        user_query,
        web_results,
        "WEB_SEARCH",
        []
    )

    return {
        "response": response,
        "web_results": web_results,
        "agents_used": ["WEB", "CRITIC"]
    }


if __name__ == "__main__":

    print("\nSistema avviato.")
    print(
        "Scrivi una richiesta oppure digita "
        "'exit', 'quit' o 'esci' per terminare.\n"
    )

    history = []

    while True:

        user_input = input("Utente: ")

        if user_input.lower() in [
            "exit",
            "quit",
            "esci"
        ]:
            print("\nSistema terminato.")
            break

        result = run_agent(
            user_input,
            history
        )

        print("\nAssistente:")
        print(result["response"])
        print()

        if result["needs_web_search"]:

            choice = input(
                "Approfondire sul web? (si/no): "
            )

            if choice.lower() == "si":

                web_response = run_web_search(
                    user_input
                )

                print("\nAssistente:")
                print(web_response)
                print()