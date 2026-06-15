import requests
from openai import OpenAI
import os
import json

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
        
    print("RISULTATI GOOGLE:", len(results))

    return results[:5]

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

    Restituisci SOLO un JSON valido.

    NON utilizzare markdown.
    NON utilizzare ```json.
    Restituisci esclusivamente il JSON.

    Formato:
    {{
        "answer": "breve valutazione delle risorse trovate",
        "ids": [2663, 1450]
    }}

    oppure

    {{
        "answer": "non sono state trovate risorse pertinenti",
        "ids": []
    }}

    Regole:
    - answer deve contenere massimo 3 frasi
    - answer deve spiegare se le risorse sono pertinenti alla richiesta
    - ids deve contenere esclusivamente i book_id delle risorse pertinenti
    - non utilizzare markdown
    - restituisci esclusivamente il JSON
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    text = response.choices[0].message.content.strip()

    print("INDICI SELEZIONATI:", text)

    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    try:

        data = json.loads(text)

        answer = data["answer"]
        selected_ids = data["ids"]

        filtered_recommendations = [
            rec
            for rec in recommendations
            if rec["book_id"] in selected_ids
        ]

        return {
            "answer": answer,
            "recommendations": filtered_recommendations
        }

    except Exception as e:

        print("ERRORE PARSING JSON:", e)
        print("RISPOSTA GPT:", text)

        return {
            "answer": "Errore durante la valutazione delle risorse.",
            "recommendations": recommendations
        }

    # =========================
    # 🔹 recommendation NCF - Marta
    # =========================

def recommendation_agent(query):

    recs = recommender.recommend(
        query=query,
        top_k=10
    )

    result = filter_recommendations_llm(
        query,
        recs
    )

    return result

# =========================
# 🔹 AGENTE 4: CRITICO
# =========================

def critic_agent(user_query, web_results, intent, recommendations):

    prompt = f"""
    Richiesta utente:
    {user_query}

    Risorse NCF:
    {recommendations}

    Risultati Web:
    {web_results}

    Intent:
    {intent}

    Regole:

    - usa SOLO le informazioni fornite
    - non inventare informazioni
    - non fare riassunti
    - non elencare titoli
    - non creare classifiche
    - non numerare risultati

    Se intent = BOTH:
    valuta le risorse NCF.

    Se intent = WEB_SEARCH:
    valuta i risultati Web.

    Se i risultati mostrati sono pertinenti,
    spiega brevemente che il sistema ha trovato
    contenuti rilevanti e che l'utente può aprirli
    per approfondire.

    Se intent = WEB_SEARCH:
    NON dire che non sono stati trovati risultati.
    NON suggerire ulteriori ricerche web.
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

    recommendations_answer: str

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

    result = recommendation_agent(
        state["user_input"]
    )

    recs = result["recommendations"]

    answer = result["answer"]

    best_score = max(
        [r["score"] for r in recs],
        default=0
    )

    return {
        "recommendations": recs,
        "recommendations_answer": answer,
        "recommendations_found": len(recs) > 0,
        "best_recommendation_score": best_score
    }

def critic_node(state: AgentState):

    if state["intent"] == "CHAT":

        return {
            "final_output": "Ciao! Come posso aiutarti?"
        }

    if state["intent"] == "BOTH":

        if state.get("recommendations_found", False):

            return {
                "final_output":
                    state.get(
                        "recommendations_answer",
                        ""
                    )
            }

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

    history.append({
        "user": user_input,
        "assistant": result["final_output"]
    })

    needs_web_search = result.get(
        "recommendations_found",
        False
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