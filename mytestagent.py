import requests
from openai import OpenAI
import chromadb
import os

from langgraph.graph import StateGraph
from typing import TypedDict
from dotenv import load_dotenv
from recommender.ncf_recommender import NCFRecommender

load_dotenv()

print("""
=====================================
 MULTI-AGENT EDUCATIONAL SYSTEM
 RAG + LANGGRAPH + CHROMADB
=====================================
""")

# =========================
# 🔹 API KEYS
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
recommender = NCFRecommender()

print("=== SISTEMA MULTI-AGENTE CON RAG + LANGGRAPH ===")

# =========================
# 🔹 CHROMA DB (RAG)
# =========================

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="knowledge_base"
)

if collection.count() == 0:
    collection.add(

         documents=[

        # ITALIANO - SCOLASTICO
        "Gioco del telefono senza fili per sviluppare il linguaggio nei bambini",
        "Attività di lettura condivisa per bambini della scuola dell'infanzia",
        "Carte illustrate per aumentare il vocabolario nei bambini",

        # INGLESE - SCOLASTICO
        "Storytelling activities for preschool language development",
        "Interactive games for improving children's vocabulary",

        # SPAGNOLO - SCOLASTICO
        "Actividades lingüísticas para niños en edad preescolar",
        "Juegos educativos para mejorar la comprensión verbal",

        # PIATTAFORMA
        "La piattaforma consente agli insegnanti di creare attività personalizzate",
        "Users can upload educational resources directly into the platform",

        # ALTRO
        "Suggerimenti generali per migliorare la comunicazione nei bambini"

    ],

    metadatas=[

        {"ambito": "scolastico", "lingua": "it"},
        {"ambito": "scolastico", "lingua": "it"},
        {"ambito": "scolastico", "lingua": "it"},

        {"ambito": "scolastico", "lingua": "en"},
        {"ambito": "scolastico", "lingua": "en"},

        {"ambito": "scolastico", "lingua": "es"},
        {"ambito": "scolastico", "lingua": "es"},

        {"ambito": "piattaforma", "lingua": "it"},
        {"ambito": "piattaforma", "lingua": "en"},

        {"ambito": "altro", "lingua": "it"}

    ],

    ids=[
        "1","2","3","4","5",
        "6","7","8","9","10"
    ]
)

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

    - KB_SEARCH
    → domande sulla piattaforma o informazioni interne

    - BOTH
    → richieste educative che richiedono sia web che knowledge base

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

    "come funziona la piattaforma"
    → KB_SEARCH

    "ricerca siti educativi"
    → WEB_SEARCH

    IMPORTANTE:
    Le richieste educative per bambini,
    giochi didattici,
    attività linguistiche,
    risorse scolastiche
    devono essere classificate come BOTH.

    Rispondi SOLO con:
    WEB_SEARCH
    KB_SEARCH
    BOTH
    CHAT

    Conversazione:
    Utente: attività linguistiche bambini
    Utente: fammi qualcosa in inglese
    → BOTH

    Conversazione:
    Utente: come funziona la piattaforma
    Utente: gli insegnanti possono caricare file?
    → KB_SEARCH
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    intent = response.choices[0].message.content.strip()

    valid_intents = [
        "WEB_SEARCH",
        "KB_SEARCH",
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
# 🔹 AGENTE 3: KB (RAG)
# =========================

def kb_agent(query):

    print(">>> KB AGENT (CHROMA RAG) <<<")

    # =========================
    # 🔹 METADATA FILTER
    # =========================

    filter_metadata = {}

    text = query.lower()

    # lingua
    if "inglese" in text or "english" in text:
        filter_metadata["lingua"] = "en"

    elif "spagnolo" in text or "spanish" in text:
        filter_metadata["lingua"] = "es"

    else:
        filter_metadata["lingua"] = "it"

    # ambito
    if "piattaforma" in text:
        filter_metadata["ambito"] = "piattaforma"

    elif any(k in text for k in ["bambini", "gioco", "attività"]):
        filter_metadata["ambito"] = "scolastico"

    print(">>> FILTER KB:", filter_metadata)


    # =========================
    # 🔹 QUERY CHROMA
    # =========================

    # costruzione filtro ChromaDB

    where_filter = None

    if len(filter_metadata) == 1:

        where_filter = filter_metadata

    elif len(filter_metadata) > 1:

        where_filter = {
            "$and": [
                {k: v} for k, v in filter_metadata.items()
            ]
        }

    # query database

    results = collection.query(
        query_texts=[query],
        n_results=2,
        where=where_filter
    )

    # documenti recuperati

    docs = results.get("documents", [[]])

    if not docs or len(docs[0]) == 0:

        return [{
            "title": "Nessuna risorsa trovata",
            "description": "La knowledge base non contiene risultati coerenti con la richiesta.",
            "source": "KB"
        }]

    docs = docs[0]

    print("\n>>> DOCUMENTI RECUPERATI KB:")
    print(docs)

    # =========================
    # 🔹 FORMAT OUTPUT
    # =========================

    formatted = []

    for d in docs:

        formatted.append({
            "title": "Risorsa KB",
            "description": d,
            "source": "KB"
        })

    return formatted

    # =========================
    # 🔹 recommendation NCF - Marta
    # =========================

def recommendation_agent(user_id=0):

    print(">>> NCF RECOMMENDER AGENT <<<")

    recs = recommender.recommend(
        user_id=user_id,
        top_k=3
    )

    return recs

# =========================
# 🔹 AGENTE 4: CRITICO
# =========================

def critic_agent(web_results, kb_results, intent, recommendations):

    if intent == "KB_SEARCH":

        prompt = f"""
        Hai questi risultati dalla knowledge base:
        {kb_results}

        Usa SOLO queste informazioni.

        Spiega chiaramente all’utente come funziona la piattaforma
        o rispondi alla domanda informativa.

        NON parlare di bambini 3-6 anni
        se non richiesto.
        """

    else:

        prompt = f"""
        Hai questi risultati WEB:
        {web_results}

        Hai questi risultati KB:
        {kb_results}

        NCF RECOMMENDATIONS:
        {recommendations}

        Intent:
        {intent}

        REGOLE:
        - usa SOLO risultati forniti
        - NON inventare informazioni

        Seleziona le risorse più utili per bambini 3-6 anni.
        Spiega brevemente perché sono utili.
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

    detected_language: str

    last_topic: str

    intent: str

    query: str

    web_results: list

    recommendations: list

    kb_results: list

    final_output: str

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

def kb_node(state: AgentState):

    # se non esiste query usa input utente
    query = state.get("query", state["user_input"])

    return {
        "kb_results": kb_agent(query)
    }

def recommendation_node(state):

    if state["intent"] != "BOTH":

        return {
            "recommendations": []
        }

    return {
        "recommendations": recommendation_agent(
            user_id=0
        )
    }

def critic_node(state: AgentState):

    if state["intent"] == "CHAT":

        return {
            "final_output": "Ciao! Come posso aiutarti?"
        }

    return {
    "final_output": critic_agent(
        state.get("web_results", []),
        state.get("kb_results", []),
        state["intent"],
        state.get("recommendations", [])
    )
}

# =========================
# 🔹 ROUTER DINAMICO
# =========================

def route_intent(state: AgentState):

    intent = state["intent"]

    print("\n>>> ROUTING:", intent)

    if intent == "CHAT":
        return "critic"

    elif intent == "KB_SEARCH":
        return "kb"

    elif intent == "WEB_SEARCH":
        return "query"

    elif intent == "BOTH":
        return "query"

    return "critic"

# =========================
# 🔹 COSTRUZIONE GRAFO
# =========================

graph = StateGraph(AgentState)

graph.add_node("intent", intent_node)
graph.add_node("query", query_node)
graph.add_node("web", web_node)
graph.add_node("kb", kb_node)
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
graph.add_conditional_edges(
    "web",
    lambda state:
        "kb" if state["intent"] == "BOTH"
        else "critic"
)

# NCF
graph.add_edge("kb", "recommendation")
graph.add_edge("recommendation", "critic")


app = graph.compile()

# =========================
# 🔹 RUN
# =========================

print("\nSistema avviato.")
print("Scrivi una richiesta oppure digita 'exit', 'quit' o 'esci' per terminare.\n")

# =========================
# 🔹 CONVERSAZIONE APERTA
# =========================

history = []

while True:

    user_input = input("Utente: ")

    # uscita
    if user_input.lower() in ["exit", "quit", "esci"]:
        print("\nSistema terminato.")
        break

    result = app.invoke({
    "user_input": user_input,
    "conversation_history": history
})

    history.append({
    "user": user_input,
    "assistant": result["final_output"]
})

    print("\nAssistente:")
    print(result["final_output"])
    print()