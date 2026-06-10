# Multi-Agent System

Multi-Agent System è una piattaforma sperimentale basata su un'architettura multi-agente sviluppata con LangGraph, OpenAI, ChromaDB e Streamlit.

Il sistema è progettato per supportare la ricerca e la raccomandazione di risorse educative attraverso l'integrazione di:

* Intent Detection
* Knowledge Base Retrieval (RAG)
* Recommendation System
* Web Search
* Critical Evaluation Agent

## Features

### Intent Detection Agent

Identifica automaticamente l'intento dell'utente classificando le richieste nelle seguenti categorie:

* CHAT
* KB_SEARCH
* WEB_SEARCH
* BOTH

### Knowledge Base Agent (RAG)

Utilizza ChromaDB per recuperare informazioni da una knowledge base locale contenente risorse educative e informazioni sulla piattaforma.

### Recommendation Agent

Genera raccomandazioni utilizzando un sistema di scoring basato su:

* corrispondenza testuale tra query e contenuti
* categorie delle risorse
* domini educativi e professionali
* popolarità delle risorse

### Web Search Agent

Quando le risorse interne non risultano sufficientemente pertinenti, il sistema può effettuare ricerche sul web tramite SerpAPI.

### Critic Agent

Valuta criticamente le risorse recuperate e produce una spiegazione della loro pertinenza rispetto alla richiesta dell'utente.

### Streamlit Frontend

Interfaccia conversazionale sviluppata con Streamlit che consente di:

* interagire con il sistema tramite chat
* visualizzare le risorse raccomandate
* richiedere approfondimenti sul web tramite pulsanti dedicati

## Technologies

* Python
* LangGraph
* OpenAI API
* ChromaDB
* Streamlit
* SerpAPI
* Pandas

## Project Structure

```text
agent/
│
├── frontend.py
├── mytestagent.py
│
├── recommender/
│   └── simulated_ncf.py
│
├── chroma_db/
│
├── .env
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/ileniasalerno1/educational-multi-agent-system.git
cd educational-multi-agent-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure environment variables:

```env
OPENAI_API_KEY=your_key
SERP_API_KEY=your_key
```

Run the application:

```bash
streamlit run frontend.py
```

## Authors

Ilenia Salerno
