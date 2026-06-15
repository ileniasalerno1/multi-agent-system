# Multi-Agent System

Multi-Agent System is an educational recommendation platform based on a multi-agent architecture built with LangGraph, OpenAI, Streamlit and SerpAPI.

The system combines recommendation, web search and critical evaluation capabilities to help users discover educational and professional learning resources.

---

## Features

### Intent Detection Agent

Classifies user requests and determines the most appropriate execution flow.

Supported interaction types include:

* General conversation
* Educational resource recommendation
* Professional resource recommendation
* Web search enrichment

---

### Recommendation Agent (Simulated NCF)

The recommendation layer retrieves educational resources from a dataset of over 6,700 items.

Recommendations are generated using:

* textual similarity
* resource categories
* educational domain matching
* professional domain matching
* popularity score

---

### Web Search Agent

When additional information may be useful, the system can perform web searches through SerpAPI.

Users can decide whether to:

* accept the recommended resources only
* request a web-based enrichment

If no relevant recommendation is found, the system automatically falls back to web search.

---

### Critic Agent

The Critic Agent evaluates the retrieved resources and generates a final response explaining why the selected resources are relevant to the user's request.

The user never sees internal system details or execution logic.

---

### Streamlit Frontend

The Streamlit interface provides:

* conversational chat interface
* recommendation cards
* clickable web resources
* clickable dataset resources
* dedicated resource detail pages
* agent execution badges
* optional web search enrichment

---

## Resource Detail Pages

Dataset resources do not contain external URLs.

To improve usability, each recommended resource can be opened through a dedicated page displaying:

* Title
* Author
* Publication year
* Average rating
* Description

Resources are retrieved through their unique `book_id`.

---

## Agent Workflow

```text
User Query
    |
    v
Intent Agent
    |
    +--------------------+
    |                    |
    v                    v
Recommendation Agent   Web Search Agent
       \                /
        \              /
         v            v
          Critic Agent
                |
                v
            Frontend
```

---

## Technologies

* Python
* LangGraph
* OpenAI API
* Streamlit
* SerpAPI
* Pandas

---

## Project Structure

```text
agent/
│
├── frontend.py
├── mytestagent.py
│
├── pages/
│   └── resource.py
│
├── recommender/
│   ├── simulated_ncf.py
│   └── book_rich_metadata.csv
│
├── .env
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ileniasalerno1/multi-agent-system.git
cd multi-agent-system
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

---

## Authors

Ilenia Salerno
