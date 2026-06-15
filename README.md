# Multi-Agent System

Multi-Agent System is an educational recommendation platform based on a multi-agent architecture built with LangGraph, OpenAI, Streamlit and SerpAPI.

The system combines semantic recommendation, web search and critical evaluation capabilities to help users discover educational and professional learning resources.

---

## Features

### Intent Detection Agent

Classifies user requests and determines the most appropriate execution flow.

Supported interaction types include:

* General conversation
* Resource recommendation
* Web search enrichment
* Combined recommendation and web search

---

### Recommendation Agent (Embedding-Based Retrieval)

The recommendation layer retrieves resources from a dataset containing more than 6,700 books and learning materials.

Recommendations are generated through semantic retrieval based on sentence embeddings.

The system:

* generates embeddings from resource title and description
* stores precomputed embeddings in the dataset
* generates an embedding for the user query
* computes cosine similarity between query and resources
* retrieves the Top-10 most relevant candidates

This approach allows semantic matching beyond simple keyword search.

---

### Critic Agent

The Critic Agent performs a second-level evaluation of the retrieved resources.

It receives:

* user query
* candidate resources
* titles
* descriptions
* metadata

The agent filters irrelevant resources and returns a structured JSON response:

```json
{
  "answer": "Brief explanation of the retrieved resources",
  "ids": [1008, 5670]
}
```

Where:

* `answer` contains a short relevance assessment
* `ids` contains only the identifiers of the resources considered relevant

Only resources approved by the Critic Agent are shown to the user.

---

### Web Search Agent

When additional information may be useful, the system can perform web searches through SerpAPI.

The web module:

* retrieves between 3 and 5 web resources
* supports optional enrichment after recommendations
* acts as a fallback when no relevant recommendation is found

Users can decide whether to explore web resources after receiving recommendations.

---

### Streamlit Frontend

The Streamlit interface provides:

* conversational chat interface
* semantic resource recommendations
* clickable web resources
* clickable dataset resources
* dedicated resource detail pages
* agent execution badges
* optional web search enrichment

---

## Resource Detail Pages

Dataset resources do not contain external URLs.

Each recommended resource can be opened through a dedicated page displaying:

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
Embedding Retrieval   Web Search Agent
       |
       v
Top-10 Candidates
       |
       v
Critic Agent
       |
       v
Filtered Resources
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
* SentenceTransformers
* Scikit-learn
* Pandas

---

## Project Structure

```text
agent/
│
├── frontend.py
├── mytestagent.py
├── generate_embeddings.py
│
├── pages/
│   └── resource.py
│
├── recommender/
│   ├── simulated_ncf.py
│   ├── book_rich_metadata.csv
│   └── book_rich_metadata_embeddings.csv
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
