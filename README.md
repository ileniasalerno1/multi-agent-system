# Multi-Agent System

Multi-Agent System is an educational recommendation platform based on a modular multi-agent architecture developed using **LangGraph**, **OpenAI**, **Streamlit**, and **SerpAPI**.

The system combines semantic recommendation, conversational interaction, web search and critical evaluation to support users in discovering educational and technical learning resources.

The architecture is composed of specialized agents that cooperate to retrieve, validate and present the most relevant resources according to the user's request.

---

# Architecture

The system is organized into five main components:

* **Intent Agent**
* **Recommendation Agent**
* **Critic Agent**
* **Web Search Agent**
* **Streamlit Frontend**

Each agent is responsible for a specific task within the recommendation pipeline.

---

# Features

## Intent Agent

The Intent Agent represents the entry point of the architecture.

Its purpose is to classify each user request and route it through the correct execution flow.

Supported intents include:

* General conversation
* Resource recommendation
* Combined recommendation and web search

The agent exploits an LLM-based classifier and considers both the current request and the conversation history.

---

## Recommendation Agent

The Recommendation Agent retrieves educational resources from a dataset containing more than **6,700 books and learning materials**.

Unlike traditional keyword-based search engines, recommendations are generated through **semantic similarity**.

The recommendation process consists of:

1. generating sentence embeddings for every resource title and description;
2. storing the precomputed embeddings inside the dataset;
3. generating an embedding for the user query;
4. computing cosine similarity between the query embedding and every resource embedding;
5. retrieving the Top-k most similar resources.

The embedding generation process is performed offline through the `generate_embeddings.py` script using the **SentenceTransformer all-MiniLM-L6-v2** model.

---

## Critic Agent

The Critic Agent performs a second-level validation of the retrieved resources.

Instead of directly presenting the recommendations to the user, the Critic Agent receives:

* the original user query;
* candidate resources;
* titles;
* descriptions;
* metadata.

Using an LLM, the agent filters irrelevant resources and returns a structured JSON object containing only the identifiers of the resources considered relevant.

Example:

```json
{
  "answer": "Brief explanation of the retrieved resources.",
  "ids": [2663, 2474]
}
```

Only the resources approved by the Critic Agent are displayed within the user interface.

---

## Web Search Agent

The Web Search Agent integrates **SerpAPI** to retrieve external learning resources whenever additional information is required.

The Web Agent is activated in two situations:

* when the user explicitly requests a web search;
* when the Recommendation Agent cannot retrieve sufficiently relevant resources (automatic fallback).

Retrieved resources are displayed as clickable links inside the Streamlit interface.

---

## Streamlit Frontend

The frontend provides a conversational interface where users can interact naturally with the system.

Available features include:

* conversational chat interface;
* semantic recommendations;
* clickable dataset resources;
* dedicated resource detail pages;
* clickable web resources;
* agent execution badges;
* optional web search enrichment;
* automatic fallback suggestions.

---

# Recommendation Pipeline

```text
User Query
      │
      ▼
Intent Agent
      │
      ├──────────────► Conversation
      │
      ▼
Recommendation Agent
      │
      ▼
Embedding Retrieval
      │
      ▼
Top-k Candidate Resources
      │
      ▼
Critic Agent
      │
      ├──────────────► Relevant resources found
      │
      ▼
No relevant resources
      │
      ▼
Web Search Agent
      │
      ▼
Final Response
```

---

# Resource Detail Pages

Resources stored inside the dataset do not contain external URLs.

Each recommendation can therefore be explored through a dedicated Streamlit page showing:

* Title
* Author
* Publication Year
* Average Rating
* Description

Each resource is uniquely identified through its `book_id`.

---

# Evaluation Framework

The project includes a complete experimental evaluation framework located inside the `evaluation` folder.

The benchmark evaluates every component of the architecture.

## Recommendation Agent

Implemented metrics:

* Task Success Rate (TSR)
* Precision
* Recall
* F1-score
* Precision@5
* Recall@5
* Precision@10
* Recall@10
* Mean Average Precision (MAP)
* Normalized Discounted Cumulative Gain (nDCG)

---

## Intent Agent

Implemented metrics:

* Intent Accuracy

---

## Routing

Implemented metrics:

* Routing Precision
* Routing Recall
* Routing F1-score
* Confusion Matrix

---

## System Performance

Implemented metrics:

* Response Time
* Communication Cost

---

## Web Agent

Qualitative evaluation includes:

* Answer Accuracy
* Faithfulness
* Citation Accuracy

---

# Experimental Results

Current benchmark results:

| Metric                | Value      |
| --------------------- | ---------- |
| Intent Accuracy       | **100%**   |
| Routing Precision     | **1.000**  |
| Routing Recall        | **1.000**  |
| Routing F1-score      | **1.000**  |
| Task Success Rate     | **80%**    |
| Precision             | **0.129**  |
| Recall                | **0.800**  |
| F1-score              | **0.220**  |
| MAP                   | **0.640**  |
| nDCG                  | **0.677**  |
| Average Response Time | **4.47 s** |

---

# Technologies

* Python
* LangGraph
* OpenAI API
* Streamlit
* SerpAPI
* SentenceTransformers
* Scikit-learn
* Pandas
* NumPy

---

# Project Structure

```text
agent/
│
├── frontend.py
├── mytestagent.py
├── generate_embeddings.py
│
├── evaluation/
│   ├── benchmark.csv
│   ├── evaluate_intent.py
│   ├── evaluate_recommender.py
│   ├── metrics.py
│   ├── response_time.py
│   ├── routing_metrics.py
│   ├── task_success_rate.py
│   ├── intent_results.csv
│   ├── recommender_results.csv
│   ├── response_times.csv
│   ├── routing_confusion_matrix.csv
│   └── task_success_results.csv
│
├── recommender/
│   ├── simulated_ncf.py
│   ├── book_rich_metadata.csv
│   └── book_rich_metadata_embeddings.csv
│
├── pages/
│   └── resource.py
│
├── .env
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/ileniasalerno1/multi-agent-system.git
cd multi-agent-system
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Configure the environment variables:

```env
OPENAI_API_KEY=your_key
SERP_API_KEY=your_key
```

Generate the embeddings (only required the first time):

```bash
python generate_embeddings.py
```

Run the application:

```bash
streamlit run frontend.py
```

---

# Future Work

Possible future developments include:

* integration of real Neural Collaborative Filtering models;
* larger educational datasets;
* hybrid recommendation strategies;
* multilingual support;
* automatic evaluation of the Web Agent;
* user profile personalization.

---

# Authors

**Ilenia Salerno**
