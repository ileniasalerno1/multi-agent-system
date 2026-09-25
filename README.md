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

> **Note on Neural Collaborative Filtering (NCF).** The Recommendation Agent does not implement a real NCF model. The system operates without a persistent user-item interaction history (every request is treated as an independent query), which makes classic collaborative filtering inapplicable. For this reason, semantic retrieval on embeddings was adopted instead, occupying the same architectural position originally intended for NCF without replicating its collaborative learning behavior. See the thesis, Chapter 2, for a full discussion.

---

# Features

## Intent Agent

The Intent Agent represents the entry point of the architecture.

Its purpose is to classify each user request and route it through the correct execution flow.

Supported intents include:

* General conversation (`CHAT`)
* Resource recommendation (`BOTH`)
* Explicit web search request (`WEB_SEARCH`)

The agent exploits an LLM-based classifier and considers both the current request and the conversation history.

---

## Recommendation Agent

The Recommendation Agent retrieves educational resources from a dataset containing more than **6,700 books and learning materials**.

Unlike traditional keyword-based search engines, recommendations are generated through **semantic similarity**.

The recommendation process consists of:

1. generating sentence embeddings for every resource title and description;
2. storing the precomputed embeddings inside the dataset;
3. generating an embedding for the user query;
4. computing cosine similarity between the query embedding and every resource embedding, filtered by a similarity threshold (0.3);
5. retrieving the Top-k (k=10) most similar resources.

The similarity threshold and the value of k were chosen empirically during development, based on qualitative observation of retrieval results; no systematic tuning procedure was applied.

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

Only the resources approved by the Critic Agent are displayed within the user interface. If the JSON output is malformed, the system falls back to the unfiltered recommendations rather than failing.

---

## Web Search Agent

The Web Search Agent integrates **SerpAPI** to retrieve external learning resources whenever additional information is required.

It is important to distinguish between two separate moments of the fallback mechanism: the *activation of the option* and the *actual execution* of the search.

The web search option is proposed to the user in two scenarios:

* **automatically**, when the Critic Agent does not find any relevant resource in the internal dataset (the interface displays a "Search the Web" option, without running any search yet);
* **on direct user request**, even when the internal dataset has already returned relevant resources.

In both cases, however, the **execution** of the web search is never automatic: it always requires an explicit confirmation action from the user (e.g. clicking the proposed option). This design choice follows an earlier version of the system in which the web search ran automatically on every request, producing often superfluous information and increasing response times.

Retrieved resources are displayed as clickable links inside the Streamlit interface and are also validated by the Critic Agent before being shown.

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
* automatic fallback suggestions (activation only — execution requires user confirmation).

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
Web Search Agent (option proposed, execution on user confirmation)
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

The benchmark evaluates every component of the architecture. Two separate benchmarks are used for methodological reasons: the Recommendation Agent evaluation requires a manually annotated ground truth at the resource level (`book_id`), which is costly to produce and therefore limited in size; the Intent Agent evaluation only requires a category label, allowing a larger benchmark to be built without the same annotation cost.

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

The evaluation isolates the specific contribution of the Critic Agent by comparing two configurations on the same retrieved candidates: **baseline** (raw semantic retrieval, no filtering) and **with Critic Agent** (candidates validated and filtered).

---

## Intent Agent

Implemented metrics:

* Intent Accuracy
* Per-class Precision / Recall / F1-score

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

This evaluation was conducted manually by the author, through direct inspection of generated answers, on a small set of exploratory queries used during development — not on an automated procedure or an external evaluation model. Given its non-systematic nature, these results should be interpreted as a preliminary qualitative indication rather than a statistically robust measure.

---

# Experimental Results

## Recommendation Agent: baseline vs. Critic Agent

The benchmark contains 8 `BOTH` queries, of which 7 have a defined ground truth (the query "machine learning" has no annotated reference resources and is excluded from metric computation).

| Metric                  | Baseline  | With Critic Agent |
| ------------------------ | --------- | ------------------ |
| Precision                | 0.341     | **0.786**          |
| Recall                   | 0.857     | 0.784              |
| Task Success Rate        | 0.857     | 0.857              |
| F1-score                 | 0.425     | 0.765              |
| MAP                      | 0.852     | 0.784              |
| nDCG                     | 0.856     | 0.857              |
| Avg. resources returned  | 8.0       | 2.57               |

The Critic Agent nearly doubles Precision while keeping Task Success Rate unchanged, confirming its effectiveness in filtering false positives from pure semantic retrieval.

## Intent Agent (99-query balanced benchmark)

An initial evaluation on a 10-query benchmark (2 `CHAT`, 8 `BOTH`) had produced a 100% accuracy. Further analysis revealed a methodological limitation: that benchmark contained no `WEB_SEARCH` query, leaving a third of the categories untested, and the `BOTH` queries were limited to a single technical domain. The benchmark was therefore expanded to 99 queries, balanced across the three categories and including non-technical domains and ambiguous edge cases.

| Metric                    | Value     |
| -------------------------- | --------- |
| Intent Accuracy            | **88.9%** |
| Routing Precision (macro)  | 0.884     |
| Routing Recall (macro)     | 0.897     |
| Routing F1 (macro)         | 0.888     |

| Class        | Precision | Recall | F1-score |
| ------------ | --------- | ------ | -------- |
| BOTH         | 0.93      | 0.86   | 0.89     |
| CHAT         | 0.90      | 0.87   | 0.88     |
| WEB_SEARCH   | 0.83      | 0.96   | 0.89     |

The main limitation observed is reduced generalization on non-technical domains (e.g. fiction, cooking), since the prompt's few-shot examples are mostly technical/programming-oriented.

## Response Time

| Metric              | Value    |
| -------------------- | -------- |
| Average response time | 4.47 s  |
| Minimum               | 1.12 s  |
| Maximum                | 8.98 s  |

Higher response times consistently correspond to cases where the web fallback is triggered.

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
│   ├── benchmark_v2.csv
│   ├── evaluate_intent.py
│   ├── evaluate_intent_v2.py
│   ├── evaluate_recommender.py
│   ├── evaluate_baseline_vs_critic.py
│   ├── metrics.py
│   ├── response_time.py
│   ├── routing_metrics.py
│   ├── task_success_rate.py
│   ├── intent_results.csv
│   ├── intent_results_v2.csv
│   ├── recommender_results.csv
│   ├── baseline_vs_critic_results.csv
│   ├── response_times.csv
│   ├── routing_confusion_matrix.csv
│   ├── routing_confusion_matrix_v2.csv
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

* expanding the Recommendation Agent benchmark, currently limited to 7 usable queries;
* integrating a dedicated vector database (e.g. ChromaDB) for retrieval scalability on larger datasets;
* enriching the Intent Agent's few-shot examples to improve generalization on non-technical domains;
* complementing automatic metrics with a user study (Likert-scale questionnaire);
* integration of a real Neural Collaborative Filtering model, should a persistent user-item interaction history become available;
* automatic (rather than manual) evaluation of the Web Search Agent;
* multilingual support and user profile personalization.

---

# Authors

**Ilenia Salerno**
