# Multi-Agent System

Prototype of a multi-agent recommendation system developed with **LangGraph**, **OpenAI**, **Sentence Transformers**, **SerpAPI** and **Streamlit**.

This repository contains the code and the experimental data of the bachelor's thesis *"Dal retrieval semantico alla raccomandazione affidabile: un approccio multi-agente con validazione critica"* (Ilenia Salerno, Università degli Studi di Milano, Informatica per la Comunicazione Digitale, A.Y. 2025/2026), developed during an internship at Social Thingum.

The system helps users who look for study and learning resources on their own, describing their need in natural language. It works **without any persistent user-item interaction history**: there is no authentication and every request is processed as an independent event.

---

## Architecture

The system is made of four agents, coordinated by a **LangGraph state graph** with a shared state and conditional routing. The routing rules are written in code and are deterministic with respect to the state; the state itself, however, contains the output of LLM-based components (the classified intent and the outcome of the validation).

| Agent | Role |
| --- | --- |
| **Intent Agent** | Classifies the request as `CHAT`, `BOTH` (resource recommendation) or `WEB_SEARCH`, using the conversation history |
| **Recommendation Agent** | Retrieves candidate resources through semantic retrieval and submits them to validation |
| **Critic Agent** | Validates the candidates (LLM, JSON output) and composes the final answer |
| **Web Search Agent** | Formulates a search query and retrieves external resources through SerpAPI |

> **Note on Neural Collaborative Filtering (NCF).** The Recommendation Agent does not implement an NCF model. Without user identification and interaction history, a collaborative model would have no data to be trained on. Semantic retrieval occupies the architectural position originally intended for NCF, but measures the semantic relevance of the request instead of learning individual preferences. The class implementing it is still called `SimulatedNCF` for historical reasons. See Chapter 3 of the thesis.

### Execution paths

```text
CHAT                          intent → critic
BOTH, resources validated     intent → recommendation → critic
BOTH, no resource validated   intent → recommendation → query → web → critic
WEB_SEARCH                    intent → query → web → critic
```

In addition, after every recommendation request the interface offers a **web search on user confirmation**, which runs the query, search and evaluation steps outside the graph.

---

## Components

### Intent Agent

Classifies each request with a few-shot prompt (full text in Appendix B of the thesis). Labels that are not among the three admitted values fall back to `CHAT`. The conversation history is used only for this classification step, not for retrieval.

### Recommendation Agent

Operates on a dataset of **6,761 resources** (`recommender/book_rich_metadata.csv`). The dataset is a general-purpose book collection: fiction is the largest category (37.9%), while computer science resources are 91. Descriptions are almost entirely in English.

Retrieval steps:

1. embeddings of title and description of every resource are precomputed offline with **Sentence Transformers `all-MiniLM-L6-v2`** (`generate_embeddings.py`);
2. the query embedding is computed at request time;
3. cosine similarity is computed against every resource;
4. resources below a similarity threshold of **0.3** are discarded and the top **k = 10** are kept.

The threshold and k were chosen empirically during development, without a systematic tuning procedure. Note that the embedding model is trained on English, while user requests are in Italian.

### Critic Agent

Receives the request and the candidate resources and returns a JSON object with a short assessment and the identifiers of the relevant resources:

```json
{
  "answer": "Brief assessment of the retrieved resources.",
  "ids": [2663, 2474]
}
```

Only identifiers belonging to the submitted candidates are kept. If the JSON cannot be parsed, the system falls back to the **unfiltered** candidates; in that case the resources are shown without being distinguished from validated ones.

### Web Search Agent

The web branch is reached in three situations:

* **explicit web search request** (`WEB_SEARCH` intent): the search runs automatically and its results are evaluated and shown;
* **deepening on user confirmation**: after every recommendation request the interface asks whether to search the web; the search runs only if the user accepts;
* **no validated resource** (`BOTH` intent): the graph runs a web search, but the final node does not use its results and shows a fixed message proposing a search on confirmation, which then runs a second search. This is a known limitation of the current version: in this case the system does **not** provide an automatic fallback to web content.

### Streamlit interface

Conversational interface with clickable recommended resources, a detail page for each resource (`pages/resource.py`), clickable web results and badges showing the executed agents (`INTENT`, `RECOMMENDER`, `QUERY`, `WEB`, `CRITIC`). Badges indicate that an agent was executed, not necessarily that it contributed to the displayed answer.

---

## Evaluation

All evaluation scripts and results are in the `evaluation` folder. Every measure comes from a **single execution**: since LLM outputs are not deterministic, a new run may produce slightly different values.

### Recommendation Agent: baseline vs. Critic Agent

Benchmark: 10 queries (`benchmark.csv`), 8 of type `BOTH`, of which **7 with an annotated ground truth** ("machine learning" has none and is excluded). The ground truth was annotated by a single person with a topical criterion and is incomplete for at least one query; Precision and Recall therefore refer to the annotated resources.

The comparison runs on the same retrieved candidates, with and without validation (`evaluate_baseline_vs_critic.py`, `baseline_vs_critic_results.csv`).

| Metric | Baseline | With Critic Agent |
| --- | --- | --- |
| Precision | 0.341 | **0.786** |
| Recall | 0.857 | 0.784 |
| F1-score | 0.425 | 0.765 |
| Task Success Rate | 0.857 | 0.857 |
| MAP | 0.852 | 0.784 |
| nDCG@10 | 0.856 | 0.807 |
| Avg. resources returned | 8.00 | 2.57 |

**Diagnostic comparison.** Truncating the baseline ranking to the same length produced by the Critic gives F1 = 0.749, against 0.765 for the Critic; truncating to the top 3 gives 0.507. Most of the gain therefore comes from deciding *how many* resources to keep, rather than *which* ones.

**Sensitivity analysis.** Excluding the two queries with problematic annotation ("libri di programmazione", "cerca siti per imparare python"), values change but the relations between configurations do not (Precision 0.455 → 0.900, F1 0.872 vs. 0.848 for the same-length cut).

### Intent Agent

An initial benchmark of 10 queries (`benchmark.csv`, `intent_results.csv`) gave 100% accuracy but contained no `WEB_SEARCH` query. It was replaced by a 99-query benchmark (`benchmark_v2.csv`, `evaluate_intent_v2.py`, `intent_results_v2.csv`), with 44 `BOTH`, 30 `CHAT` and 25 `WEB_SEARCH` queries.

| Metric | Value |
| --- | --- |
| Intent Accuracy | **88.9%** (88/99) |
| Routing Precision (macro) | 0.884 |
| Routing Recall (macro) | 0.897 |
| Routing F1 (macro) | 0.888 |

These values measure the agreement with the adopted labels, which do not follow an explicit policy for conflicts between topic and requested channel: relabeling "cerca siti per imparare python" consistently with the rule stated in the thesis would give 87.9%. Routing metrics only cover the initial routing decision, not the full path in the graph.

### System

Single execution of the 10 benchmark queries (`response_time.py`, `response_times.csv`): average response time **4.47 s** (min 1.12 s, max 8.98 s). The average number of executed graph nodes is **3.20**.

### Deterministic controls

`test_controlli_deterministici.py` runs 13 tests on the real validation code, replacing only the LLM output with predefined text. It requires **no API keys**:

```bash
python evaluation/test_controlli_deterministici.py
```

In all cases the code behaves as implemented. The tests also show a low tolerance to format variations: identifiers returned as strings discard all resources, and correct intent labels written in lowercase or with punctuation fall back to `CHAT`.

### Web Search Agent

No quantitative metrics are reported. The thesis analyzes two documented cases, one for each path that reaches the web branch.

---

## Technologies

Python · LangGraph · OpenAI API (GPT-4o) · Sentence Transformers · scikit-learn · Pandas · SerpAPI · Streamlit · Docker

---

## Project structure

```text
multi-agent-system/
├── frontend.py
├── mytestagent.py
├── generate_embeddings.py
├── requirements.txt
├── Dockerfile
├── pages/
│   └── resource.py
├── recommender/
│   ├── simulated_ncf.py
│   ├── book_rich_metadata.csv
│   └── book_rich_metadata_embeddings.csv
└── evaluation/
    ├── benchmark.csv                      # 10-query benchmark
    ├── benchmark_v2.csv                   # 99-query intent benchmark
    ├── evaluate_recommender.py
    ├── evaluate_baseline_vs_critic.py
    ├── evaluate_intent.py
    ├── evaluate_intent_v2.py
    ├── metrics.py
    ├── routing_metrics.py
    ├── task_success_rate.py
    ├── response_time.py
    ├── test_controlli_deterministici.py
    ├── recommender_results.csv
    ├── baseline_vs_critic_results.csv
    ├── intent_results.csv
    ├── intent_results_v2.csv
    ├── routing_confusion_matrix.csv
    ├── routing_confusion_matrix_v2.csv
    ├── response_times.csv
    └── task_success_results.csv
```

---

## Installation

```bash
git clone https://github.com/ileniasalerno1/multi-agent-system.git
cd multi-agent-system
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_key
SERP_API_KEY=your_key
```

Run the application:

```bash
streamlit run frontend.py
```

The precomputed embeddings are already included in `recommender/book_rich_metadata_embeddings.csv`; run `python generate_embeddings.py` only if the dataset changes.

---

## Known limitations and future work

* when no resource is validated, the web search results are discarded instead of being shown;
* the ground truth has a single annotator and is incomplete; the benchmark is small and was run once;
* the conversation history is not used for retrieval, and the `CHAT` path returns a fixed message;
* deterministic checks compare values without normalization;
* the embedding model is monolingual English.

Planned improvements, in order: fix the system behavior listed above; build a documented ground truth with multiple annotators and a larger benchmark; normalize values before comparison and flag failed validations; run a user study; consider a vector database or a collaborative model only if the usage conditions require it.

---

## Author

**Ilenia Salerno**, Università degli Studi di Milano
