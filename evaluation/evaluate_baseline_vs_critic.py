"""
Script di valutazione: retrieval semantico puro vs retrieval + Critic Agent.

Da eseguire in locale, dalla cartella principale del progetto
(dove si trova mytestagent.py), con le chiavi API già configurate nel file .env.

Uso:
    python evaluation/evaluate_baseline_vs_critic.py

Output:
    evaluation/baseline_vs_critic_results.csv
"""

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import pandas as pd

from recommender.simulated_ncf import SimulatedNCF
from mytestagent import filter_recommendations_llm

print("=== VALUTAZIONE: BASELINE vs CRITIC AGENT ===")

benchmark = pd.read_csv("evaluation/benchmark.csv")
recommender = SimulatedNCF()

results = []

for _, row in benchmark.iterrows():

    if row["intent"] != "BOTH":
        continue

    query = row["query"]

    expected = ""
    if pd.notna(row["expected_books"]):
        expected = str(row["expected_books"])

    # --- BASELINE: retrieval semantico puro (nessun filtro) ---
    baseline_recs = recommender.recommend(query, top_k=10)
    baseline_ids = [str(r["book_id"]) for r in baseline_recs]

    # --- CON CRITIC AGENT: stesso retrieval, poi filtrato dal Critic ---
    critic_result = filter_recommendations_llm(query, baseline_recs)
    critic_ids = [
        str(r["book_id"])
        for r in critic_result["recommendations"]
    ]

    print("\n" + "-" * 60)
    print("QUERY          :", query)
    print("ATTESI         :", expected)
    print("BASELINE (raw) :", baseline_ids)
    print("CON CRITIC     :", critic_ids)

    results.append({
        "query": query,
        "expected": expected,
        "baseline_retrieved": "|".join(baseline_ids),
        "critic_retrieved": "|".join(critic_ids),
    })

results_df = pd.DataFrame(results)
results_df.to_csv(
    "evaluation/baseline_vs_critic_results.csv",
    index=False
)

print("\n==============================")
print("Valutazione completata.")
print("Risultati salvati in evaluation/baseline_vs_critic_results.csv")
print("Manda questo file a Claude per completare il capitolo di valutazione.")
