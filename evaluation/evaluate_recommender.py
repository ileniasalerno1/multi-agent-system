import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import pandas as pd

from recommender.simulated_ncf import SimulatedNCF


print("=== VALUTAZIONE RECOMMENDER ===")

# Caricamento benchmark
benchmark = pd.read_csv("evaluation/benchmark.csv")

# Inizializzazione recommender
recommender = SimulatedNCF()

results = []

for _, row in benchmark.iterrows():

    # Valutiamo solo le query che usano il recommender
    if row["intent"] != "BOTH":
        continue

    # Recupero raccomandazioni
    recommendations = recommender.recommend(
        row["query"],
        top_k=10
    )

    # Book ID restituiti
    retrieved_ids = [
        str(r["book_id"])
        for r in recommendations
    ]

    # Gestione expected_books
    expected = ""

    if pd.notna(row["expected_books"]):
        expected = str(row["expected_books"])

    print("\n------------------------------")
    print("QUERY      :", row["query"])
    print("ATTESI     :", expected)
    print("RESTITUITI :", retrieved_ids)

    results.append({

        "query": row["query"],

        "expected": expected,

        "retrieved": "|".join(retrieved_ids)

    })

# Salvataggio risultati
results_df = pd.DataFrame(results)

results_df.to_csv(
    "evaluation/recommender_results.csv",
    index=False
)

print("\n==============================")
print("Valutazione completata.")
print("Risultati salvati in evaluation/recommender_results.csv")