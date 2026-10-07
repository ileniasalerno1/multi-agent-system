"""
Valutazione dell'Intent Agent sul benchmark ampliato (99 query, 3 categorie).

Da eseguire in locale, dalla cartella principale del progetto
(dove si trova mytestagent.py), con le chiavi API già configurate nel file .env.

Uso:
    python evaluation/evaluate_intent_v2.py

Output:
    evaluation/intent_results_v2.csv
    evaluation/routing_confusion_matrix_v2.csv
    (metriche stampate a schermo)
"""

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from mytestagent import intent_agent

benchmark = pd.read_csv("evaluation/benchmark_v2.csv")

results = []

for _, row in benchmark.iterrows():

    predicted = intent_agent(row["query"], [])
    expected = row["intent"]
    ok = predicted == expected

    results.append({
        "query": row["query"],
        "expected": expected,
        "predicted": predicted,
        "correct": ok
    })

    print(f"{row['query']} -> Atteso: {expected} | Predetto: {predicted} | {'OK' if ok else 'ERRORE'}")

results_df = pd.DataFrame(results)
results_df.to_csv("evaluation/intent_results_v2.csv", index=False)

expected_col = results_df["expected"]
predicted_col = results_df["predicted"]

accuracy = accuracy_score(expected_col, predicted_col)
precision = precision_score(expected_col, predicted_col, average="macro", zero_division=0)
recall = recall_score(expected_col, predicted_col, average="macro", zero_division=0)
f1 = f1_score(expected_col, predicted_col, average="macro", zero_division=0)

print("\n==============================")
print("INTENT / ROUTING METRICS (benchmark ampliato)")
print("==============================")
print(f"Intent Accuracy   : {accuracy:.3f}")
print(f"Routing Precision : {precision:.3f}")
print(f"Routing Recall    : {recall:.3f}")
print(f"Routing F1        : {f1:.3f}")

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================\n")
print(classification_report(expected_col, predicted_col, zero_division=0))

labels = sorted(set(expected_col) | set(predicted_col))
cm = confusion_matrix(expected_col, predicted_col, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

print("==============================")
print("CONFUSION MATRIX")
print("==============================")
print(cm_df)

cm_df.to_csv("evaluation/routing_confusion_matrix_v2.csv")

print("\nRisultati salvati in evaluation/intent_results_v2.csv")
print("Confusion matrix salvata in evaluation/routing_confusion_matrix_v2.csv")
