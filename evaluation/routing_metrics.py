import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("===== ROUTING EVALUATION =====\n")

# Carica i risultati dell'Intent Agent
df = pd.read_csv(
    "evaluation/intent_results.csv"
)

expected = df["expected"]
predicted = df["predicted"]

accuracy = accuracy_score(
    expected,
    predicted
)

precision = precision_score(
    expected,
    predicted,
    average="macro"
)

recall = recall_score(
    expected,
    predicted,
    average="macro"
)

f1 = f1_score(
    expected,
    predicted,
    average="macro"
)

print("==============================")
print("ROUTING METRICS")
print("==============================")

print(f"Routing Accuracy : {accuracy:.3f}")
print(f"Routing Precision: {precision:.3f}")
print(f"Routing Recall   : {recall:.3f}")
print(f"Routing F1       : {f1:.3f}")

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================\n")

print(
    classification_report(
        expected,
        predicted
    )
)

print("==============================")
print("CONFUSION MATRIX")
print("==============================")

labels = sorted(expected.unique())

cm = confusion_matrix(
    expected,
    predicted,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print(cm_df)

cm_df.to_csv(
    "evaluation/routing_confusion_matrix.csv"
)

print("\nConfusion matrix salvata in evaluation/routing_confusion_matrix.csv")