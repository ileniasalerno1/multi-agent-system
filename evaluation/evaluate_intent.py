import sys
import os
import pandas as pd

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from mytestagent import intent_agent

benchmark = pd.read_csv("evaluation/benchmark.csv")

correct = 0

results = []

for _, row in benchmark.iterrows():

    predicted = intent_agent(
        row["query"],
        []
    )

    expected = row["intent"]

    ok = predicted == expected

    if ok:
        correct += 1

    results.append({
        "query": row["query"],
        "expected": expected,
        "predicted": predicted,
        "correct": ok
    })

    print(
        f"{row['query']} -> "
        f"Atteso: {expected} | "
        f"Predetto: {predicted} | "
        f"{'OK' if ok else 'ERRORE'}"
    )

accuracy = correct / len(benchmark)

print("\n==========================")
print(f"Intent Accuracy = {accuracy:.2%}")

pd.DataFrame(results).to_csv(
    "evaluation/intent_results.csv",
    index=False
)

print("Risultati salvati in evaluation/intent_results.csv")