import pandas as pd

print("===== TASK SUCCESS RATE =====\n")

# Carica i risultati del recommender
df = pd.read_csv("evaluation/recommender_results.csv")

total_tasks = 0
successful_tasks = 0

results = []

for _, row in df.iterrows():

    expected = []

    if pd.notna(row["expected"]) and row["expected"] != "":
        expected = str(row["expected"]).split("|")

    # Se nel benchmark non è stato definito un ground truth,
    # la query non viene considerata nel calcolo.
    if len(expected) == 0:
        continue

    retrieved = []

    if pd.notna(row["retrieved"]) and row["retrieved"] != "":
        retrieved = str(row["retrieved"]).split("|")

    success = len(set(expected) & set(retrieved)) > 0

    total_tasks += 1

    if success:
        successful_tasks += 1

    results.append({
        "query": row["query"],
        "success": success
    })

    print("-" * 50)
    print("Query   :", row["query"])
    print("Success :", "YES" if success else "NO")

tsr = successful_tasks / total_tasks if total_tasks > 0 else 0

print("\n==============================")
print(f"Successful Tasks : {successful_tasks}")
print(f"Total Tasks      : {total_tasks}")
print(f"Task Success Rate: {tsr:.2%}")

pd.DataFrame(results).to_csv(
    "evaluation/task_success_results.csv",
    index=False
)

print("\nRisultati salvati in evaluation/task_success_results.csv")