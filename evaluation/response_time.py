import sys
import os
import time
import pandas as pd

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from mytestagent import run_agent

print("===== RESPONSE TIME EVALUATION =====\n")

benchmark = pd.read_csv(
    "evaluation/benchmark.csv"
)

results = []

times = []

history = []

for _, row in benchmark.iterrows():

    start = time.perf_counter()

    run_agent(
        row["query"],
        history
    )

    end = time.perf_counter()

    elapsed = end - start

    times.append(elapsed)

    results.append({
        "query": row["query"],
        "response_time": elapsed
    })

    print("-" * 50)
    print("Query :", row["query"])
    print(f"Time  : {elapsed:.3f} s")

results_df = pd.DataFrame(results)

results_df.to_csv(
    "evaluation/response_times.csv",
    index=False
)

print("\n==============================")
print(f"Average : {sum(times)/len(times):.3f} s")
print(f"Min     : {min(times):.3f} s")
print(f"Max     : {max(times):.3f} s")

print("\nRisultati salvati in evaluation/response_times.csv")