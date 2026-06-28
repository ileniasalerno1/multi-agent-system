import pandas as pd
import math

def precision_at_k(expected, retrieved, k):

    retrieved_k = retrieved[:k]

    tp = len(set(expected) & set(retrieved_k))

    if len(retrieved_k) == 0:
        return 0

    return tp / len(retrieved_k)


def recall_at_k(expected, retrieved, k):

    retrieved_k = retrieved[:k]

    tp = len(set(expected) & set(retrieved_k))

    if len(expected) == 0:
        return 0

    return tp / len(expected)


def f1_score(precision, recall):

    if precision + recall == 0:
        return 0

    return 2 * precision * recall / (precision + recall)

def average_precision(expected, retrieved):

    hits = 0
    ap = 0

    for i, item in enumerate(retrieved):

        if item in expected:

            hits += 1

            ap += hits / (i + 1)

    if len(expected) == 0:
        return 0

    return ap / len(expected)


def ndcg(expected, retrieved):

    dcg = 0

    for i, item in enumerate(retrieved):

        if item in expected:

            dcg += 1 / math.log2(i + 2)

    ideal_hits = min(len(expected), len(retrieved))

    idcg = 0

    for i in range(ideal_hits):

        idcg += 1 / math.log2(i + 2)

    if idcg == 0:
        return 0

    return dcg / idcg


df = pd.read_csv(
    "evaluation/recommender_results.csv"
)

precision_scores = []
recall_scores = []
f1_scores = []

precision5_scores = []
recall5_scores = []

precision10_scores = []
recall10_scores = []

map_scores = []
ndcg_scores = []

print("\n===== RECOMMENDER EVALUATION =====\n")

for _, row in df.iterrows():

    if pd.isna(row["expected"]) or row["expected"] == "":
        continue

    expected = str(row["expected"]).split("|")
    retrieved = str(row["retrieved"]).split("|")

    precision = precision_at_k(
        expected,
        retrieved,
        len(retrieved)
    )

    recall = recall_at_k(
        expected,
        retrieved,
        len(retrieved)
    )

    f1 = f1_score(
        precision,
        recall
    )

    precision5 = precision_at_k(
        expected,
        retrieved,
        5
    )

    recall5 = recall_at_k(
        expected,
        retrieved,
        5
    )

    precision10 = precision_at_k(
        expected,
        retrieved,
        10
    )

    recall10 = recall_at_k(
        expected,
        retrieved,
        10
    )

    ap = average_precision(
        expected,
        retrieved
    )

    ndcg_score = ndcg(
        expected,
        retrieved
    )

    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    precision5_scores.append(precision5)
    recall5_scores.append(recall5)

    precision10_scores.append(precision10)
    recall10_scores.append(recall10)

    map_scores.append(ap)
    ndcg_scores.append(ndcg_score)

    print("-" * 50)
    print("Query:", row["query"])
    print("Expected :", expected)
    print("Retrieved:", retrieved)

    print(f"Precision      : {precision:.3f}")
    print(f"Recall         : {recall:.3f}")
    print(f"F1             : {f1:.3f}")
    print(f"Precision@5    : {precision5:.3f}")
    print(f"Recall@5       : {recall5:.3f}")
    print(f"Precision@10   : {precision10:.3f}")
    print(f"Recall@10      : {recall10:.3f}")
    print(f"MAP            : {ap:.3f}")
    print(f"nDCG           : {ndcg_score:.3f}")

print("\n==============================")
print("AVERAGE METRICS")
print("==============================")

print(f"Precision     : {sum(precision_scores)/len(precision_scores):.3f}")
print(f"Recall        : {sum(recall_scores)/len(recall_scores):.3f}")
print(f"F1            : {sum(f1_scores)/len(f1_scores):.3f}")

print()

print(f"Precision@5   : {sum(precision5_scores)/len(precision5_scores):.3f}")
print(f"Recall@5      : {sum(recall5_scores)/len(recall5_scores):.3f}")

print()

print(f"Precision@10  : {sum(precision10_scores)/len(precision10_scores):.3f}")
print(f"Recall@10     : {sum(recall10_scores)/len(recall10_scores):.3f}")

print()

print(f"MAP           : {sum(map_scores)/len(map_scores):.3f}")
print(f"nDCG          : {sum(ndcg_scores)/len(ndcg_scores):.3f}")