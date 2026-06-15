import pandas as pd
import json

from sentence_transformers import SentenceTransformer

print("Caricamento modello...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Caricamento dataset...")

df = pd.read_csv(
    "recommender/book_rich_metadata.csv"
).fillna("")

embeddings = []

for i, row in df.iterrows():

    text = (
        str(row["title"]) +
        " " +
        str(row["description"])
    )

    embedding = model.encode(
        text,
        convert_to_numpy=True
    )

    embeddings.append(
        json.dumps(
            embedding.tolist()
        )
    )

    if i % 100 == 0:

        print(
            f"{i}/{len(df)}"
        )

df["embedding"] = embeddings

df.to_csv(
    "recommender/book_rich_metadata_embeddings.csv",
    index=False
)

print("COMPLETATO")