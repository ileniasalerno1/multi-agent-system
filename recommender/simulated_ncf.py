import pandas as pd
import json

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SimulatedNCF:

    def __init__(self):

        self.df = pd.read_csv(
            "recommender/book_rich_metadata_embeddings.csv"
        ).fillna("")

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.df["embedding"] = self.df[
            "embedding"
        ].apply(json.loads)

    def recommend(self, query, top_k=10):

        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True
        )

        scores = []

        for _, row in self.df.iterrows():

            resource_embedding = row["embedding"]

            similarity = cosine_similarity(
                [query_embedding],
                [resource_embedding]
            )[0][0]

            scores.append(similarity)

        self.df["score"] = scores

        results = (
            self.df[self.df["score"] >= 0.3]
            .sort_values(
                "score",
                ascending=False
            )
            .head(top_k)
        )

        records = results[
            [
                "book_id",
                "title",
                "description",
                "clean_author",
                "year",
                "avg_rating",
                "simple_category",
                "score"
            ]
        ].to_dict("records")

        for r in records:

            r["type"] = "document"

        return records