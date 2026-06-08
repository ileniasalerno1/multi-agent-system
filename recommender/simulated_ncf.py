import pandas as pd

class SimulatedNCF:

    def __init__(self):

        self.df = pd.read_csv(
            "recommender/book_rich_metadata.csv"
        ).fillna("")

    def recommend(self, query, top_k=3):

        query = query.lower()

        scores = []

        educational_keywords = [
            "bambini",
            "bambino",
            "infanzia",
            "linguaggio",
            "lettura",
            "gioco",
            "giochi",
            "scuola",
            "educazione",
            "didattica"
        ]

        professional_keywords = [
            "programmazione",
            "coding",
            "software",
            "sviluppatore",
            "informatica",
            "computer",
            "lavoro",
            "carriera",
            "professionale",
            "competenze"
        ]

        educational_domain = any(
            keyword in query
            for keyword in educational_keywords
        )

        professional_domain = any(
            keyword in query
            for keyword in professional_keywords
        )

        for _, row in self.df.iterrows():

            score = 0

            text = (
                str(row.get("title", "")) + " " +
                str(row.get("description", "")) + " " +
                str(row.get("simple_category", ""))
            ).lower()

            category = str(
                row.get("simple_category", "")
            ).lower()

            # Match testuale
            for word in query.split():

                if word in text:
                    score += 5

            # Dominio educational
            if educational_domain:

                if category in [
                    "juvenile fiction",
                    "juvenile nonfiction",
                    "social science"
                ]:
                    score += 15

            # Dominio professionale
            if professional_domain:

                if category in [
                    "computers",
                    "business & economics",
                    "self-help"
                ]:
                    score += 15

            # Popolarità
            try:
                score += float(
                    row.get("popularity_score", 0)
                )
            except:
                pass

            scores.append(score)

        self.df["score"] = scores

        results = (
            self.df
            .sort_values("score", ascending=False)
            .head(top_k)
        )

        return results[
            [
                "title",
                "description",
                "simple_category",
                "score"
            ]
        ].to_dict("records")