import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Risorsa",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

params = st.query_params

if "id" not in params:

    st.error("Risorsa non trovata")
    st.stop()

book_id = int(params["id"])

df = pd.read_csv(
    "recommender/book_rich_metadata.csv"
).fillna("")

resource = df[
    df["book_id"] == book_id
]

if len(resource) == 0:

    st.error("Risorsa non trovata")
    st.stop()

resource = resource.iloc[0]

st.markdown(
    f"# 📚 {resource['title']}"
)

st.markdown(
    f"**Autore:** {resource['clean_author']}"
)

st.markdown(
    f"**Anno:** {resource['year']}"
)

st.markdown(
    f"**Rating medio:** {resource['avg_rating']}"
)

st.markdown("---")

st.write(resource["description"])