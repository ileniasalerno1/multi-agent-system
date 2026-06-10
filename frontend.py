import streamlit as st

from mytestagent import (
    run_agent,
    run_web_search
)

st.set_page_config(
    page_title="Multi-Agent System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Agent System")

# =========================
# SESSION STATE
# =========================

if "history" not in st.session_state:
    st.session_state.history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_web_search" not in st.session_state:
    st.session_state.pending_web_search = False

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# =========================
# CHAT HISTORY
# =========================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if "recommendations" in msg:

            for rec in msg["recommendations"][:1]:

                st.markdown(
                    f"""
                    <div style="
                    background-color:#1e293b;
                    color:white;
                    border:1px solid #334155;
                    padding:15px;
                    border-radius:12px;
                    margin-top:10px;
                    margin-bottom:10px;
                    ">
                    <b>📚 Risorsa consigliata</b><br><br>
                    {rec["title"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# =========================
# USER INPUT
# =========================

prompt = st.chat_input(
    "Scrivi una richiesta..."
)

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Sto elaborando la richiesta..."):

        result = run_agent(
            prompt,
            st.session_state.history
        )

    response = result["response"]
    recommendations = result["recommendations"]

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "recommendations": recommendations
    })

    if result["needs_web_search"]:

        st.session_state.pending_web_search = True
        st.session_state.last_query = prompt

    st.rerun()

# =========================
# WEB SEARCH BUTTONS
# =========================

if st.session_state.pending_web_search:

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🔍 Approfondisci sul Web",
            use_container_width=True
        ):

            web_response = run_web_search(
                st.session_state.last_query
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": web_response
            })

            st.session_state.pending_web_search = False

            st.rerun()

    with col2:

        if st.button(
            "❌ No grazie",
            use_container_width=True
        ):

            st.session_state.pending_web_search = False

            st.rerun()