import streamlit as st

from mytestagent import (
    run_agent,
    run_web_search
)

st.set_page_config(
    page_title="Multi-Agent System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
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
        
        if "web_results" in msg:

            st.markdown("### 🌐 Risorse Web")

            for web in msg["web_results"]:

                st.link_button(
                    label=web["title"],
                    url=web["link"],
                    use_container_width=True
                )

        if "recommendations" in msg:

            for rec in msg["recommendations"][:1]:

                st.link_button(
                    label=f"📚 {rec['title']}",
                    url=f"http://localhost:8501/resource?id={rec['book_id']}",
                    use_container_width=True
                )

        if "agents_used" in msg:

            badges = ""

            for agent in msg["agents_used"]:

                color = "#475569"

                if agent == "NCF":
                    color = "#2563eb"

                elif agent == "WEB":
                    color = "#ca8a04"

                elif agent == "CRITIC":
                    color = "#16a34a"

                badges += f"""
                <span style="
                background:{color};
                color:white;
                padding:4px 10px;
                border-radius:999px;
                margin-right:6px;
                font-size:12px;
                font-weight:bold;
                ">
                {agent}
                </span>
                """

            st.markdown(
                f"""
                <div style="margin-top:10px">
                <b>🏷️ Agenti utilizzati</b><br><br>
                {badges}
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

    if recommendations:
        print(recommendations[0])

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "recommendations": recommendations,
        "agents_used": result["agents_used"]
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

            web_result = run_web_search(
                st.session_state.last_query
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": web_result["response"],
                "web_results": web_result["web_results"],
                "agents_used": web_result["agents_used"]
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