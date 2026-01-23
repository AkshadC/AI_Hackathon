import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Maple Lens • Results", layout="wide")

# ---------- Query ----------
query = st.session_state.get("query")
if not query:
    st.warning("No query found. Go back and enter a topic.")
    if st.button("⬅️ Back to search"):
        st.switch_page("landing_page.py")
    st.stop()

# ---------- Mock data ----------
def mock_gemini_summary(q: str):
    return [
        "People are discussing what’s happening, why it matters, and who is impacted.",
        "Common themes include affordability pressures and policy debates.",
        "Expect polarized takes: anecdotes vs data-driven arguments.",
    ]

def mock_reddit_posts(q: str):
    today = datetime.now().strftime("%Y-%m-%d")
    return [
        {
            "title": f"[Discussion] {q} — what are people seeing in their city?",
            "meta": f"r/canada • {today} • ↑ 1842 • 💬 963",
            "snippet": "Users share regional experiences and debate causes and fixes.",
        },
        {
            "title": f"Explainer: key numbers and sources about {q}",
            "meta": f"r/canada • {today} • ↑ 925 • 💬 311",
            "snippet": "A data-heavy thread with links and counterarguments.",
        },
    ]

summary = mock_gemini_summary(query)
posts = mock_reddit_posts(query)

# ---------- Header ----------
st.title(f"Results for: {query}")

# ✅ back works reliably
if st.button("⬅️ New search"):
    st.switch_page("landing_page.py")

st.markdown("")

# ---------- Layout ----------
left, right = st.columns(2, gap="large")

with left:
    st.subheader("✨ Gemini Summary")
    st.caption("Placeholder — will be replaced by Gemini API")
    with st.container(border=True):
        for bullet in summary:
            st.write("•", bullet)

with right:
    st.subheader("🧵 Related Reddit Threads (r/Canada)")
    st.caption("Placeholder — will be replaced by Reddit pipeline")
    with st.container(border=True):
        for post in posts:
            st.markdown(f"**{post['title']}**")
            st.caption(post["meta"])
            st.write(post["snippet"])
            st.divider()
