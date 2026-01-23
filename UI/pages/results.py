import os
import sys
from pathlib import Path

import requests
import streamlit as st
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from results_getter import get_results

st.set_page_config(page_title="Maple Lens • Results", layout="wide")

API_BASE = os.getenv("MAPLELENS_API_BASE", "http://localhost:8000")  # your backend

# ---------- Query ----------
query = st.session_state.get("query")
if not query:
    st.warning("No query found. Go back and enter a topic.")
    if st.button("⬅️ Back to search"):
        st.switch_page("UI/landing_page.py")
    st.stop()

def api_post(path: str, payload: dict, timeout: int = 60) -> dict:
    url = f"{API_BASE}{path}"
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False, ttl=3600)
def get_summary(query: str) -> dict:
    # returns {"summary": "...", "bullets": [...]}
    return api_post("/summarize", {"query": query})

@st.cache_data(show_spinner=False, ttl=3600)
def get_related(text: str, k: int = 10) -> dict:
    # returns {"posts": [...]}
    return api_post("/related_posts", {"text": text, "k": k})

# ---------- Header ----------
st.title(f"Results for: {query}")

if st.button("⬅️ New search"):
    st.switch_page("landing_page.py")

st.markdown("")

# ---------- Call backend ----------
with st.spinner("Generating Gemini summary..."):
    summary_res = get_summary(query)

res = get_results(query)

facts_text = res.get("facts_view", "")
community_text = res.get("community_view", "")

# ✅ use community summary (best) or fallback to query
retrieval_text = community_text.strip() or query

with st.spinner("Finding related Reddit threads..."):
    related_res = get_related(retrieval_text, k=10)

posts = related_res.get("posts", [])

# ---------- Layout ----------
left, right = st.columns(2, gap="large")

with left:
    st.subheader("✨ What the sources say")
    st.caption("Key facts extracted from URLs / sources")

    with st.container(border=True):
        if facts_text.strip():
            st.markdown(facts_text)
        else:
            # fallback to Gemini summary if facts missing
            bullets = summary_res.get("bullets", [])
            if bullets:
                for b in bullets:
                    st.write("•", b)
            else:
                st.write(summary_res.get("summary", "No facts returned."))

with right:
    st.subheader("💬 Community Takeaways (r/Canada)")
    st.caption("Reddit sentiment ")

    with st.container(border=True):
        # ✅ show community summary at top
        if community_text.strip():
            st.markdown(community_text)
            st.divider()
        else:
            st.info("No community summary returned.")

