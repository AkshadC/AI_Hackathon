import streamlit as st

st.set_page_config(page_title="Maple Lens", layout="wide")



# ---- MAIN LAYOUT ----
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("📽️ Quick Tutorial")
    # st.video("assets/tutorial.mp4")
    st.info("Tutorial video placeholder (add `assets/tutorial.mp4` later).")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.image("UI/maple_lens.png", width=120)
    st.markdown("## Reddit AI Search")
    st.caption("Search Reddit. Understand instantly.")

    st.markdown("### 🔍 Search Topic")
    query = st.text_input(
        "",
        placeholder="e.g. Canada housing crisis, grocery prices, immigration",
        key="query_input",
    )

    st.markdown("")  # spacing
    if st.button("Get Insights", use_container_width=True):
        if query.strip():
            st.session_state["query"] = query.strip()
            st.switch_page("pages/results.py")
        else:
            st.warning("Please enter a topic to search")

    st.markdown("</div>", unsafe_allow_html=True)
