"""Project Plan page — displays the AI Artist Agent project specification."""

import sys
from pathlib import Path

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------- Page config ----------
st.set_page_config(
    page_title="Project Plan — AI Artist Agent",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Load custom CSS ----------
css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.title("AI Artist Agent")
    st.caption("Write songs in any artist's style")
    st.divider()
    st.page_link("streamlit_app.py", label="Back to Chat", icon="🎵")
    st.divider()
    st.caption(
        "Built with LangChain, ChromaDB, and Claude. "
        "Lyrics sourced from Genius."
    )

# ---------- Load and display the project plan ----------
plan_path = Path(__file__).parent.parent.parent / "AI_Artist_Agent_PROJECT_PLAN.md"

st.title("📋 Project Plan")
st.caption("Complete specification for the AI Artist Agent project")

st.divider()

if plan_path.exists():
    plan_content = plan_path.read_text(encoding="utf-8")
    st.markdown(plan_content)
else:
    st.error(
        "Project plan file not found. "
        f"Expected at: `{plan_path}`"
    )
