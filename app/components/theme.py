"""Theme utilities for AI Artist 2.0 Streamlit app."""

import streamlit as st
from pathlib import Path


def init_theme():
    """Initialize theme in session state if not set."""
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"


def load_css_and_theme():
    """Load the custom CSS file and inject the theme class via JS."""
    css_path = Path(__file__).parent.parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    is_dark = st.session_state.get("theme", "dark") == "dark"
    action = "add" if is_dark else "remove"
    st.markdown(
        f"""<script>
        var app = window.parent.document.querySelector('.stApp');
        if (app) app.classList.{action}('dark-theme');
        </script>""",
        unsafe_allow_html=True,
    )


def render_theme_toggle():
    """Render the dark mode toggle. Call inside a `with st.sidebar:` block."""
    is_dark = st.toggle(
        "Dark mode",
        value=st.session_state.get("theme", "dark") == "dark",
    )
    new_theme = "dark" if is_dark else "light"
    if new_theme != st.session_state.get("theme", "dark"):
        st.session_state.theme = new_theme
        st.rerun()
