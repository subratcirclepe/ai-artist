"""Sidebar component for artist selection and settings."""

import streamlit as st
from src.utils import load_all_artists, vectorstore_exists, graph_store_exists, slugify
from app.components.theme import render_theme_toggle


def render_sidebar() -> dict:
    """
    Render the sidebar with artist selector and settings.

    Returns:
        Dict with current settings: artist_slug, k, temperature.
    """
    with st.sidebar:
        st.title("AI Artist 2.0")
        st.caption("AI-powered songwriting engine")

        render_theme_toggle()

        st.divider()

        # Load artist configs
        artists = load_all_artists()
        artist_options = {
            slug: config["name"] for slug, config in artists.items()
        }

        # Artist selector
        st.subheader("Select Artist")
        selected_slug = st.selectbox(
            "Choose an artist",
            options=list(artist_options.keys()),
            format_func=lambda x: artist_options[x],
            key="artist_selector",
        )

        # Show artist info
        if selected_slug:
            config = artists[selected_slug]
            st.caption(f"**Style:** {config.get('musical_style', 'N/A')}")
            st.caption(f"**Language:** {config.get('language', 'N/A')}")

            # Data status
            if vectorstore_exists(selected_slug):
                if graph_store_exists(selected_slug):
                    st.success("Graph RAG ready", icon="🔗")
                else:
                    st.success("Vector RAG ready", icon="✅")
                    st.caption("Build Knowledge Graph for deeper style modeling")
            else:
                st.warning("No data yet", icon="⚠️")

        st.divider()

        # Settings
        st.subheader("Settings")

        temperature = st.slider(
            "Creativity",
            min_value=0.1,
            max_value=1.0,
            value=0.85,
            step=0.05,
            help="Higher = more creative, lower = more predictable",
        )

        k_references = st.slider(
            "Reference songs",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="How many reference lyrics to retrieve for context",
        )

        st.divider()

        # New Chat button
        if st.button("New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pop("agent", None)
            st.rerun()

        st.divider()

        # Page links
        st.page_link("pages/Project_Plan.py", label="Project Plan", icon="📋")
        st.page_link("pages/Version_History.py", label="Version History", icon="📜")

        st.divider()

        # Credits
        st.caption(
            "Built with LangChain, KuzuDB, ChromaDB, and Claude. "
            "Lyrics sourced from Genius."
        )

    return {
        "artist_slug": selected_slug,
        "temperature": temperature,
        "k": k_references,
    }
