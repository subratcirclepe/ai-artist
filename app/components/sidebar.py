"""Sidebar component for artist selection and settings."""

import streamlit as st
from src.utils import load_all_artists, vectorstore_exists, slugify


def render_sidebar() -> dict:
    """
    Render the sidebar with artist selector and settings.

    Returns:
        Dict with current settings: artist_slug, k, temperature.
    """
    with st.sidebar:
        st.title("AI Artist Agent")
        st.caption("Write songs in any artist's style")

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

            # Vectorstore status
            if vectorstore_exists(selected_slug):
                st.success("Data ready", icon="✅")
            else:
                st.warning("No data yet — click Setup below", icon="⚠️")

        st.divider()

        # Settings
        st.subheader("Settings")

        temperature = st.slider(
            "Creativity (temperature)",
            min_value=0.1,
            max_value=1.0,
            value=0.85,
            step=0.05,
            help="Higher = more creative, lower = more predictable",
        )

        k_references = st.slider(
            "Reference songs to use",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="How many reference lyrics chunks to retrieve for context",
        )

        st.divider()

        # New Chat button
        if st.button("New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pop("agent", None)
            st.rerun()

        st.divider()

        # Project Plan link
        st.page_link("pages/Project_Plan.py", label="View Project Plan", icon="📋")

        st.divider()

        # Credits
        st.caption(
            "Built with LangChain, ChromaDB, and Claude. "
            "Lyrics sourced from Genius."
        )

    return {
        "artist_slug": selected_slug,
        "temperature": temperature,
        "k": k_references,
    }
