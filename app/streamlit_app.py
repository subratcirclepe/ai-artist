"""Main Streamlit application for AI Artist 2.0."""

import sys
import random
from pathlib import Path

import streamlit as st

# Add project root to path so imports work when running via streamlit
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import ArtistAgent
from src.utils import (
    vectorstore_exists,
    load_artist_config,
    slugify,
    ensure_dirs,
    graph_store_exists,
)
from app.components.sidebar import render_sidebar
from app.components.chat import render_chat_history, render_references, render_validation, add_message
from app.components.theme import init_theme, load_css_and_theme

# ---------- Page config ----------
st.set_page_config(
    page_title="AI Artist 2.0",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme & CSS ----------
init_theme()
load_css_and_theme()

# ---------- Initialize session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_artist" not in st.session_state:
    st.session_state.current_artist = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "setup_running" not in st.session_state:
    st.session_state.setup_running = False

# ---------- Render sidebar ----------
settings = render_sidebar()
artist_slug = settings["artist_slug"]
temperature = settings["temperature"]
k_refs = settings["k"]

# ---------- Handle artist switching ----------
if st.session_state.current_artist != artist_slug:
    st.session_state.current_artist = artist_slug
    st.session_state.messages = []
    st.session_state.agent = None

# ---------- Setup flow if no vectorstore ----------
if not vectorstore_exists(artist_slug):
    artist_config = load_artist_config(artist_slug)
    st.title(f"AI Artist 2.0 — {artist_config['name']}")
    st.warning(
        f"No lyrics data found for **{artist_config['name']}**. "
        "You need to set up the artist's data first."
    )

    st.markdown(
        "This will:\n"
        "1. Scrape lyrics from Genius API\n"
        "2. Process and chunk the lyrics\n"
        "3. Create vector embeddings for RAG\n"
        "4. Build Knowledge Graph for deep artist understanding\n\n"
        "**You need a Genius API token in your `.env` file.**"
    )

    col1, col2 = st.columns(2)
    with col1:
        max_songs = st.number_input(
            "Max songs to scrape",
            min_value=5,
            max_value=100,
            value=30,
            step=5,
        )
    with col2:
        build_graph = st.checkbox(
            "Build Knowledge Graph (recommended)",
            value=True,
            help="Enables deep artist style modeling with graph-powered generation"
        )
        setup_btn = st.button(
            "Setup Artist Data",
            type="primary",
            use_container_width=True,
        )

    if setup_btn:
        try:
            with st.status("Setting up artist data...", expanded=True) as status:
                ensure_dirs()

                st.write("Scraping lyrics from Genius...")
                from src.scraper import run_scraper
                run_scraper(artist_config["genius_name"], max_songs=max_songs)
                st.write("Scraping complete!")

                st.write("Processing lyrics...")
                from src.preprocessor import run_preprocessor
                run_preprocessor(artist_slug)
                st.write("Processing complete!")

                st.write("Creating vector embeddings...")
                from src.embeddings import create_vectorstore
                create_vectorstore(artist_slug, force=True)
                st.write("Embeddings created!")

                if build_graph:
                    st.write("Building Knowledge Graph (this takes a few minutes)...")
                    from src.graph_rag_chain import setup_graph_pipeline
                    graph_stats = setup_graph_pipeline(artist_slug)
                    st.write(f"Knowledge Graph built! "
                             f"({graph_stats.get('analysis', {}).get('total_songs', 0)} songs, "
                             f"{graph_stats.get('analysis', {}).get('total_lines', 0)} lines analyzed)")

                status.update(label="Setup complete!", state="complete")

            st.success("Artist data is ready! Reloading...")
            st.rerun()

        except Exception as e:
            st.error(f"Setup failed: {e}")
            st.info(
                "Make sure your `.env` file has valid API keys:\n"
                "- `GENIUS_API_TOKEN`\n"
                "- `ANTHROPIC_API_KEY`"
            )

else:
    # ---------- Main chat interface ----------
    artist_config = load_artist_config(artist_slug)
    is_graph = graph_store_exists(artist_slug)
    st.title(f"{artist_config['name']} — Creative Twin")
    if is_graph:
        st.caption("Powered by Knowledge Graph")

    # Initialize agent
    if st.session_state.agent is None:
        st.session_state.agent = ArtistAgent(artist_slug)

    agent: ArtistAgent = st.session_state.agent

    # Auto-greeting on first visit
    if not st.session_state.messages:
        greeting = agent.get_greeting()
        add_message("assistant", greeting)

    # Render chat history
    render_chat_history()

    # Chat input
    if user_input := st.chat_input("Ask me to write a song, or just chat..."):
        add_message("user", user_input)
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            loading_messages = [
                "Composing...",
                "Writing lyrics...",
                "Finding the right chords...",
                "Feeling the melody...",
                "Channeling the artist...",
            ]
            with st.spinner(random.choice(loading_messages)):
                try:
                    result = agent.chat(
                        user_input,
                        k=k_refs,
                        temperature=temperature,
                    )

                    response = result["response"]
                    references = result.get("references", [])
                    validation = result.get("validation", None)

                    st.markdown(response)

                    if validation:
                        render_validation(validation)

                    if references:
                        render_references(references)

                    add_message("assistant", response, references)

                except Exception as e:
                    error_msg = f"Something went wrong: {e}"
                    st.error(error_msg)
                    add_message("assistant", error_msg)
