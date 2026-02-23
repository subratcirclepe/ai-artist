"""Main Streamlit application for AI Artist Agent."""

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
)
from app.components.sidebar import render_sidebar
from app.components.chat import render_chat_history, add_message

# ---------- Page config ----------
st.set_page_config(
    page_title="AI Artist Agent",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Load custom CSS ----------
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
    st.title(f"🎵 AI Artist Agent — {artist_config['name']}")
    st.warning(
        f"No lyrics data found for **{artist_config['name']}**. "
        "You need to set up the artist's data first."
    )

    st.markdown(
        "This will:\n"
        "1. Scrape lyrics from Genius API\n"
        "2. Process and chunk the lyrics\n"
        "3. Create vector embeddings for RAG\n\n"
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
        st.markdown("")  # spacing
        setup_btn = st.button(
            "Setup Artist Data",
            type="primary",
            use_container_width=True,
        )

    if setup_btn:
        try:
            with st.status("Setting up artist data...", expanded=True) as status:
                ensure_dirs()

                # Step 1: Scrape
                st.write("🔍 Scraping lyrics from Genius...")
                from src.scraper import run_scraper
                run_scraper(artist_config["genius_name"], max_songs=max_songs)
                st.write("✅ Scraping complete!")

                # Step 2: Preprocess
                st.write("🔧 Processing lyrics...")
                from src.preprocessor import run_preprocessor
                run_preprocessor(artist_slug)
                st.write("✅ Processing complete!")

                # Step 3: Embed
                st.write("🧠 Creating vector embeddings...")
                from src.embeddings import create_vectorstore
                create_vectorstore(artist_slug, force=True)
                st.write("✅ Embeddings created!")

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
    st.title(f"🎵 {artist_config['name']} — AI Creative Twin")

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
        # Show user message
        add_message("user", user_input)
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            loading_messages = [
                "🎵 Composing...",
                "✍️ Writing lyrics...",
                "🎸 Finding the right chords...",
                "🎹 Feeling the melody...",
                "🎶 Channeling the artist...",
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

                    st.markdown(response)

                    # Show references
                    if references:
                        # Deduplicate by song title
                        seen = set()
                        unique_refs = []
                        for ref in references:
                            title = ref.get("metadata", {}).get("song_title", "")
                            if title not in seen:
                                seen.add(title)
                                unique_refs.append(ref)

                        with st.expander(
                            f"📚 Reference songs used ({len(unique_refs)})"
                        ):
                            for ref in unique_refs:
                                meta = ref.get("metadata", {})
                                title = meta.get("song_title", "Unknown")
                                mood = meta.get("estimated_mood", "")
                                st.markdown(f"**{title}** — _{mood}_")
                                st.text(ref.get("text", "")[:150] + "...")
                                st.divider()

                    # Store in session
                    add_message("assistant", response, references)

                except Exception as e:
                    error_msg = f"Something went wrong: {e}"
                    st.error(error_msg)
                    add_message("assistant", error_msg)
