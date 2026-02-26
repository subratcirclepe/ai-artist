"""Version History page — changelog for AI Artist 2.0."""

import sys
from pathlib import Path

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.components.theme import init_theme, load_css_and_theme, render_theme_toggle

# ---------- Page config ----------
st.set_page_config(
    page_title="Version History — AI Artist 2.0",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_theme()
load_css_and_theme()

# ---------- Sidebar ----------
with st.sidebar:
    st.title("AI Artist 2.0")
    st.caption("AI-powered songwriting engine")
    render_theme_toggle()
    st.divider()
    st.page_link("streamlit_app.py", label="Back to Chat", icon="🎵")
    st.page_link("pages/Project_Plan.py", label="Project Plan", icon="📋")
    st.divider()
    st.caption(
        "Built with LangChain, KuzuDB, ChromaDB, and Claude. "
        "Lyrics sourced from Genius."
    )

# ---------- Main content ----------
st.title("Version History")
st.caption("What changed and why")
st.divider()

# ===== v2.0 =====
st.markdown(
    '<div class="version-entry">'
    '<span class="version-badge badge-current">CURRENT</span>'
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("### v2.0 — Knowledge Graph Integration")
st.caption("Complete rebuild of the retrieval and generation pipeline")

st.markdown("""
**Core Changes:**
- Replaced flat vector RAG with a **Knowledge Graph-powered pipeline** using KuzuDB
- **18 node types** and **22 relationship types** capture every dimension of an artist's style
- **7-stage parallel retrieval** (semantic search, vocabulary, rhyme schemes, emotional arcs, metaphors, cultural references, structure templates)
- **Hybrid BM25 + semantic search** with Reciprocal Rank Fusion (ported from GitNexus)
""")

st.markdown("""
**Analysis Pipeline:**
- LLM-powered linguistic analysis: metaphor extraction, theme assignment, emotional arc mapping
- Hindi/English phonetic rhyme detection (perfect, slant, assonance, cross-language)
- Leiden community detection for thematic song clustering
- Per-artist **StyleFingerprint** computation (vocabulary richness, code-switch frequency, metaphor density)
""")

st.markdown("""
**Generation Quality:**
- **Two-part dynamic prompt system** — graph-computed artist identity replaces static YAML profiles
- **5-check validation layer** with automatic re-generation (max 3 attempts)
  - Originality (4-gram overlap check)
  - Vocabulary authenticity (artist word distribution)
  - Rhyme compliance (phonetic matching)
  - Emotional arc consistency
  - Structural compliance
- Quality scores visible in the UI for every generation
""")

st.markdown("""
**UI:**
- Light/dark mode toggle
- Graph RAG status indicator in sidebar
- Quality score display with per-metric breakdown
- Section-level reference attribution
- Renamed to AI Artist 2.0
""")

st.markdown("""
**Technical Details:**
- Architecture inspired by **GitNexus** codebase intelligence engine
- KuzuDB graph database (embeddable, Cypher support, HNSW vector index)
- Dual-mode operation: Graph RAG (primary) with Vector RAG automatic fallback
- New modules: `src/graph/`, `src/analysis/`, `src/retrieval/`, `src/prompt/`, `src/validation/`
""")

st.divider()

# ===== v1.0 =====
st.markdown(
    '<div class="version-entry version-entry-past">'
    '<span class="version-badge badge-previous">v1.0</span>'
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("### v1.0 — Original RAG System")
st.caption("Foundation build with flat vector retrieval")

st.markdown("""
**Features:**
- Genius API lyrics scraping with `lyricsgenius`
- ChromaDB vector store with sentence-transformers embeddings (all-MiniLM-L6-v2, 384D)
- Flat similarity search (top-k retrieval) for style context
- Static YAML-based artist profiles and system prompts (`config/artists.yaml`, `config/prompts.yaml`)
- LangChain orchestration with multi-provider LLM fallback (Claude, Gemini, Groq, Cohere)
- Basic mood/language detection during preprocessing
- Streamlit chat interface with artist selector
- 3 pre-configured artists: Anuv Jain, Arijit Singh, Prateek Kuhad
""")

st.markdown("""
**Limitations (addressed in v2.0):**
- No structural understanding of lyrics (section headers stripped during preprocessing)
- No rhyme scheme analysis
- No metaphor or cultural reference extraction
- No vocabulary fingerprinting or anti-vocabulary enforcement
- No emotional arc tracking
- No post-generation validation — output quality was unchecked
- Single flat retrieval pass (no multi-dimensional context)
- Static artist identity prompts (hand-written, not data-driven)
""")
