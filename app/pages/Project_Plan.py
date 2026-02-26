"""Project Plan page — AI Artist 2.0 specification in a readable tabbed layout."""

import sys
from pathlib import Path

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.components.theme import init_theme, load_css_and_theme, render_theme_toggle

# ---------- Page config ----------
st.set_page_config(
    page_title="Project Plan — AI Artist 2.0",
    page_icon="📋",
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
    st.page_link("pages/Version_History.py", label="Version History", icon="📜")
    st.divider()
    st.caption(
        "Built with LangChain, KuzuDB, ChromaDB, and Claude. "
        "Lyrics sourced from Genius."
    )

# ---------- Main content ----------
st.title("Project Plan")
st.caption("Knowledge Graph-Powered Creative Engine")
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Architecture",
    "Knowledge Graph",
    "Retrieval & Prompts",
    "Validation & Setup",
])

# ===== Tab 1: Overview =====
with tab1:
    st.markdown("""
### What is AI Artist 2.0?

A Knowledge Graph-powered AI Agent that deeply understands and replicates the songwriting style
of any Indian singer-songwriter. Unlike traditional flat RAG, this system builds a comprehensive
Music Knowledge Graph per artist — capturing structural patterns, rhyme schemes, emotional arcs,
metaphor habits, vocabulary fingerprints, and cultural references.

**Architecture inspiration:** KuzuDB graph patterns from the GitNexus codebase intelligence engine,
adapted for the music/lyrics domain.

**Constraints:** 3-5 users. Scalability irrelevant. Depth and accuracy are the only priorities.
""")

    st.markdown("### Tech Stack")
    st.markdown("""
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Graph Database | **KuzuDB** | Music Knowledge Graph (embeddable, Cypher, HNSW) |
| Vector Database | ChromaDB | Legacy flat RAG fallback |
| Embeddings | all-MiniLM-L6-v2 | 384D multilingual embeddings |
| LLM (primary) | Anthropic Claude | Song generation + LLM-powered analysis |
| LLM (fallbacks) | Gemini, Groq, Cohere | Multi-provider retry chain |
| Community Detection | leidenalg + igraph | Thematic song clustering |
| Lyrics Source | Genius API | Raw lyrics scraping |
| Web UI | Streamlit | Chat interface + setup flow |
""")

    st.markdown("### Dual-Mode Operation")
    st.markdown("""
| Mode | When | Pipeline |
|------|------|----------|
| **Graph RAG** | Knowledge Graph exists | 7-stage retrieval → graph-computed prompt → LLM → 5-check validation → re-gen loop |
| **Vector RAG** | Only ChromaDB exists | Flat similarity search → static YAML prompt → LLM → direct return |

The agent automatically detects which mode to use per artist.
""")

    st.markdown("### Pre-configured Artists")
    st.markdown("""
| Artist | Language | Style |
|--------|----------|-------|
| **Anuv Jain** | Hindi-English (Hinglish) | Soft acoustic, lo-fi indie folk |
| **Arijit Singh** | Hindi/Urdu (Bollywood) | Orchestral ballads, sufi rock |
| **Prateek Kuhad** | English and Hindi | Indie folk, bedroom pop |
""")

# ===== Tab 2: Architecture =====
with tab2:
    st.markdown("### System Architecture")
    st.code("""
                     ┌────────────────────────────────┐
                     │        User (Streamlit)         │
                     └───────────────┬────────────────┘
                                     │
                     ┌───────────────v────────────────┐
                     │   Agent (Intent Detection)      │
                     │   Routes: graph vs flat RAG     │
                     └───────────────┬────────────────┘
                                     │
                       ┌─────────────v──────────────┐
                       │  Stage 0: Request Analysis  │
                       └─────────────┬──────────────┘
                                     │
       ┌────────┬────────┬───────────┼──────────┬────────┬────────┐
       v        v        v           v          v        v        v
    Stage 1  Stage 2  Stage 3    Stage 4    Stage 5  Stage 6  Stage 7
    Semantic Vocab    Rhyme      Emotional  Metaphor Cultural Structure
    Search   Patterns Schemes    Arcs       Bank     Refs     Templates
       │        │        │           │          │        │        │
       └────────┴────────┴───────────┼──────────┴────────┴────────┘
                                     │
                       ┌─────────────v──────────────┐
                       │   Context Assembly (~15K)   │
                       ├─────────────┬──────────────┤
                       │  Two-Part Prompt Build      │
                       │  System: identity+constraints│
                       │  User: examples+task        │
                       └─────────────┬──────────────┘
                                     │
                       ┌─────────────v──────────────┐
                       │    LLM Generation           │
                       └─────────────┬──────────────┘
                                     │
                       ┌─────────────v──────────────┐
                       │    Validation Layer          │
                       │    (5 checks, max 3 retries) │
                       └─────────────┬──────────────┘
                                     │
                                  Output
""", language=None)

    st.markdown("### Data Pipeline (one-time per artist)")
    st.code("""
Genius API → Scraper (preserves section headers)
  → Structural Decomposer (Song → Section → Line)
  → Linguistic Analyzer (phrases, cultural references, meter)
  → Phonetic Analyzer (rhyme pairs: perfect, slant, assonance)
  → LLM Analysis (metaphors, themes, emotional arcs)
  → Style Fingerprint Builder (per-artist statistical DNA)
  → Leiden Clustering (thematic song communities)
  → KuzuDB Graph Ingestion (all nodes + relationships)
  → Embedding Generator (Song + Section + Line with HNSW)
""", language=None)

    st.markdown("### Project Structure")
    st.code("""
src/
  scraper.py              # Genius API (preserves section headers)
  preprocessor.py         # Legacy flat chunking
  embeddings.py           # ChromaDB vectors
  rag_chain.py            # Original flat RAG (fallback)
  graph_rag_chain.py      # Graph-powered RAG + setup orchestrator
  agent.py                # Routes graph vs flat RAG

  graph/                  # KuzuDB graph layer
    schema.py             # 18 node tables, 22 relationship tables
    connection.py         # Singleton DB connection
    ingestion.py          # Structural data → graph
    loader.py             # Embeddings + advanced data
    queries.py            # Cypher query templates

  analysis/               # Analysis layer
    lyric_analyzer.py     # Structural decomposition + phrase extraction
    phonetics.py          # Hindi/English rhyme detection
    fingerprint.py        # LLM analysis + StyleFingerprint
    thematic_clustering.py # Leiden community detection

  retrieval/              # Multi-stage retrieval
    pipeline.py           # 7-stage orchestrator
    hybrid_search.py      # BM25 + semantic with RRF

  prompt/                 # Dynamic prompt construction
    assembler.py          # Two-part prompt from graph data

  validation/             # Post-generation QA
    validator.py          # 5 validation checks
    regenerator.py        # Re-generation strategy
""", language=None)

# ===== Tab 3: Knowledge Graph =====
with tab3:
    st.markdown("### Node Types (18)")

    st.markdown("**Structural** (containment hierarchy)")
    st.markdown("""
| Node | Key Properties | Purpose |
|------|---------------|---------|
| `Artist` | id, name, slug, language | Root entity |
| `Album` | id, name, artistId | Album grouping |
| `Song` | id, title, mood, lineCount, wordCount | Song-level entity |
| `Section` | id, sectionType, text, lineCount, mood | Structural unit (verse/chorus/bridge) |
| `Line` | id, text, romanized, syllableCount, hasCodeSwitch | Atomic lyric unit |
""")

    st.markdown("**Linguistic** (atoms of expression)")
    st.markdown("""
| Node | Key Properties | Purpose |
|------|---------------|---------|
| `Phrase` | text, frequency, isSignature | Recurring expressions |
| `Metaphor` | sourceText, sourceDomain, targetDomain | Imagery patterns |
| `CulturalReference` | referenceText, category, culturalContext | Cultural touchstones |
| `RhymePair` | wordA, wordB, rhymeType, frequency | Rhyme vocabulary |
""")

    st.markdown("**Stylistic** (defining patterns)")
    st.markdown("""
| Node | Key Properties | Purpose |
|------|---------------|---------|
| `Theme` | name, description, songCount | Thematic categories |
| `Mood` | name, valence, arousal | Emotion coordinates |
| `MeterPattern` | pattern, frequency | Rhythm patterns |
| `StructureTemplate` | pattern, frequency | Song architecture |
""")

    st.markdown("**Analytical/Computed**")
    st.markdown("""
| Node | Key Properties | Purpose |
|------|---------------|---------|
| `ThematicCluster` | label, keywords, cohesion | Leiden-detected communities |
| `EmotionalArc` | arcType, moodSequence | Per-song mood progression |
| `StyleFingerprint` | avgLineLength, vocabularyRichness, codeSwitchFrequency | Statistical artist DNA |
| `VocabularyCluster` | label, words | Semantic word groups |
| `LyricEmbedding` | nodeId, embedding FLOAT[384] | Vector storage with HNSW |
""")

    st.markdown("### Relationship Types (22)")
    st.markdown("""
Key relationships include:

- **Structural:** `WRITTEN_BY`, `BELONGS_TO`, `CONTAINS_SECTION`, `CONTAINS_LINE`, `SECTION_FOLLOWS`, `LINE_FOLLOWS`
- **Linguistic:** `USES_PHRASE`, `CONTAINS_METAPHOR`, `REFERENCES_CULTURE`, `RHYMES_WITH`
- **Stylistic:** `HAS_THEME`, `EXPRESSES_MOOD`, `USES_STRUCTURE`, `HAS_ARC`, `HAS_METER`
- **Analytical:** `MEMBER_OF_CLUSTER`, `SIMILAR_TO`, `TRANSITIONS_TO`, `HAS_FINGERPRINT`
""")

# ===== Tab 4: Retrieval & Prompts =====
with tab4:
    st.markdown("### 7-Stage Retrieval Pipeline")
    st.markdown("""
| Stage | Source | What It Retrieves |
|-------|--------|-------------------|
| **0. Request Analysis** | User input | Topic, mood signals, thematic keywords |
| **1. Thematic Search** | Hybrid (BM25 + semantic + RRF) | Top-10 similar sections |
| **2. Vocabulary** | Graph query | Signature phrases, vocabulary set, anti-vocabulary |
| **3. Rhyme Schemes** | Graph query | Top rhyme pairs, preferred patterns |
| **4. Emotional Arcs** | Graph query | Common arc patterns |
| **5. Metaphors** | Graph query | Artist's metaphor domains |
| **6. Cultural Refs** | Graph query | Ranked cultural references |
| **7. Structure** | Graph query | Top structure templates, avg line counts |

Stages 2-4, 6-7 are artist-level (cached). Only stages 1 and 5 vary per request.
""")

    st.markdown("### Two-Part Prompt Architecture")
    st.markdown("""
**System Prompt** (~3,500 tokens) — computed from graph, not hand-written:
- Core identity from `StyleFingerprint`
- Structural instincts from `StructureTemplate` nodes
- Language rules from `Phrase` and vocabulary data
- Rhyme DNA from `RhymePair` nodes
- Emotional architecture from `EmotionalArc` nodes
- Metaphor palette from `Metaphor` nodes
- Cultural anchors from `CulturalReference` nodes
- Absolute rules (authenticity, no copying, linguistic accuracy)

**User Prompt** (~8,000 tokens) — topic-specific:
- 3-4 diverse reference sections (verse, chorus, bridge) with structural annotations
- Generation task with recommended structure and emotional arc
- Format requirements (section labels, transliteration, delivery directions)
""")

    st.markdown("### Hybrid Search: BM25 + Semantic with RRF")
    st.markdown("""
Ported from GitNexus's hybrid search implementation:
- **BM25** keyword search on Section/Line text fields
- **Semantic** search using cosine similarity on `LyricEmbedding` (384D vectors)
- **Reciprocal Rank Fusion** (k=60) merges both ranked lists into a single score
""")

# ===== Tab 5: Validation & Setup =====
with tab5:
    st.markdown("### 5-Check Validation Layer")
    st.markdown("""
| Check | Weight | Method | Catches |
|-------|--------|--------|---------|
| **Originality** | 0.30 | 4-gram overlap against all existing lines | Copied/paraphrased lyrics |
| **Vocabulary** | 0.25 | Word overlap with artist's vocabulary set | Wrong-register words |
| **Rhyme** | 0.15 | Adjacent + alternating line end-word matching | Broken rhyme schemes |
| **Emotional Arc** | 0.15 | Mood keyword detection per section | Mood inconsistency |
| **Structure** | 0.15 | Section label parsing, count verification | Wrong song structure |
""")

    st.markdown("### Re-Generation Strategy")
    st.markdown("""
- **Score >= 0.8** + no critical flags → Accept
- **Score >= 0.6** + only 1-2 flagged lines → Partial re-gen (repair prompt)
- **Score < 0.6** or originality < 0.7 → Full re-gen with strengthened constraints
- Maximum 3 attempts, then return best-scoring attempt
""")

    st.markdown("### Setup Instructions")
    st.markdown("""
**Prerequisites:**
```
GENIUS_API_TOKEN=your_genius_token
ANTHROPIC_API_KEY=your_anthropic_key
```

**Install & Run:**
```bash
pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

**First-Time Artist Setup:**
1. Select an artist in the sidebar
2. Set max songs to scrape (30 recommended)
3. Check "Build Knowledge Graph"
4. Click "Setup Artist Data"
5. Wait ~10-15 minutes for the full pipeline
6. Start chatting!
""")
