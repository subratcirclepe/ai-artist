# AI Artist Agent — Knowledge Graph-Powered Creative Engine
## Complete Project Specification

---

## PROJECT OVERVIEW

A Knowledge Graph-powered AI Agent that deeply understands and replicates the songwriting style of any Indian singer-songwriter. Unlike traditional flat RAG approaches, this system builds a comprehensive Music Knowledge Graph per artist — capturing structural patterns, rhyme schemes, emotional arcs, metaphor habits, vocabulary fingerprints, and cultural references — then uses multi-stage graph retrieval to generate lyrics that are authentically artist-specific.

**Architecture inspiration**: KuzuDB graph patterns ported from the GitNexus codebase intelligence engine, adapted for music/lyrics domain.

**Constraints**: 3-5 users. Scalability irrelevant. Depth and accuracy are the only priorities.

**Final deliverable:** A Streamlit web app where users can:
1. Select an artist and set up their Knowledge Graph (one-time)
2. Ask the agent to write songs on any topic — output passes 5-check validation
3. Chat with the artist persona backed by graph-computed identity
4. View quality scores (vocabulary, originality, rhyme compliance)
5. See reference songs with section-level attribution

---

## TECH STACK

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.11+ | Core runtime |
| LLM Framework | LangChain | LLM orchestration, multi-provider fallback |
| Graph Database | **KuzuDB** | Music Knowledge Graph (embeddable, Cypher, HNSW) |
| Vector Database | ChromaDB | Legacy flat RAG fallback |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | 384D multilingual embeddings |
| LLM (primary) | Anthropic Claude | Song generation + LLM-powered analysis |
| LLM (fallbacks) | Google Gemini, Groq, Cohere | Multi-provider retry chain |
| Community Detection | leidenalg + igraph | Thematic song clustering |
| Lyrics Source | lyricsgenius (Genius API) | Raw lyrics scraping |
| Web UI | Streamlit | Chat interface + setup flow |

---

## SYSTEM ARCHITECTURE

```
                         ┌─────────────────────────────────────────┐
                         │              User (Streamlit)            │
                         └──────────────────┬──────────────────────┘
                                            │
                                            v
                         ┌──────────────────────────────────────────┐
                         │        Agent (Intent Detection)          │
                         │     src/agent.py — routes to graph or    │
                         │     flat RAG based on data availability  │
                         └──────────────────┬───────────────────────┘
                                            │
                              ┌─────────────v──────────────┐
                              │   Stage 0: Request Analysis │
                              │   (topic, mood, structure)  │
                              └─────────────┬──────────────┘
                                            │
          ┌─────────┬──────────┬────────────┼────────────┬──────────┬──────────┐
          v         v          v            v            v          v          v
      Stage 1   Stage 2    Stage 3     Stage 4      Stage 5    Stage 6    Stage 7
      Semantic  Vocabulary  Rhyme      Emotional    Metaphor   Cultural   Structure
      Search    Patterns    Schemes    Arcs         Bank       Refs       Templates
      (hybrid)  (graph)     (graph)    (graph)      (graph)    (graph)    (graph)
          │         │          │            │            │          │          │
          └─────────┴──────────┴────────────┼────────────┴──────────┴──────────┘
                                            │
                              ┌─────────────v──────────────┐
                              │  Context Assembly & Budget  │
                              │  (~15K tokens, prioritized) │
                              └─────────────┬──────────────┘
                                            │
                              ┌─────────────v──────────────┐
                              │  Two-Part Prompt Build      │
                              │  System: identity+constraints│
                              │  User: examples+task        │
                              └─────────────┬──────────────┘
                                            │
                              ┌─────────────v──────────────┐
                              │  LLM Generation             │
                              │  (multi-provider fallback)   │
                              └─────────────┬──────────────┘
                                            │
                              ┌─────────────v──────────────┐
                              │  Validation Layer            │
                              │  vocab + rhyme + originality │
                              │  + emotional arc + structure │
                              └─────────────┬──────────────┘
                                            │
                                   ┌────────┴────────┐
                                   │  Pass?          │
                                   ├─Yes─→ Return    │
                                   └─No──→ Re-gen    │
                                     (max 3 attempts)
```

---

## DATA PIPELINE (one-time per artist)

```
Genius API → Scraper (preserves section headers [Verse], [Chorus], etc.)
    → Structural Decomposer (Song → Section → Line hierarchy)
    → Linguistic Analyzer (phrases, cultural references, meter patterns)
    → Phonetic Analyzer (rhyme pairs: perfect, slant, assonance, cross-language)
    → LLM-Powered Analysis (metaphors, themes, emotional arcs)
    → Style Fingerprint Builder (per-artist statistical DNA)
    → Leiden Clustering (thematic song communities)
    → KuzuDB Graph Ingestion (all nodes + relationships)
    → Advanced Data Ingestion (themes, metaphors, arcs, clusters, rhyme pairs)
    → Embedding Generator (Song + Section + Line embeddings with HNSW index)
```

---

## PROJECT STRUCTURE

```
ai-artist/
├── KNOWLEDGE_GRAPH_INTEGRATION_PLAN.md   # Detailed integration design document
├── AI_Artist_Agent_PROJECT_PLAN.md       # This file — project specification
├── requirements.txt
├── .env / .env.example
│
├── config/
│   ├── artists.yaml              # Artist profiles (3 pre-configured)
│   └── prompts.yaml              # Legacy static prompts (replaced by graph-computed prompts)
│
├── data/
│   ├── raw/                      # Raw scraped lyrics (JSON per artist)
│   ├── processed/                # Graph data, advanced analysis, clusters (JSON)
│   ├── vectorstore/              # ChromaDB persistent storage (fallback)
│   └── graphstore/               # KuzuDB databases (per artist)
│
├── src/
│   ├── scraper.py                # Genius API scraper (preserves section headers)
│   ├── preprocessor.py           # Legacy flat chunking (ChromaDB path)
│   ├── embeddings.py             # ChromaDB vector creation
│   ├── rag_chain.py              # Original flat RAG pipeline (fallback)
│   ├── graph_rag_chain.py        # Graph-powered RAG pipeline + setup orchestrator
│   ├── agent.py                  # Artist agent — routes graph vs flat RAG
│   ├── utils.py                  # Helpers, paths, config loaders
│   │
│   ├── graph/                    # KuzuDB graph layer
│   │   ├── schema.py             # 18 node tables, 22 relationship tables, indexes
│   │   ├── connection.py         # Singleton DB connection, query execution
│   │   ├── ingestion.py          # Structural data → graph nodes + relationships
│   │   ├── loader.py             # Embedding generation + advanced data ingestion
│   │   └── queries.py            # Centralized Cypher query templates
│   │
│   ├── analysis/                 # Analysis layer
│   │   ├── lyric_analyzer.py     # Structural decomposition + phrase/cultural extraction
│   │   ├── phonetics.py          # Hindi/English phonetic rhyme detection
│   │   ├── fingerprint.py        # LLM analysis + StyleFingerprint computation
│   │   └── thematic_clustering.py # Leiden community detection
│   │
│   ├── retrieval/                # Multi-stage retrieval
│   │   ├── pipeline.py           # 7-stage retrieval orchestrator
│   │   └── hybrid_search.py      # BM25 + semantic with RRF fusion
│   │
│   ├── prompt/                   # Dynamic prompt construction
│   │   └── assembler.py          # Two-part prompt builder from graph data
│   │
│   └── validation/               # Post-generation quality assurance
│       ├── validator.py          # 5 validation checks + scoring
│       └── regenerator.py        # Re-generation strategy (partial/full)
│
├── app/
│   ├── streamlit_app.py          # Main Streamlit application
│   ├── pages/
│   │   └── Project_Plan.py       # This plan displayed in the UI
│   ├── components/
│   │   ├── sidebar.py            # Artist selector + graph/vector status
│   │   └── chat.py               # Chat UI + validation display + references
│   └── assets/
│       └── style.css             # Custom styling
│
└── tests/
    ├── test_scraper.py
    ├── test_rag.py
    └── test_agent.py
```

---

## KNOWLEDGE GRAPH SCHEMA

### Node Types (18 types across 5 categories)

**Structural** (containment hierarchy):
| Node | Key Properties | Purpose |
|------|---------------|---------|
| `Artist` | id, name, slug, language, musicalStyle | Root entity |
| `Album` | id, name, artistId | Album grouping |
| `Song` | id, title, artistId, language, mood, fullLyrics, lineCount, sectionCount, wordCount | Song-level entity |
| `Section` | id, songId, sectionType (verse/chorus/bridge/mukhda/antara), text, lineCount, mood | Structural unit |
| `Line` | id, sectionId, songId, text, romanized, wordCount, syllableCount, language, hasCodeSwitch | Atomic lyric unit |

**Linguistic** (atoms of expression):
| Node | Key Properties | Purpose |
|------|---------------|---------|
| `Phrase` | id, text, frequency, artistId, isSignature | Recurring multi-word expressions |
| `Metaphor` | id, sourceText, sourceDomain, targetDomain, frequency | Imagery patterns (LLM-extracted) |
| `CulturalReference` | id, referenceText, category, culturalContext, frequency | Cultural touchstones |
| `RhymePair` | id, wordA, wordB, rhymeType (perfect/slant/assonance/cross_language), frequency | Rhyme vocabulary |

**Stylistic** (patterns that define an artist):
| Node | Key Properties | Purpose |
|------|---------------|---------|
| `Theme` | id, name, description, artistId, songCount, keywords[] | Thematic categories |
| `Mood` | id, name, valence (-1 to 1), arousal (0 to 1) | Emotion coordinates |
| `MeterPattern` | id, pattern, artistId, frequency | Line length/rhythm patterns |
| `StructureTemplate` | id, pattern, artistId, frequency | Song architecture patterns |

**Analytical/Computed**:
| Node | Key Properties | Purpose |
|------|---------------|---------|
| `ThematicCluster` | id, label, keywords[], description, cohesion, songCount | Leiden-detected song communities |
| `EmotionalArc` | id, songId, arcType, moodSequence[], intensitySequence[] | Per-song mood progression |
| `StyleFingerprint` | id, artistId, avgLineLength, vocabularyRichness, codeSwitchFrequency, topRhymeTypes[], metaphorDensity, repetitionIndex | Statistical artist DNA |
| `VocabularyCluster` | id, label, words[], artistId | Semantic word groups |

**Embedding**:
| Node | Key Properties | Purpose |
|------|---------------|---------|
| `LyricEmbedding` | nodeId, embedding FLOAT[384] | Vector storage with HNSW index |

### Relationship Types (22 tables)

| Type | From → To | Purpose |
|------|-----------|---------|
| `WRITTEN_BY` | Song → Artist | Authorship |
| `BELONGS_TO` | Song → Album | Album membership |
| `CONTAINS_SECTION` | Song → Section | Song structure |
| `CONTAINS_LINE` | Section → Line | Section content |
| `SECTION_FOLLOWS` | Section → Section | Sequential ordering |
| `LINE_FOLLOWS` | Line → Line | Line ordering |
| `USES_PHRASE` | Line → Phrase | Phrase usage |
| `CONTAINS_METAPHOR` | Line → Metaphor | Metaphor occurrence |
| `SONG_REFERENCES_CULTURE` | Song → CulturalReference | Cultural reference tracking |
| `LINE_REFERENCES_CULTURE` | Line → CulturalReference | Line-level cultural refs |
| `HAS_THEME` | Song → Theme | Thematic tagging |
| `SONG_EXPRESSES_MOOD` | Song → Mood | Song-level mood |
| `SECTION_EXPRESSES_MOOD` | Section → Mood | Section-level mood |
| `RHYMES_WITH` | Line → Line | Rhyme connections |
| `MEMBER_OF_CLUSTER` | Song → ThematicCluster | Leiden community membership |
| `SIMILAR_TO` | Song → Song | Computed similarity |
| `TRANSITIONS_TO` | Mood → Mood | Emotional transition patterns |
| `USES_STRUCTURE` | Song → StructureTemplate | Structural pattern usage |
| `HAS_ARC` | Song → EmotionalArc | Emotional arc link |
| `HAS_METER` | Section → MeterPattern | Meter pattern usage |
| `HAS_FINGERPRINT` | Artist → StyleFingerprint | Artist DNA link |
| `HAS_VOCAB_CLUSTER` | Artist → VocabularyCluster | Vocabulary grouping |

---

## 7-STAGE RETRIEVAL PIPELINE

**Stage 0 — Request Analysis**: Decompose user request into topic, mood signals, structural request, language preference, and thematic keywords.

**Stage 1 — Thematic Section Retrieval** (hybrid search): BM25 + semantic search on Section nodes merged via Reciprocal Rank Fusion (RRF, k=60) → top-10 sections.

**Stage 2 — Vocabulary Patterns** (graph query): Signature Phrases (frequency >= 3), vocabulary set, anti-vocabulary list from StyleFingerprint.

**Stage 3 — Rhyme Schemes** (graph query): Top RhymePair nodes by frequency, preferred rhyme patterns from fingerprint.

**Stage 4 — Emotional Arcs** (graph query): Common EmotionalArc patterns (gentle_rise, crescendo_crash, steady_melancholy, oscillating, slow_build).

**Stage 5 — Metaphor Bank** (graph query): Artist's Metaphor nodes ranked by frequency — source domain, target domain, source text.

**Stage 6 — Cultural References** (graph query): Ranked CulturalReference nodes (mythology, religion, geography, food, etc.).

**Stage 7 — Structural Templates** (graph query): Top StructureTemplate nodes + average line counts per section type.

**Caching**: Stages 2-4, 6-7 are artist-level (not topic-dependent). Only stages 1 and 5 vary per request.

---

## TWO-PART PROMPT ARCHITECTURE

Replaces the monolithic `prompts.yaml` with graph-computed prompts.

### System Prompt (~3,500 tokens — computed from graph, not hand-written)

Built dynamically from `StyleFingerprint`, `StructureTemplate`, `Phrase`, `RhymePair`, `EmotionalArc`, `Metaphor`, and `CulturalReference` nodes:

- **Core Identity**: "You are {artist_name}'s creative consciousness"
- **Structural Instincts**: Top song structures, section sizes
- **Language Rules**: Vocabulary space, anti-vocabulary, signature expressions
- **Rhyme DNA**: Preferred patterns, common rhyming pairs
- **Emotional Architecture**: Typical arc patterns, mood transitions
- **Metaphor Palette**: Signature metaphor domains
- **Cultural Anchors**: Cultural touchstones
- **Absolute Rules**: Authenticity test, no generic filler, no copying, linguistic authenticity

### User Prompt (~8,000 tokens — topic-specific)

Built from Stage 1 retrieval results (thematic sections) + generation instructions:

- **Reference Lyrics**: 3-4 diverse section examples (verse, chorus, bridge) with structural annotations
- **Generation Task**: Topic, recommended structure, emotional arc, section targets
- **Format Requirements**: Section labels, delivery directions, transliteration rules

---

## VALIDATION LAYER (5 checks)

Every generated output runs through 5 validation checks producing a `ValidationReport`:

| Check | Weight | Method | Threshold |
|-------|--------|--------|-----------|
| **Originality** | 0.30 | 4-gram overlap against all existing Line nodes | < 0.6 overlap per line |
| **Vocabulary Authenticity** | 0.25 | Word overlap with artist's vocabulary set, anti-vocab penalty | > 0.85 overlap |
| **Rhyme Compliance** | 0.15 | Adjacent + alternating line end-word phonetic matching | > 70% compliance |
| **Emotional Arc** | 0.15 | Mood keyword detection per section vs expected progression | Within 1 step |
| **Structural Compliance** | 0.15 | Section label parsing, count/type verification | Match template |

### Re-Generation Strategy
- `score >= 0.8` + no critical flags → **Accept**
- `score >= 0.6` + only 1-2 flagged lines → **Partial re-gen** (repair prompt)
- `score < 0.6` OR originality < 0.7 → **Full re-gen** with strengthened constraints
- Max 3 attempts, then return best-scoring attempt

---

## ARTIST CONFIGURATION

Pre-configured in `config/artists.yaml`:

| Artist | Language | Style | Graph Captures |
|--------|----------|-------|----------------|
| **Anuv Jain** | Hindi-English (Hinglish) | Soft acoustic, lo-fi indie folk | Rain metaphors, chai references, code-switching patterns |
| **Arijit Singh** | Hindi/Urdu (Bollywood) | Orchestral ballads, sufi rock | Urdu shayari, crescendo arcs, devotion themes |
| **Prateek Kuhad** | English and Hindi (separate) | Indie folk, bedroom pop | Domestic imagery, understated emotion, clean lyrics |

Adding a new artist: Edit `artists.yaml` → click "Setup Artist Data" in the app → the full pipeline runs automatically.

---

## SETUP AND RUN

### Prerequisites
```
GENIUS_API_TOKEN=your_genius_token    # genius.com/api-clients (free)
ANTHROPIC_API_KEY=your_anthropic_key  # console.anthropic.com
```

Optional fallback LLM keys: `GOOGLE_API_KEY`, `GROQ_API_KEY`, `COHERE_API_KEY`

### Install & Run
```bash
cd ai-artist
pip install -r requirements.txt
# Edit .env with your API keys
python -m streamlit run app/streamlit_app.py
```

### First-Time Artist Setup
1. Select an artist in the sidebar
2. Set max songs to scrape (30 recommended)
3. Check "Build Knowledge Graph" (recommended)
4. Click "Setup Artist Data"
5. Wait for the pipeline (~10-15 minutes including LLM analysis)
6. Start chatting!

### Dependencies
```
# Core
langchain, langchain-anthropic, langchain-google-genai, langchain-groq, langchain-cohere
langchain-community, langchain-huggingface
chromadb, sentence-transformers, lyricsgenius
streamlit, pandas, python-dotenv, pyyaml

# Knowledge Graph
kuzu>=0.8.0          # Graph database (embeddable, Cypher, HNSW)
igraph>=0.11.0       # Graph algorithms
leidenalg>=0.10.0    # Leiden community detection
```

---

## DUAL-MODE OPERATION

The system operates in two modes with automatic fallback:

| Mode | When | Pipeline |
|------|------|----------|
| **Graph RAG** | Knowledge Graph exists for artist | 7-stage retrieval → graph-computed prompt → LLM → 5-check validation → re-gen loop |
| **Vector RAG** | Only ChromaDB exists (no graph) | Flat similarity search → static YAML prompt → LLM → direct return |

The agent (`src/agent.py`) checks `graph_exists(artist_slug)` on every request and routes accordingly. Users see "Powered by Knowledge Graph" in the UI when graph mode is active.

---

## EXPECTED OUTPUT

**User:** Write a song about missing someone during monsoon season

**Agent (as Anuv Jain, Graph RAG mode):**

```
[Verse 1 — softly, with acoustic guitar]
Baarish ke baad woh smell aati hai na
(After the rain, that smell comes, doesn't it)
Mitti ki, yaad ki, teri baaton ki
(Of soil, of memories, of your words)
Main window pe baitha hoon chai leke
(I'm sitting by the window with chai)
Aur tu... kahin door hai
(And you... you're somewhere far away)

[Chorus — slightly louder, strumming]
Tujhe yaad karna bhi aadat hai meri
(Missing you is also my habit now)
Har baarish mein tera chehra dikhta hai
(In every rain I see your face)
...
```

**Quality Score: 87%** — Vocabulary: 91% | Originality: 94% | Rhyme: 78% | 1 attempt

**References:** Baarishein (verse), Riha (chorus), Mishri (bridge) — section-level attribution from Knowledge Graph

---

## WHAT SUCCESS LOOKS LIKE

1. Running `python -m streamlit run app/streamlit_app.py` opens the app
2. User can build a Knowledge Graph for any configured artist from the UI
3. Generated lyrics pass 5-check validation with score >= 0.8
4. Lyrics feel authentically artist-specific — not "AI-generated"
5. Vocabulary, rhyme patterns, and emotional arcs match the artist's real catalog
6. No copied/paraphrased lines from existing songs (originality check)
7. Seamless fallback to flat RAG when graph isn't available
8. Quality scores visible in the UI for every generation
