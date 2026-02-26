# Integration Plan: Knowledge Graph from GitNexus into AI Artist

## Context

**Problem**: The ai-artist system currently generates lyrics using flat vector-based RAG (ChromaDB similarity search on 200-word chunks). This produces output that, while topically relevant, lacks deep structural understanding of an artist's writing patterns — rhyme schemes, emotional arcs, metaphor habits, vocabulary fingerprints, and song architecture. The result is lyrics that sound "AI-generated" rather than authentically artist-specific.

**Goal**: Build the most accurate small-scale AI artist engine possible by integrating a Music Knowledge Graph (inspired by GitNexus's architecture) into ai-artist. This replaces the flat retrieval with a graph-powered, multi-dimensional understanding of each artist's creative DNA.

**Constraints**: 3-5 users only. Scalability irrelevant. Depth and accuracy are the only priorities.

---

## Current State Summary

### ai-artist (Python)
- **Pipeline**: Genius API scraping → preprocessing (200-word chunks, basic mood/language detection) → ChromaDB embeddings (all-MiniLM-L6-v2, 384D) → flat similarity search (top-k) → single system prompt with static YAML artist profile → LLM generation (Claude primary, Gemini/Groq/Cohere fallbacks)
- **Key files**: `src/scraper.py`, `src/preprocessor.py`, `src/embeddings.py`, `src/rag_chain.py`, `src/agent.py`, `config/artists.yaml`, `config/prompts.yaml`
- **Critical gap**: No structural decomposition of lyrics (section headers stripped), no rhyme analysis, no metaphor extraction, no vocabulary fingerprinting, no emotional arc tracking, no post-generation validation

### GitNexus (TypeScript) — Architecture Reference
- **Graph DB**: KuzuDB with typed node tables, single `CodeRelation` table with `type` property, separate `CodeEmbedding` table (FLOAT[384] + HNSW index)
- **Ingestion**: Tree-sitter AST parsing → import/call/heritage resolution → Leiden community detection → process tracing → CSV bulk loading into KuzuDB
- **Search**: Hybrid BM25 + semantic with Reciprocal Rank Fusion (RRF, k=60)
- **Key patterns to port**: KuzuDB schema design, CSV bulk loading, Leiden clustering for thematic communities, hybrid search with RRF, embedding table pattern

---

## 1. High-Level System Architecture

```
                         ┌─────────────────────────────────────────┐
                         │              User (Streamlit)            │
                         └──────────────────┬──────────────────────┘
                                            │
                                            v
                         ┌──────────────────────────────────────────┐
                         │        Agent (Intent Detection)          │
                         │          src/agent.py (modified)         │
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
                              │  + style + structure checks  │
                              └─────────────┬──────────────┘
                                            │
                                   ┌────────┴────────┐
                                   │  Pass?          │
                                   ├─Yes─→ Return    │
                                   └─No──→ Re-gen    │
                                     (max 3 attempts)
```

**Data Pipeline (one-time per artist, replaces current flat pipeline)**:
```
Genius API → Scraper (preserve section headers)
    → Structural Decomposer (Song → Section → Line nodes)
    → Linguistic Analyzer (phrases, metaphors, cultural refs, rhymes via LLM batch)
    → Style Analyzer (themes, emotional arcs, structure templates)
    → Leiden Clustering (thematic communities)
    → Fingerprint Builder (per-artist StyleFingerprint)
    → Embedding Generator (Section + Line + Song embeddings)
    → KuzuDB Bulk Load (CSV → COPY)
```

---

## 2. Graph Schema Design

### Node Types (19 types, 5 categories)

**Structural** (containment hierarchy — analogous to GitNexus's Project > File > Function):
| Node | Key Properties | Purpose |
|------|---------------|---------|
| `Artist` | id, name, slug, language, musicalStyle, vocalStyle, vocabularyLevel | Root entity |
| `Album` | id, name, releaseDate, artistId | Album grouping |
| `Song` | id, title, artistId, albumId, language, mood, fullLyrics, url, lineCount, sectionCount, wordCount | Song-level entity |
| `Section` | id, songId, sectionType (verse/chorus/bridge/mukhda/antara), sectionIndex, text, lineCount, mood | Structural unit |
| `Line` | id, sectionId, songId, lineIndex, text, romanized, wordCount, syllableCount, language, hasCodeSwitch | Atomic lyric unit |

**Linguistic** (atoms of expression):
| Node | Key Properties | Purpose |
|------|---------------|---------|
| `Phrase` | id, text, romanized, language, frequency, artistId, isSignature | Repeated expressions |
| `Metaphor` | id, sourceText, sourceDomain, targetDomain, artistId, frequency | Imagery patterns |
| `CulturalReference` | id, referenceText, category, culturalContext, artistId, frequency | Cultural touchstones |
| `RhymePair` | id, wordA, wordB, rhymeType (perfect/slant/assonance/internal/cross_language), artistId, frequency | Rhyme vocabulary |

**Stylistic** (patterns that define an artist):
| Node | Key Properties | Purpose |
|------|---------------|---------|
| `Theme` | id, name, description, artistId, songCount, keywords[] | Thematic categories |
| `Mood` | id, name, valence (-1 to 1), arousal (0 to 1) | Emotion coordinates |
| `MeterPattern` | id, pattern, artistId, frequency | Line length/rhythm patterns |
| `StructureTemplate` | id, pattern, artistId, frequency | Song architecture patterns |

**Analytical/Computed** (derived from graph analysis, analogous to GitNexus's Community and Process nodes):
| Node | Key Properties | Purpose |
|------|---------------|---------|
| `ThematicCluster` | id, label, heuristicLabel, keywords[], description, enrichedBy, cohesion, songCount, artistId | Leiden-detected song communities |
| `EmotionalArc` | id, songId, arcType, moodSequence[], intensitySequence[] | Per-song mood progression |
| `StyleFingerprint` | id, artistId, avgLineLength, avgSectionLength, vocabularyRichness, codeSwitchFrequency, topRhymeTypes[], preferredStructures[], metaphorDensity, repetitionIndex, avgMoodValence, avgMoodArousal | Statistical artist DNA |
| `VocabularyCluster` | id, label, words[], artistId, frequency, distinctiveness | Semantic word groups |

**Embedding** (separate table, following GitNexus's CodeEmbedding pattern):
| Node | Key Properties | Purpose |
|------|---------------|---------|
| `LyricEmbedding` | nodeId, embedding FLOAT[384] | Vector storage with HNSW index |

### Relationship Types (16 types, single `LyricRelation` table)

Following GitNexus's single `CodeRelation` table with `type` property:

| Type | From → To | Properties | Purpose |
|------|-----------|-----------|---------|
| `WRITTEN_BY` | Song → Artist | — | Authorship |
| `BELONGS_TO` | Song → Album | trackNumber | Album membership |
| `CONTAINS_SECTION` | Song → Section | — | Song structure |
| `CONTAINS_LINE` | Section → Line | — | Section content |
| `FOLLOWS` | Section → Section, Line → Line | gap | Sequential ordering |
| `USES_PHRASE` | Line → Phrase | position | Phrase usage |
| `CONTAINS_METAPHOR` | Line → Metaphor | — | Metaphor occurrence |
| `REFERENCES_CULTURE` | Line/Song → CulturalReference | — | Cultural reference tracking |
| `HAS_THEME` | Song → Theme | strength (0-1) | Thematic tagging |
| `EXPRESSES_MOOD` | Section/Song → Mood | intensity (0-1) | Mood at section & song level |
| `RHYMES_WITH` | Line → Line | rhymeType, pairId | Rhyme connections |
| `MEMBER_OF_CLUSTER` | Song → ThematicCluster | — | Leiden community membership |
| `SIMILAR_TO` | Song → Song, Line → Line | score, basis | Computed similarity |
| `TRANSITIONS_TO` | Mood → Mood | frequency, artistId | Emotional transition patterns |
| `USES_STRUCTURE` | Song → StructureTemplate | — | Structural pattern usage |
| `HAS_ARC` | Song → EmotionalArc | — | Emotional arc link |

### Embedding Strategy

| Node Type | What Gets Embedded | Rationale |
|-----------|-------------------|-----------|
| Song | `"Song: {title} by {artist}. Theme: {themes}. Mood: {mood}.\n{first 300 chars}"` | Top-level semantic search |
| Section | `"Section ({type}) from {song_title} by {artist}:\n{section_text}"` | Section-level retrieval |
| Line | Raw text | Fine-grained metaphor/phrase matching |
| Theme | `"Theme: {name}. Keywords: {keywords}"` | Theme-aware retrieval |
| ThematicCluster | `"Cluster: {label}. Keywords: {keywords}"` | Cluster-level search |

**Model**: Keep `all-MiniLM-L6-v2` (384D, multilingual, already in use). Drop-in upgrade path: `paraphrase-multilingual-MiniLM-L12-v2` if Hindi/Urdu quality insufficient.

**Index**: HNSW on LyricEmbedding with cosine metric + BM25 FTS on Song.fullLyrics, Section.text, Line.text/romanized.

**Search**: Hybrid BM25 + semantic merged via Reciprocal Rank Fusion (RRF, k=60) — ported from GitNexus's `hybrid-search.ts`.

---

## 3. Data Ingestion Pipeline (6 stages)

### Stage 1: Scraping (modify existing `src/scraper.py`)
- **Change**: Stop stripping section headers `[Verse 1]`, `[Chorus]`, etc. Currently `clean_lyrics()` removes them with `re.sub(r"\[.*?\]", "", lyrics)` — we need to PRESERVE them as structural data.
- Keep everything else (Genius API, raw JSON format).

### Stage 2: Structural Decomposition (new `src/graph/ingestion.py`)
1. Parse section headers to create Section nodes (`[Verse 1]` → sectionType="verse", sectionIndex=1)
2. Split sections into Line nodes (by newline)
3. Create containment edges: Song → CONTAINS_SECTION → Section → CONTAINS_LINE → Line
4. Create FOLLOWS edges for sequential ordering
5. Detect language per line (reuse existing `detect_language()`)
6. Detect code-switching within lines (Hindi/English mix in single line)
7. Compute syllable counts, word counts per line

### Stage 3: Linguistic Analysis (new `src/analysis/lyric_analyzer.py`)
1. **Phrase extraction**: N-gram frequency analysis across artist catalog → Phrase nodes for recurring multi-word expressions. Mark as `isSignature=True` if frequency significantly higher for this artist vs general corpus.
2. **Metaphor extraction**: LLM batch analysis — send 10-15 song sections at a time to Claude with prompt: "Identify metaphors in these lyrics. For each, give: source text, source domain, target domain." → Metaphor nodes.
3. **Cultural reference detection**: Keyword matching (expand current MOOD_KEYWORDS pattern) + LLM for ambiguous cases → CulturalReference nodes.
4. **Rhyme detection**: Extract line-ending words, compute phonetic similarity using `epitran` (IPA transliteration for Hindi/Urdu) or fallback regex-based syllable matching → RhymePair nodes + RHYMES_WITH edges.
5. **Meter patterns**: Compute syllable-count sequences per section → MeterPattern nodes.

### Stage 4: Stylistic Analysis (new `src/analysis/style_analyzer.py`)
1. **Theme assignment**: Expand current keyword-based `estimate_mood()` with LLM-assisted theme tagging (batch) → Theme nodes + HAS_THEME edges.
2. **Emotional arc computation**: Assign mood to each section → compute valence/arousal sequence → classify arc shape (gentle_rise, crescendo_crash, steady_melancholy, oscillating, slow_build) → EmotionalArc nodes.
3. **Structure template detection**: Extract section-type sequences per song → StructureTemplate nodes.
4. **StyleFingerprint computation**: Aggregate statistics across all songs (avgLineLength, vocabularyRichness, codeSwitchFrequency, metaphorDensity, topRhymeTypes, preferredStructures, repetitionIndex, etc.) → StyleFingerprint node per artist.

### Stage 5: Community Detection (new `src/analysis/thematic_clustering.py`)
- Port GitNexus's Leiden algorithm pipeline (`community-processor.ts`) to Python using `leidenalg` + `igraph`
- Build graph from: HAS_THEME co-occurrence, SIMILAR_TO edges, shared Phrase/Metaphor usage
- Run Leiden → create ThematicCluster nodes + MEMBER_OF_CLUSTER edges
- LLM enrichment for cluster labels (port GitNexus's `cluster-enricher.ts` pattern)

### Stage 6: Embedding Generation & DB Load (new `src/graph/loader.py`)
1. Generate embedding text for each embeddable node type
2. Batch embed using all-MiniLM-L6-v2
3. Generate CSVs for all node and relationship types (GitNexus CSV bulk loading pattern)
4. Initialize KuzuDB schema
5. COPY CSV data into KuzuDB
6. Create HNSW vector index on LyricEmbedding
7. Create BM25 FTS indexes on text fields

---

## 4. Retrieval Strategy: Hybrid (Semantic Search + Graph Traversal)

**Why hybrid**: Semantic search answers "what songs are about this topic?" — Graph traversal answers "how does this artist write songs?" Both are necessary. Neither is sufficient alone.

### Multi-Stage Retrieval Pipeline (7 parallel stages)

**Stage 0 — Request Analysis**: Decompose user request into structured facets:
```python
@dataclass
class RequestAnalysis:
    topic: str                    # "monsoon longing"
    mood_signals: list[str]       # ["melancholic", "nostalgic"]
    structural_request: str       # "full_song" | "verse" | "chorus"
    language_preference: str      # "auto" | "hindi" | "english" | "hinglish"
    thematic_keywords: list[str]  # ["rain", "barish", "yaad"]
```

**Stage 1 — Thematic Section Retrieval** (hybrid search): BM25 + semantic on Section nodes → top-10 sections with section types and moods.

**Stage 2 — Vocabulary Patterns** (graph): Top-20 VocabularyCluster nodes, signature Phrases, anti-vocabulary list from StyleFingerprint.

**Stage 3 — Rhyme Schemes** (graph): Per-section-type rhyme scheme distributions, frequent RhymePair nodes.

**Stage 4 — Emotional Arcs** (graph): Common EmotionalArc patterns, mood transition frequencies (TRANSITIONS_TO edges).

**Stage 5 — Metaphor Bank** (graph): Artist's Metaphor nodes filtered by relevance to requested mood/theme domains.

**Stage 6 — Cultural References** (graph): Ranked CulturalReference nodes for the artist.

**Stage 7 — Structural Templates** (graph): Top StructureTemplate nodes, average line counts per section type.

**Caching**: Stages 2-4, 6-7 are artist-level (not topic-dependent) — cache per artist. Only stages 1 and 5 are topic-dependent.

**Execution**: All 7 stages run in parallel via `asyncio.gather()`.

---

## 5. Prompt Orchestration Design

### Two-Part Architecture (replaces monolithic prompts.yaml)

**System Prompt** (~3,500 tokens, static per artist — computed from graph, not hand-written YAML):
```
You are {artist_name}'s creative consciousness.

## YOUR CREATIVE DNA
{distilled identity from StyleFingerprint}

### Structural Instincts
Top structures: {from StructureTemplate nodes}
Verse: ~{avg} lines. Chorus: ~{avg} lines.

### Language Rules
- Vocabulary space: {from VocabularyCluster nodes}
- NEVER use: {anti_vocabulary from StyleFingerprint}
- Unique expressions: {from signature Phrase nodes}

### Rhyme DNA
Verse schemes: {from RhymeScheme data}
Chorus schemes: {from RhymeScheme data}
Common pairs: {from top RhymePair nodes}

### Emotional Architecture
Typical arcs: {from EmotionalArc nodes}
Transition patterns: {from TRANSITIONS_TO edges}

### Metaphor Palette
Signature metaphors: {from Metaphor nodes}
Cultural anchors: {from CulturalReference nodes}

## ABSOLUTE RULES
1. Every line must pass: "Would {artist_name} actually write this?"
2. NEVER use generic filler phrases
3. NEVER copy or closely paraphrase reference lyrics
4. Match the EXACT emotional register
```

**User Prompt** (~8,000 tokens, topic-specific):
```
## REFERENCE LYRICS (absorb STYLE, never copy WORDS)

### Example Verse (from "{song_title_1}"):
{section_text}
[Structure: {lines} lines, {rhyme_scheme}, mood: {mood}, avg {words}/line]

### Example Chorus (from "{song_title_2}"):
{section_text}
[Structure annotation]

### Example Bridge (from "{song_title_3}"):
{section_text}
[Structure annotation]

## YOUR TASK
Write an original song about "{topic}" as {artist_name}.
Structure: {recommended_structure}
Emotional arc: {start_mood} → {mid_mood} → {end_mood}
Rhyme scheme: {recommended per section}
Target: {target_lines} lines across {target_sections} sections
```

### Token Budget (~15K total context):
| Section | Budget | Priority |
|---------|--------|----------|
| Artist Identity Core | 500 | Always |
| Structural Blueprint | 800 | Always |
| Style Constraints | 1,200 | Always |
| Reference Sections (few-shot) | 6,000 | Always |
| Metaphor & Imagery | 800 | P4 |
| Cultural Vocabulary | 500 | P5 |
| Anti-Patterns | 500 | P3 |
| Generation Instructions | 1,500 | Always |
| Reserve (chat history) | ~3,200 | If chat |

Trim from lowest priority up if over budget.

---

## 6. Validation Layer

### Post-Generation Pipeline

Every output runs through 5 checks producing a `ValidationReport`:

| Check | Method | Threshold | What It Catches |
|-------|--------|-----------|-----------------|
| **Vocabulary Authenticity** | Tokenize output, check each word against artist's vocabulary distribution in graph | >0.85 overlap | Generic/wrong-register words |
| **Originality** | 4-gram overlap check + embedding similarity (>0.92 cosine) against all Line nodes | <0.6 overlap per line | Copied/paraphrased lyrics |
| **Rhyme Compliance** | Extract end-words, phonetic comparison, verify against specified scheme | >70% compliance | Broken rhyme schemes |
| **Emotional Arc** | Lightweight mood classifier on each section, compare to specified arc | Within 1 step | Mood inconsistency |
| **Structural Compliance** | Parse section labels, verify counts/types against specification | Exact match | Wrong song structure |

### Re-Generation Strategy
- `score >= 0.8` + no critical flags → **Accept**
- `score >= 0.6` + only 1-2 flagged lines → **Partial re-gen** (keep good sections, rewrite flagged ones)
- `score < 0.6` OR originality < 0.7 OR vocabulary < 0.6 → **Full re-gen** with strengthened constraints + failed output as negative example
- Max 3 attempts, then return best-scoring attempt with quality warning

---

## 7. Evaluation Framework

### Automated Metrics
| Metric | How Measured | Target |
|--------|-------------|--------|
| Vocabulary Overlap Score | `\|output_words ∩ artist_words\| / \|output_words\|` | >0.85 |
| Rhyme Scheme Accuracy | % lines following specified scheme | >70% |
| Structure Compliance | Binary: matches template? | 100% |
| Embedding Style Distance | Cosine similarity to artist's embedding centroid | >0.75 |
| Originality Score | 1 - (flagged_lines / total_lines) | >0.90 |
| Metaphor Density | metaphors_used / 100 words vs artist's typical density | Within ±30% |

### Human Evaluation (for the 3-5 users)
- **Blind A/B test**: Old system (flat RAG) vs new system (graph-powered) — same artist, same topic
- **Authenticity rating**: "On a scale of 1-5, how much does this sound like {artist}?"
- **Originality rating**: "Does this feel original or copied?"
- **Emotional resonance**: "Does the emotional arc feel natural?"

---

## 8. Bottlenecks and Trade-offs

| Issue | Impact | Mitigation |
|-------|--------|------------|
| **LLM batch analysis cost** (metaphor extraction, theme tagging, cluster enrichment) | One-time per artist ingestion; ~50 API calls per artist | Acceptable for 3-5 users; cache results |
| **Hindi/Urdu phonetic analysis** for rhyme detection | No production-grade Hindi phonetic library | Use `epitran` IPA + fallback to regex syllable matching |
| **Section header parsing** from Genius lyrics | Genius format is inconsistent | Regex patterns + LLM fallback for ambiguous cases |
| **KuzuDB Python bindings maturity** | Less ecosystem than Neo4j | Acceptable trade-off: embeddable (no server), Cypher support, HNSW built-in |
| **Graph staleness** when new songs are released | Graph needs re-ingestion | Manual trigger acceptable for 3-5 users |
| **Metaphor extraction accuracy** | LLM may miss subtle metaphors or hallucinate | Human review during initial ingestion; iterative improvement |
| **Ingestion time** per artist (~50 songs) | Estimated 10-15 minutes including LLM calls | One-time cost, acceptable |

---

## 9. New Module Structure

```
src/
  # Modified existing
  scraper.py            → Preserve section headers in clean_lyrics()
  preprocessor.py       → Refactored: structural decomposition instead of flat chunking
  embeddings.py         → Works with KuzuDB LyricEmbedding instead of ChromaDB only
  rag_chain.py          → Becomes orchestrator for multi-stage pipeline
  agent.py              → Calls new pipeline
  utils.py              → Extended with graph helpers

  # New: Graph layer
  graph/
    __init__.py
    schema.py           → KuzuDB schema (node tables, relation table, indexes)
    connection.py       → KuzuDB connection management (modeled on GitNexus kuzu-adapter.ts)
    ingestion.py        → Populate graph from analyzed data
    queries.py          → All Cypher query templates for retrieval stages
    loader.py           → CSV generation + bulk COPY loading

  # New: Analysis layer
  analysis/
    __init__.py
    lyric_analyzer.py   → Extract: sections, lines, phrases, rhymes, metaphors, mood
    fingerprint.py      → Build per-artist StyleFingerprint
    phonetics.py        → Hindi/English phonetic analysis for rhyme detection
    thematic_clustering.py → Leiden community detection

  # New: Retrieval layer
  retrieval/
    __init__.py
    pipeline.py         → Multi-stage retrieval orchestrator
    request_analyzer.py → Stage 0: request decomposition
    hybrid_search.py    → BM25 + semantic with RRF (ported from GitNexus)
    stages.py           → Stages 1-7 implementations

  # New: Prompt layer
  prompt/
    __init__.py
    assembler.py        → Context assembly with token budget management
    templates.py        → Dynamic prompt templates (replaces static prompts.yaml)

  # New: Validation layer
  validation/
    __init__.py
    validator.py        → Main validation orchestrator
    checks.py           → All 5 validation checks
    regenerator.py      → Re-generation strategy
```

---

## 10. Implementation Phases

| Phase | Work | Files to Create/Modify | Depends On |
|-------|------|----------------------|------------|
| **1** | Modify scraper to preserve section headers; build structural decomposer (Song → Section → Line) | Modify: `src/scraper.py`. New: `src/analysis/lyric_analyzer.py` | — |
| **2** | KuzuDB schema, connection manager, CSV bulk loader | New: `src/graph/schema.py`, `connection.py`, `loader.py` | Phase 1 |
| **3** | Linguistic analysis: phrase extraction, rhyme detection, cultural refs (keyword-based) | New: `src/analysis/phonetics.py`, extend `lyric_analyzer.py` | Phase 1 |
| **4** | LLM-powered analysis: metaphor extraction, theme assignment, emotional arcs, cluster enrichment | Extend: `src/analysis/lyric_analyzer.py`. New: `src/analysis/fingerprint.py` | Phase 3 |
| **5** | Leiden community detection + ThematicCluster creation + StyleFingerprint computation | New: `src/analysis/thematic_clustering.py` | Phase 4 |
| **6** | Embedding pipeline: batch embed, HNSW index, BM25 FTS indexes | Modify: `src/embeddings.py`. New: `src/graph/ingestion.py` | Phase 5 |
| **7** | Hybrid retrieval: 7-stage pipeline with RRF search + graph queries | New: `src/retrieval/pipeline.py`, `hybrid_search.py`, `stages.py` | Phase 6 |
| **8** | Prompt orchestration: two-part prompt, context assembly, token budgeting | New: `src/prompt/assembler.py`, `templates.py` | Phase 7 |
| **9** | Validation layer: 5 checks + re-generation strategy | New: `src/validation/validator.py`, `checks.py`, `regenerator.py` | Phase 8 |
| **10** | Integration: update agent.py, update Streamlit app, graph exploration UI, A/B testing vs old system | Modify: `src/agent.py`, `src/rag_chain.py`, `app/streamlit_app.py` | Phase 9 |

---

## 11. Key Dependencies to Add

```
# requirements.txt additions
kuzu>=0.8.0                  # Graph database (Python bindings)
igraph>=0.11.0               # Graph algorithms
leidenalg>=0.10.0            # Leiden community detection
epitran>=1.24                # IPA transliteration for Hindi/Urdu phonetics
```

---

## 12. Verification Plan

1. **Ingestion verification**: After Phase 6, run Cypher queries to verify node counts, relationship counts, and sample traversals for each artist. Verify StyleFingerprint values are reasonable.
2. **Retrieval verification**: After Phase 7, for a test query like "write about monsoon longing for Anuv Jain", verify all 7 stages return non-empty, relevant results. Compare retrieved context quality to old flat retrieval.
3. **Generation A/B test**: After Phase 10, generate the same 5 songs (same artist + topic) with old system and new system. Blind human evaluation by the 3-5 users on authenticity (1-5), originality (1-5), emotional resonance (1-5).
4. **Validation layer test**: Deliberately inject a generic/copied output and verify the validation layer catches it and triggers re-generation.
5. **End-to-end test**: Full flow from user input through Streamlit → agent → retrieval → generation → validation → display with references from graph.
