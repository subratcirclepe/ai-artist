"""Multi-stage retrieval pipeline orchestrator.

Executes 7 parallel retrieval stages to gather comprehensive context
for lyrics generation, then assembles results.
"""

import json
from dataclasses import dataclass, field

from src.utils import PROCESSED_DIR
from src.graph.connection import get_connection, execute_query, graph_exists
from src.graph.ingestion import _escape
from src.retrieval.hybrid_search import hybrid_search, SearchResult


# ---------------------------------------------------------------------------
# Data classes for retrieval results
# ---------------------------------------------------------------------------

@dataclass
class RequestAnalysis:
    topic: str
    mood_signals: list[str] = field(default_factory=list)
    structural_request: str = "full_song"
    language_preference: str = "auto"
    thematic_keywords: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """Aggregated results from all 7 retrieval stages."""
    # Stage 1: Similar sections
    thematic_sections: list[dict] = field(default_factory=list)
    # Stage 2: Vocabulary patterns
    vocabulary_clusters: list[str] = field(default_factory=list)
    signature_phrases: list[str] = field(default_factory=list)
    anti_vocabulary: list[str] = field(default_factory=list)
    # Stage 3: Rhyme schemes
    rhyme_schemes: dict = field(default_factory=dict)
    top_rhyme_pairs: list[dict] = field(default_factory=list)
    # Stage 4: Emotional arcs
    common_arcs: list[dict] = field(default_factory=list)
    mood_transitions: list[dict] = field(default_factory=list)
    # Stage 5: Metaphor bank
    metaphors: list[dict] = field(default_factory=list)
    # Stage 6: Cultural references
    cultural_references: list[dict] = field(default_factory=list)
    # Stage 7: Structural templates
    structures: list[dict] = field(default_factory=list)
    avg_lines_per_section: dict = field(default_factory=dict)
    # Fingerprint
    fingerprint: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Request analysis
# ---------------------------------------------------------------------------

# Mood keyword mapping for request analysis
_MOOD_SIGNAL_KEYWORDS = {
    "nostalgic": ["nostalgia", "memories", "remember", "old", "past", "yaad", "purana"],
    "romantic": ["love", "romance", "heart", "pyaar", "ishq", "dil"],
    "melancholic": ["sad", "pain", "tears", "lonely", "dard", "rona", "tanha"],
    "hopeful": ["hope", "dream", "future", "new", "umeed", "sapna"],
    "peaceful": ["peace", "calm", "rain", "baarish", "sukoon"],
    "energetic": ["party", "dance", "energy", "celebrate"],
}


def analyze_request(topic: str) -> RequestAnalysis:
    """Decompose a user request into structured facets."""
    topic_lower = topic.lower()

    # Detect mood signals
    mood_signals = []
    for mood, keywords in _MOOD_SIGNAL_KEYWORDS.items():
        if any(kw in topic_lower for kw in keywords):
            mood_signals.append(mood)

    if not mood_signals:
        mood_signals = ["neutral"]

    # Extract thematic keywords (words longer than 3 chars)
    thematic_keywords = [w for w in topic_lower.split() if len(w) > 3]

    return RequestAnalysis(
        topic=topic,
        mood_signals=mood_signals,
        structural_request="full_song",
        language_preference="auto",
        thematic_keywords=thematic_keywords,
    )


# ---------------------------------------------------------------------------
# Retrieval stages
# ---------------------------------------------------------------------------

def execute_retrieval_pipeline(
    artist_slug: str,
    request: RequestAnalysis,
) -> RetrievalResult:
    """Execute all 7 retrieval stages and assemble results.

    Args:
        artist_slug: Artist identifier.
        request: Analyzed request.

    Returns:
        Aggregated RetrievalResult.
    """
    result = RetrievalResult()

    if not graph_exists(artist_slug):
        print(f"  No graph exists for {artist_slug}, returning empty retrieval")
        return result

    conn = get_connection(artist_slug)

    # Stage 1: Thematic section retrieval (hybrid search)
    result.thematic_sections = _stage1_thematic_search(artist_slug, request)

    # Stage 2: Vocabulary patterns
    _stage2_vocabulary(conn, artist_slug, result)

    # Stage 3: Rhyme schemes
    _stage3_rhyme_schemes(conn, artist_slug, result)

    # Stage 4: Emotional arcs
    _stage4_emotional_arcs(conn, artist_slug, result)

    # Stage 5: Metaphor bank
    _stage5_metaphors(conn, artist_slug, request, result)

    # Stage 6: Cultural references
    _stage6_cultural_references(conn, artist_slug, result)

    # Stage 7: Structural templates
    _stage7_structures(conn, artist_slug, result)

    # Load fingerprint
    _load_fingerprint(conn, artist_slug, result)

    return result


def _stage1_thematic_search(
    artist_slug: str,
    request: RequestAnalysis,
) -> list[dict]:
    """Stage 1: Hybrid search for thematically similar sections."""
    try:
        results = hybrid_search(
            query=request.topic,
            artist_slug=artist_slug,
            node_type="section",
            limit=10,
        )
        return [
            {
                "node_id": r.node_id,
                "text": r.text,
                "section_type": r.metadata.get("section_type", ""),
                "mood": r.metadata.get("mood", ""),
                "line_count": r.metadata.get("line_count", 0),
                "score": r.score,
                "source": r.source,
            }
            for r in results
        ]
    except Exception as e:
        print(f"  Stage 1 error: {e}")
        return []


def _stage2_vocabulary(conn, artist_slug: str, result: RetrievalResult):
    """Stage 2: Extract vocabulary patterns from graph."""
    try:
        # Top phrases (signature expressions)
        rows = execute_query(conn,
            f"MATCH (p:Phrase) WHERE p.artist_id = '{_escape(artist_slug)}' "
            f"AND p.frequency >= 3 "
            f"RETURN p.text AS text, p.frequency AS freq "
            f"ORDER BY p.frequency DESC LIMIT 30")
        result.signature_phrases = [r["text"] for r in rows]

        # Load fingerprint for vocabulary set (from advanced analysis file)
        adv_path = PROCESSED_DIR / f"{artist_slug}_advanced_analysis.json"
        if adv_path.exists():
            with open(adv_path, "r", encoding="utf-8") as f:
                adv = json.load(f)
            fp = adv.get("fingerprint", {})
            result.vocabulary_clusters = fp.get("vocabulary_set", [])[:100]
            result.anti_vocabulary = fp.get("anti_vocabulary", [])
    except Exception as e:
        print(f"  Stage 2 error: {e}")


def _stage3_rhyme_schemes(conn, artist_slug: str, result: RetrievalResult):
    """Stage 3: Extract rhyme scheme patterns from graph."""
    try:
        # Top rhyme pairs
        rows = execute_query(conn,
            f"MATCH (rp:RhymePair) WHERE rp.artist_id = '{_escape(artist_slug)}' "
            f"RETURN rp.word_a AS word_a, rp.word_b AS word_b, "
            f"rp.rhyme_type AS rhyme_type, rp.frequency AS freq "
            f"ORDER BY rp.frequency DESC LIMIT 20")
        result.top_rhyme_pairs = [dict(r) for r in rows]

        # Rhyme schemes from fingerprint
        adv_path = PROCESSED_DIR / f"{artist_slug}_advanced_analysis.json"
        if adv_path.exists():
            with open(adv_path, "r", encoding="utf-8") as f:
                adv = json.load(f)
            fp = adv.get("fingerprint", {})
            top_types = fp.get("top_rhyme_types", [])
            result.rhyme_schemes = {
                "preferred_patterns": top_types[:5],
            }
    except Exception as e:
        print(f"  Stage 3 error: {e}")


def _stage4_emotional_arcs(conn, artist_slug: str, result: RetrievalResult):
    """Stage 4: Extract emotional arc patterns from graph."""
    try:
        rows = execute_query(conn,
            f"MATCH (a:EmotionalArc) WHERE a.song_id STARTS WITH '{_escape(artist_slug)}:' "
            f"RETURN a.arc_type AS arc_type, a.description AS description "
            f"LIMIT 20")
        result.common_arcs = [dict(r) for r in rows]
    except Exception as e:
        print(f"  Stage 4 error: {e}")


def _stage5_metaphors(conn, artist_slug: str, request: RequestAnalysis, result: RetrievalResult):
    """Stage 5: Extract relevant metaphors from graph."""
    try:
        rows = execute_query(conn,
            f"MATCH (m:Metaphor) WHERE m.artist_id = '{_escape(artist_slug)}' "
            f"RETURN m.source_text AS source_text, m.source_domain AS source_domain, "
            f"m.target_domain AS target_domain, m.frequency AS freq "
            f"ORDER BY m.frequency DESC LIMIT 15")
        result.metaphors = [dict(r) for r in rows]
    except Exception as e:
        print(f"  Stage 5 error: {e}")


def _stage6_cultural_references(conn, artist_slug: str, result: RetrievalResult):
    """Stage 6: Extract cultural references from graph."""
    try:
        rows = execute_query(conn,
            f"MATCH (cr:CulturalReference) WHERE cr.artist_id = '{_escape(artist_slug)}' "
            f"RETURN cr.reference_text AS reference, cr.category AS category, "
            f"cr.cultural_context AS context, cr.frequency AS freq "
            f"ORDER BY cr.frequency DESC LIMIT 15")
        result.cultural_references = [dict(r) for r in rows]
    except Exception as e:
        print(f"  Stage 6 error: {e}")


def _stage7_structures(conn, artist_slug: str, result: RetrievalResult):
    """Stage 7: Extract structural templates from graph."""
    try:
        rows = execute_query(conn,
            f"MATCH (t:StructureTemplate) WHERE t.artist_id = '{_escape(artist_slug)}' "
            f"RETURN t.pattern AS pattern, t.frequency AS freq, t.description AS description "
            f"ORDER BY t.frequency DESC LIMIT 5")
        result.structures = [dict(r) for r in rows]

        # Average lines per section type
        rows2 = execute_query(conn,
            f"MATCH (sec:Section) WHERE sec.song_id STARTS WITH '{_escape(artist_slug)}:' "
            f"RETURN sec.section_type AS section_type, "
            f"avg(sec.line_count) AS avg_lines "
            f"ORDER BY section_type")
        result.avg_lines_per_section = {
            r["section_type"]: round(r["avg_lines"], 1) for r in rows2
        }
    except Exception as e:
        print(f"  Stage 7 error: {e}")


def _load_fingerprint(conn, artist_slug: str, result: RetrievalResult):
    """Load the artist's StyleFingerprint."""
    try:
        rows = execute_query(conn,
            f"MATCH (f:StyleFingerprint) WHERE f.artist_id = '{_escape(artist_slug)}' "
            f"RETURN f")
        if rows:
            result.fingerprint = rows[0].get("f", {})
    except Exception as e:
        print(f"  Fingerprint load error: {e}")
