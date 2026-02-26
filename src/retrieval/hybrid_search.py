"""Hybrid search combining BM25 keyword and semantic vector search with RRF.

Ported from GitNexus's hybrid-search.ts — uses Reciprocal Rank Fusion
to merge results from keyword and semantic search without score normalization.
"""

from dataclasses import dataclass

from langchain_huggingface import HuggingFaceEmbeddings

from src.graph.connection import execute_query, get_connection
from src.graph.loader import get_embedding_function
from src.graph.ingestion import _escape


# RRF constant (same as GitNexus: hybrid-search.ts line 18)
RRF_K = 60


@dataclass
class SearchResult:
    node_id: str
    node_type: str  # "song", "section", "line"
    text: str
    metadata: dict
    score: float
    source: str  # "semantic", "keyword", "both"


def hybrid_search(
    query: str,
    artist_slug: str,
    node_type: str = "section",
    limit: int = 10,
) -> list[SearchResult]:
    """Execute hybrid search: semantic + keyword with RRF fusion.

    Args:
        query: Search query text.
        artist_slug: Artist identifier.
        node_type: Which node type to search ("song", "section", "line").
        limit: Maximum results to return.

    Returns:
        Ranked list of SearchResult.
    """
    conn = get_connection(artist_slug)

    # Run both searches in sequence (KuzuDB single-writer constraint)
    semantic_results = _semantic_search(conn, query, artist_slug, node_type, limit * 2)
    keyword_results = _keyword_search(conn, query, artist_slug, node_type, limit * 2)

    # Merge with RRF
    merged = _merge_with_rrf(semantic_results, keyword_results, limit)
    return merged


def _semantic_search(
    conn,
    query: str,
    artist_slug: str,
    node_type: str,
    limit: int,
) -> list[SearchResult]:
    """Vector similarity search on LyricEmbedding table."""
    embed_fn = get_embedding_function()
    query_vector = embed_fn.embed_query(query)
    vec_str = "[" + ",".join(f"{v:.6f}" for v in query_vector) + "]"

    # Map node_type to the right table for joining
    if node_type == "song":
        join_query = (
            f"MATCH (e:LyricEmbedding), (s:Song) "
            f"WHERE s.id = e.node_id AND s.artist_id = '{_escape(artist_slug)}' "
            f"WITH s, e, array_cosine_similarity(e.embedding, {vec_str}) AS score "
            f"ORDER BY score DESC LIMIT {limit} "
            f"RETURN s.id AS node_id, 'song' AS node_type, s.title AS text, "
            f"s.mood AS mood, s.language AS language, score"
        )
    elif node_type == "section":
        join_query = (
            f"MATCH (e:LyricEmbedding), (sec:Section) "
            f"WHERE sec.id = e.node_id AND sec.song_id STARTS WITH '{_escape(artist_slug)}:' "
            f"WITH sec, e, array_cosine_similarity(e.embedding, {vec_str}) AS score "
            f"ORDER BY score DESC LIMIT {limit} "
            f"RETURN sec.id AS node_id, 'section' AS node_type, sec.text AS text, "
            f"sec.mood AS mood, sec.section_type AS section_type, "
            f"sec.line_count AS line_count, score"
        )
    else:  # line
        join_query = (
            f"MATCH (e:LyricEmbedding), (l:Line) "
            f"WHERE l.id = e.node_id AND l.song_id STARTS WITH '{_escape(artist_slug)}:' "
            f"WITH l, e, array_cosine_similarity(e.embedding, {vec_str}) AS score "
            f"ORDER BY score DESC LIMIT {limit} "
            f"RETURN l.id AS node_id, 'line' AS node_type, l.text AS text, "
            f"l.language AS language, l.end_word AS end_word, score"
        )

    try:
        rows = execute_query(conn, join_query)
        results = []
        for row in rows:
            results.append(SearchResult(
                node_id=row.get("node_id", ""),
                node_type=row.get("node_type", node_type),
                text=row.get("text", ""),
                metadata={k: v for k, v in row.items() if k not in ("node_id", "node_type", "text", "score")},
                score=float(row.get("score", 0)),
                source="semantic",
            ))
        return results
    except Exception as e:
        print(f"  Semantic search error: {e}")
        return []


def _keyword_search(
    conn,
    query: str,
    artist_slug: str,
    node_type: str,
    limit: int,
) -> list[SearchResult]:
    """Keyword search using string matching (fallback for FTS).

    KuzuDB FTS requires explicit index creation. We use CONTAINS as fallback.
    """
    # Tokenize query into keywords for flexible matching
    keywords = [w.strip().lower() for w in query.split() if len(w.strip()) > 2]
    if not keywords:
        return []

    # Build WHERE clause with OR conditions for each keyword
    if node_type == "song":
        conditions = " OR ".join(
            f"s.full_lyrics CONTAINS '{_escape(kw)}'" for kw in keywords
        )
        kw_query = (
            f"MATCH (s:Song) WHERE s.artist_id = '{_escape(artist_slug)}' "
            f"AND ({conditions}) "
            f"RETURN s.id AS node_id, 'song' AS node_type, s.title AS text, "
            f"s.mood AS mood, s.language AS language "
            f"LIMIT {limit}"
        )
    elif node_type == "section":
        conditions = " OR ".join(
            f"sec.text CONTAINS '{_escape(kw)}'" for kw in keywords
        )
        kw_query = (
            f"MATCH (sec:Section) WHERE sec.song_id STARTS WITH '{_escape(artist_slug)}:' "
            f"AND ({conditions}) "
            f"RETURN sec.id AS node_id, 'section' AS node_type, sec.text AS text, "
            f"sec.mood AS mood, sec.section_type AS section_type, "
            f"sec.line_count AS line_count "
            f"LIMIT {limit}"
        )
    else:
        conditions = " OR ".join(
            f"l.text CONTAINS '{_escape(kw)}'" for kw in keywords
        )
        kw_query = (
            f"MATCH (l:Line) WHERE l.song_id STARTS WITH '{_escape(artist_slug)}:' "
            f"AND ({conditions}) "
            f"RETURN l.id AS node_id, 'line' AS node_type, l.text AS text, "
            f"l.language AS language, l.end_word AS end_word "
            f"LIMIT {limit}"
        )

    try:
        rows = execute_query(conn, kw_query)
        results = []
        for rank, row in enumerate(rows):
            # Compute a relevance score based on keyword match count
            text_lower = row.get("text", "").lower()
            match_count = sum(1 for kw in keywords if kw in text_lower)
            score = match_count / max(len(keywords), 1)

            results.append(SearchResult(
                node_id=row.get("node_id", ""),
                node_type=row.get("node_type", node_type),
                text=row.get("text", ""),
                metadata={k: v for k, v in row.items() if k not in ("node_id", "node_type", "text")},
                score=score,
                source="keyword",
            ))
        return results
    except Exception as e:
        print(f"  Keyword search error: {e}")
        return []


def _merge_with_rrf(
    semantic_results: list[SearchResult],
    keyword_results: list[SearchResult],
    limit: int,
) -> list[SearchResult]:
    """Merge two result lists using Reciprocal Rank Fusion.

    RRF formula: score = 1 / (RRF_K + rank + 1)
    This avoids the need to normalize different score scales.
    """
    rrf_scores: dict[str, float] = {}
    result_map: dict[str, SearchResult] = {}
    sources: dict[str, set[str]] = {}

    # Score semantic results
    for rank, r in enumerate(semantic_results):
        rrf_score = 1.0 / (RRF_K + rank + 1)
        rrf_scores[r.node_id] = rrf_scores.get(r.node_id, 0) + rrf_score
        result_map[r.node_id] = r
        sources.setdefault(r.node_id, set()).add("semantic")

    # Score keyword results
    for rank, r in enumerate(keyword_results):
        rrf_score = 1.0 / (RRF_K + rank + 1)
        rrf_scores[r.node_id] = rrf_scores.get(r.node_id, 0) + rrf_score
        if r.node_id not in result_map:
            result_map[r.node_id] = r
        sources.setdefault(r.node_id, set()).add("keyword")

    # Sort by RRF score and return top results
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    merged = []
    for node_id in sorted_ids[:limit]:
        result = result_map[node_id]
        result.score = rrf_scores[node_id]
        src = sources.get(node_id, set())
        result.source = "both" if len(src) > 1 else next(iter(src))
        merged.append(result)

    return merged
