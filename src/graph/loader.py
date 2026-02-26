"""Embedding generation and graph loading pipeline.

Generates vector embeddings for graph nodes, loads them into KuzuDB,
and creates HNSW + FTS indexes for hybrid search.
"""

import json
from dataclasses import asdict

from langchain_huggingface import HuggingFaceEmbeddings

from src.utils import PROCESSED_DIR
from src.graph.connection import (
    get_connection,
    initialize_schema,
    create_indexes,
    execute_query,
)
from src.graph.ingestion import _escape, _safe_execute


# Embedding model (same as existing system)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_function() -> HuggingFaceEmbeddings:
    """Create the HuggingFace embedding function."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def generate_and_load_embeddings(
    artist_slug: str,
    graph_data: dict | None = None,
) -> dict:
    """Generate embeddings for Song, Section, and Line nodes, store in KuzuDB.

    Args:
        artist_slug: Artist identifier.
        graph_data: Pre-loaded graph data. If None, loads from disk.

    Returns:
        Stats dict with embedding counts.
    """
    if graph_data is None:
        data_path = PROCESSED_DIR / f"{artist_slug}_graph_data.json"
        with open(data_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

    conn = get_connection(artist_slug)
    embed_fn = get_embedding_function()

    stats = {"song_embeddings": 0, "section_embeddings": 0, "line_embeddings": 0}

    # --- Song-level embeddings ---
    print("Generating song embeddings...")
    song_texts = []
    song_ids = []
    for song in graph_data.get("songs", []):
        text = _build_song_embedding_text(song)
        song_texts.append(text)
        song_ids.append(song["id"])

    if song_texts:
        song_vectors = embed_fn.embed_documents(song_texts)
        for node_id, vector in zip(song_ids, song_vectors):
            _insert_embedding(conn, node_id, vector)
            stats["song_embeddings"] += 1
        print(f"  {stats['song_embeddings']} song embeddings created")

    # --- Section-level embeddings ---
    print("Generating section embeddings...")
    section_texts = []
    section_ids = []
    for song in graph_data.get("songs", []):
        for section in song.get("sections", []):
            text = _build_section_embedding_text(section, song["title"], artist_slug)
            section_texts.append(text)
            section_ids.append(section["id"])

    if section_texts:
        # Batch embed in chunks of 100
        for i in range(0, len(section_texts), 100):
            batch_texts = section_texts[i:i+100]
            batch_ids = section_ids[i:i+100]
            vectors = embed_fn.embed_documents(batch_texts)
            for node_id, vector in zip(batch_ids, vectors):
                _insert_embedding(conn, node_id, vector)
                stats["section_embeddings"] += 1
        print(f"  {stats['section_embeddings']} section embeddings created")

    # --- Line-level embeddings (only lines with 3+ words for quality) ---
    print("Generating line embeddings...")
    line_texts = []
    line_ids = []
    for song in graph_data.get("songs", []):
        for section in song.get("sections", []):
            for line in section.get("lines", []):
                if line.get("word_count", 0) >= 3:
                    line_texts.append(line["text"])
                    line_ids.append(line["id"])

    if line_texts:
        for i in range(0, len(line_texts), 200):
            batch_texts = line_texts[i:i+200]
            batch_ids = line_ids[i:i+200]
            vectors = embed_fn.embed_documents(batch_texts)
            for node_id, vector in zip(batch_ids, vectors):
                _insert_embedding(conn, node_id, vector)
                stats["line_embeddings"] += 1
        print(f"  {stats['line_embeddings']} line embeddings created")

    # --- Create indexes ---
    try:
        create_indexes(conn)
    except Exception as e:
        print(f"  Warning creating indexes: {e}")

    total = sum(stats.values())
    print(f"Total embeddings: {total}")
    return stats


def _build_song_embedding_text(song: dict) -> str:
    """Build embedding text for a Song node."""
    title = song.get("title", "")
    mood = song.get("mood", "")
    lyrics_preview = song.get("full_lyrics_clean", "")[:300]
    return f"Song: {title}. Mood: {mood}.\n{lyrics_preview}"


def _build_section_embedding_text(section: dict, song_title: str, artist_slug: str) -> str:
    """Build embedding text for a Section node."""
    sec_type = section.get("section_type", "section")
    text = section.get("text", "")[:500]
    return f"Section ({sec_type}) from {song_title}:\n{text}"


def _insert_embedding(conn, node_id: str, vector: list[float]) -> None:
    """Insert a single embedding into the LyricEmbedding table."""
    # Format vector as Kuzu array literal
    vec_str = "[" + ",".join(f"{v:.6f}" for v in vector) + "]"
    _safe_execute(conn,
        f"CREATE (e:LyricEmbedding {{node_id: '{_escape(node_id)}', "
        f"embedding: {vec_str}}})")


# ---------------------------------------------------------------------------
# Advanced data ingestion (themes, metaphors, arcs, fingerprint, clusters)
# ---------------------------------------------------------------------------

def ingest_advanced_data(
    artist_slug: str,
    advanced_data: dict | None = None,
    clusters: list | None = None,
    rhyme_pairs: list | None = None,
) -> dict:
    """Ingest advanced analysis data into the knowledge graph.

    Args:
        artist_slug: Artist identifier.
        advanced_data: Dict with themes, metaphors, emotional_arcs, fingerprint.
        clusters: List of ThematicClusterData dicts.
        rhyme_pairs: List of RhymePairData dicts.

    Returns:
        Stats dict.
    """
    if advanced_data is None:
        adv_path = PROCESSED_DIR / f"{artist_slug}_advanced_analysis.json"
        with open(adv_path, "r", encoding="utf-8") as f:
            advanced_data = json.load(f)

    conn = get_connection(artist_slug)
    stats = {
        "themes": 0, "metaphors": 0, "emotional_arcs": 0,
        "fingerprints": 0, "clusters": 0, "rhyme_pairs": 0,
        "rel_has_theme": 0, "rel_has_arc": 0, "rel_member_of_cluster": 0,
        "rel_has_fingerprint": 0,
    }

    # --- Themes ---
    for theme in advanced_data.get("themes", []):
        _safe_execute(conn,
            f"CREATE (t:Theme {{id: '{_escape(theme['id'])}', "
            f"name: '{_escape(theme['name'])}', "
            f"description: '{_escape(theme['description'])}', "
            f"artist_id: '{_escape(theme['artist_id'])}', "
            f"song_count: {theme.get('song_count', 0)}}})")
        stats["themes"] += 1

    # Link songs to themes
    from src.analysis.fingerprint import THEME_KEYWORDS
    graph_data_path = PROCESSED_DIR / f"{artist_slug}_graph_data.json"
    if graph_data_path.exists():
        with open(graph_data_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        for song in graph_data.get("songs", []):
            text_lower = song.get("full_lyrics_clean", "").lower()
            for theme in advanced_data.get("themes", []):
                theme_key = theme["name"].lower().replace(" ", "_")
                keywords = THEME_KEYWORDS.get(theme_key, [])
                score = sum(1 for kw in keywords if kw in text_lower)
                if score >= 2:
                    strength = min(score / 10.0, 1.0)
                    _safe_execute(conn,
                        f"MATCH (s:Song), (t:Theme) "
                        f"WHERE s.id = '{_escape(song['id'])}' "
                        f"AND t.id = '{_escape(theme['id'])}' "
                        f"CREATE (s)-[:HAS_THEME {{strength: {strength}, "
                        f"type: 'HAS_THEME'}}]->(t)")
                    stats["rel_has_theme"] += 1

    # --- Metaphors ---
    for metaphor in advanced_data.get("metaphors", []):
        _safe_execute(conn,
            f"CREATE (m:Metaphor {{id: '{_escape(metaphor['id'])}', "
            f"source_text: '{_escape(metaphor['source_text'])}', "
            f"source_domain: '{_escape(metaphor['source_domain'])}', "
            f"target_domain: '{_escape(metaphor['target_domain'])}', "
            f"artist_id: '{_escape(metaphor['artist_id'])}', "
            f"frequency: {metaphor.get('frequency', 0)}}})")
        stats["metaphors"] += 1

    # --- Emotional Arcs ---
    for arc in advanced_data.get("emotional_arcs", []):
        _safe_execute(conn,
            f"CREATE (a:EmotionalArc {{id: '{_escape(arc['id'])}', "
            f"song_id: '{_escape(arc['song_id'])}', "
            f"arc_type: '{_escape(arc['arc_type'])}', "
            f"description: '{_escape(arc.get('description', ''))}'}})")
        stats["emotional_arcs"] += 1

        # HAS_ARC relationship
        _safe_execute(conn,
            f"MATCH (s:Song), (a:EmotionalArc) "
            f"WHERE s.id = '{_escape(arc['song_id'])}' "
            f"AND a.id = '{_escape(arc['id'])}' "
            f"CREATE (s)-[:HAS_ARC {{type: 'HAS_ARC'}}]->(a)")
        stats["rel_has_arc"] += 1

    # --- StyleFingerprint ---
    fp = advanced_data.get("fingerprint", {})
    if fp:
        _safe_execute(conn,
            f"CREATE (f:StyleFingerprint {{id: '{_escape(fp['id'])}', "
            f"artist_id: '{_escape(fp['artist_id'])}', "
            f"avg_line_length: {fp.get('avg_line_length', 0)}, "
            f"avg_section_length: {fp.get('avg_section_length', 0)}, "
            f"vocabulary_richness: {fp.get('vocabulary_richness', 0)}, "
            f"code_switch_frequency: {fp.get('code_switch_frequency', 0)}, "
            f"metaphor_density: {fp.get('metaphor_density', 0)}, "
            f"repetition_index: {fp.get('repetition_index', 0)}, "
            f"avg_mood_valence: {fp.get('avg_mood_valence', 0)}, "
            f"avg_mood_arousal: {fp.get('avg_mood_arousal', 0)}}})")
        stats["fingerprints"] += 1

        # HAS_FINGERPRINT relationship
        _safe_execute(conn,
            f"MATCH (a:Artist), (f:StyleFingerprint) "
            f"WHERE a.id = '{_escape(fp['artist_id'])}' "
            f"AND f.id = '{_escape(fp['id'])}' "
            f"CREATE (a)-[:HAS_FINGERPRINT {{type: 'HAS_FINGERPRINT'}}]->(f)")
        stats["rel_has_fingerprint"] += 1

    # --- Thematic Clusters ---
    if clusters:
        for cluster in clusters:
            c = cluster if isinstance(cluster, dict) else asdict(cluster)
            _safe_execute(conn,
                f"CREATE (tc:ThematicCluster {{id: '{_escape(c['id'])}', "
                f"label: '{_escape(c.get('label', ''))}', "
                f"heuristic_label: '{_escape(c.get('heuristic_label', ''))}', "
                f"description: '{_escape(c.get('description', ''))}', "
                f"enriched_by: '{_escape(c.get('enriched_by', 'heuristic'))}', "
                f"cohesion: {c.get('cohesion', 0)}, "
                f"song_count: {c.get('song_count', 0)}, "
                f"artist_id: '{_escape(c.get('artist_id', ''))}' }})")
            stats["clusters"] += 1

            # MEMBER_OF_CLUSTER relationships
            for sid in c.get("song_ids", []):
                _safe_execute(conn,
                    f"MATCH (s:Song), (tc:ThematicCluster) "
                    f"WHERE s.id = '{_escape(sid)}' "
                    f"AND tc.id = '{_escape(c['id'])}' "
                    f"CREATE (s)-[:MEMBER_OF_CLUSTER {{type: 'MEMBER_OF_CLUSTER'}}]->(tc)")
                stats["rel_member_of_cluster"] += 1

    # --- Rhyme Pairs ---
    if rhyme_pairs:
        for rp in rhyme_pairs[:100]:  # Top 100 pairs
            r = rp if isinstance(rp, dict) else asdict(rp)
            _safe_execute(conn,
                f"CREATE (rp:RhymePair {{id: '{_escape(r['id'])}', "
                f"word_a: '{_escape(r['word_a'])}', "
                f"word_b: '{_escape(r['word_b'])}', "
                f"rhyme_type: '{_escape(r['rhyme_type'])}', "
                f"language: '{_escape(r.get('language', ''))}', "
                f"frequency: {r.get('frequency', 0)}, "
                f"artist_id: '{_escape(r.get('artist_id', ''))}'}})")
            stats["rhyme_pairs"] += 1

    print(f"\nAdvanced data ingestion complete:")
    for key, val in stats.items():
        if val > 0:
            print(f"  {key}: {val}")

    return stats
