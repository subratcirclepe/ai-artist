"""Populate the KuzuDB graph from analyzed lyrics data.

Reads the structured analysis JSON produced by lyric_analyzer.py
and inserts all nodes and relationships into the knowledge graph.
"""

import json
from pathlib import Path
from collections import Counter

import kuzu

from src.utils import PROCESSED_DIR, load_artist_config
from src.graph.connection import (
    get_connection,
    initialize_schema,
    execute_query,
)


def _escape(val: str) -> str:
    """Escape a string value for Cypher query insertion."""
    if val is None:
        return ""
    return val.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "")


def ingest_artist(artist_slug: str, graph_data: dict | None = None) -> dict:
    """Ingest all analyzed data for an artist into the knowledge graph.

    Args:
        artist_slug: Artist identifier.
        graph_data: Pre-loaded graph data dict. If None, loads from disk.

    Returns:
        Stats dict with counts of nodes and relationships created.
    """
    if graph_data is None:
        data_path = PROCESSED_DIR / f"{artist_slug}_graph_data.json"
        if not data_path.exists():
            raise FileNotFoundError(
                f"No graph data at {data_path}. Run analyze_artist() first."
            )
        with open(data_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

    conn = get_connection(artist_slug)
    initialize_schema(conn)

    stats = {
        "artists": 0, "albums": 0, "songs": 0, "sections": 0, "lines": 0,
        "phrases": 0, "cultural_refs": 0, "meter_patterns": 0, "moods": 0,
        "structure_templates": 0,
        "rel_written_by": 0, "rel_contains_section": 0, "rel_contains_line": 0,
        "rel_section_follows": 0, "rel_line_follows": 0, "rel_uses_phrase": 0,
        "rel_song_refs_culture": 0, "rel_expresses_mood": 0,
        "rel_uses_structure": 0, "rel_has_meter": 0,
    }

    # --- 1. Create Artist node ---
    try:
        artist_config = load_artist_config(artist_slug)
    except ValueError:
        artist_config = {}

    _create_artist_node(conn, artist_slug, artist_config, graph_data)
    stats["artists"] = 1

    # --- 2. Create Mood nodes (fixed set) ---
    mood_set = _create_mood_nodes(conn)
    stats["moods"] = len(mood_set)

    # --- 3. Process songs ---
    album_ids_created = set()
    structure_counter: Counter = Counter()

    for song_data in graph_data.get("songs", []):
        # Create Album node if needed
        album_name = song_data.get("album", "")
        if album_name and album_name not in album_ids_created:
            _create_album_node(conn, artist_slug, album_name)
            album_ids_created.add(album_name)
            stats["albums"] += 1

        # Create Song node
        _create_song_node(conn, song_data)
        stats["songs"] += 1

        # WRITTEN_BY relationship
        _safe_execute(conn,
            f"MATCH (s:Song), (a:Artist) WHERE s.id = '{_escape(song_data['id'])}' "
            f"AND a.id = '{_escape(artist_slug)}' "
            f"CREATE (s)-[:WRITTEN_BY {{type: 'WRITTEN_BY'}}]->(a)")
        stats["rel_written_by"] += 1

        # BELONGS_TO relationship
        if album_name:
            album_id = f"{artist_slug}:album:{album_name.lower().replace(' ', '_')[:50]}"
            _safe_execute(conn,
                f"MATCH (s:Song), (a:Album) WHERE s.id = '{_escape(song_data['id'])}' "
                f"AND a.id = '{_escape(album_id)}' "
                f"CREATE (s)-[:BELONGS_TO {{track_number: 0, type: 'BELONGS_TO'}}]->(a)")

        # SONG_EXPRESSES_MOOD relationship
        song_mood = song_data.get("mood", "neutral")
        if song_mood in mood_set:
            _safe_execute(conn,
                f"MATCH (s:Song), (m:Mood) WHERE s.id = '{_escape(song_data['id'])}' "
                f"AND m.id = '{_escape(song_mood)}' "
                f"CREATE (s)-[:SONG_EXPRESSES_MOOD {{intensity: 0.7, type: 'EXPRESSES_MOOD'}}]->(m)")
            stats["rel_expresses_mood"] += 1

        # Track structure template
        sec_types = [sec["section_type"] for sec in song_data.get("sections", [])]
        structure_pattern = "-".join(sec_types)
        structure_counter[structure_pattern] += 1

        # Process sections
        prev_section_id = None
        for sec_data in song_data.get("sections", []):
            _create_section_node(conn, sec_data)
            stats["sections"] += 1

            # CONTAINS_SECTION
            _safe_execute(conn,
                f"MATCH (s:Song), (sec:Section) WHERE s.id = '{_escape(song_data['id'])}' "
                f"AND sec.id = '{_escape(sec_data['id'])}' "
                f"CREATE (s)-[:CONTAINS_SECTION {{type: 'CONTAINS_SECTION'}}]->(sec)")
            stats["rel_contains_section"] += 1

            # SECTION_FOLLOWS
            if prev_section_id:
                _safe_execute(conn,
                    f"MATCH (s1:Section), (s2:Section) WHERE s1.id = '{_escape(prev_section_id)}' "
                    f"AND s2.id = '{_escape(sec_data['id'])}' "
                    f"CREATE (s1)-[:SECTION_FOLLOWS {{gap: 0, type: 'FOLLOWS'}}]->(s2)")
                stats["rel_section_follows"] += 1
            prev_section_id = sec_data["id"]

            # SECTION_EXPRESSES_MOOD
            sec_mood = sec_data.get("mood", "neutral")
            if sec_mood in mood_set:
                _safe_execute(conn,
                    f"MATCH (sec:Section), (m:Mood) WHERE sec.id = '{_escape(sec_data['id'])}' "
                    f"AND m.id = '{_escape(sec_mood)}' "
                    f"CREATE (sec)-[:SECTION_EXPRESSES_MOOD {{intensity: 0.7, type: 'EXPRESSES_MOOD'}}]->(m)")
                stats["rel_expresses_mood"] += 1

            # Process lines
            prev_line_id = None
            for line_data in sec_data.get("lines", []):
                _create_line_node(conn, line_data)
                stats["lines"] += 1

                # CONTAINS_LINE
                _safe_execute(conn,
                    f"MATCH (sec:Section), (l:Line) WHERE sec.id = '{_escape(sec_data['id'])}' "
                    f"AND l.id = '{_escape(line_data['id'])}' "
                    f"CREATE (sec)-[:CONTAINS_LINE {{type: 'CONTAINS_LINE'}}]->(l)")
                stats["rel_contains_line"] += 1

                # LINE_FOLLOWS
                if prev_line_id:
                    _safe_execute(conn,
                        f"MATCH (l1:Line), (l2:Line) WHERE l1.id = '{_escape(prev_line_id)}' "
                        f"AND l2.id = '{_escape(line_data['id'])}' "
                        f"CREATE (l1)-[:LINE_FOLLOWS {{gap: 0, type: 'FOLLOWS'}}]->(l2)")
                    stats["rel_line_follows"] += 1
                prev_line_id = line_data["id"]

    # --- 4. Create Phrase nodes and USES_PHRASE relationships ---
    phrase_map = {}
    for phrase_data in graph_data.get("phrases", [])[:200]:  # Top 200 phrases
        _create_phrase_node(conn, phrase_data)
        phrase_map[phrase_data["text"].lower()] = phrase_data["id"]
        stats["phrases"] += 1

    # Link lines to phrases
    for song_data in graph_data.get("songs", []):
        for sec_data in song_data.get("sections", []):
            for line_data in sec_data.get("lines", []):
                line_text_lower = line_data["text"].lower()
                for phrase_text, phrase_id in phrase_map.items():
                    if phrase_text in line_text_lower:
                        pos = line_text_lower.index(phrase_text)
                        _safe_execute(conn,
                            f"MATCH (l:Line), (p:Phrase) WHERE l.id = '{_escape(line_data['id'])}' "
                            f"AND p.id = '{_escape(phrase_id)}' "
                            f"CREATE (l)-[:USES_PHRASE {{position: {pos}, type: 'USES_PHRASE'}}]->(p)")
                        stats["rel_uses_phrase"] += 1
                        break  # One phrase per line to avoid explosion

    # --- 5. Create CulturalReference nodes ---
    for cr_data in graph_data.get("cultural_references", []):
        _create_cultural_ref_node(conn, cr_data)
        stats["cultural_refs"] += 1

    # Link songs to cultural references
    for song_data in graph_data.get("songs", []):
        text_lower = song_data.get("full_lyrics_clean", "").lower()
        for cr_data in graph_data.get("cultural_references", []):
            if cr_data["reference_text"].lower() in text_lower:
                _safe_execute(conn,
                    f"MATCH (s:Song), (cr:CulturalReference) "
                    f"WHERE s.id = '{_escape(song_data['id'])}' "
                    f"AND cr.id = '{_escape(cr_data['id'])}' "
                    f"CREATE (s)-[:SONG_REFERENCES_CULTURE {{type: 'REFERENCES_CULTURE'}}]->(cr)")
                stats["rel_song_refs_culture"] += 1

    # --- 6. Create MeterPattern nodes ---
    for mp_data in graph_data.get("meter_patterns", [])[:50]:
        _create_meter_pattern_node(conn, mp_data)
        stats["meter_patterns"] += 1

    # --- 7. Create StructureTemplate nodes ---
    for pattern, freq in structure_counter.most_common(10):
        template_id = f"{artist_slug}:structure:{pattern[:60].replace('-', '_')}"
        _safe_execute(conn,
            f"CREATE (t:StructureTemplate {{id: '{_escape(template_id)}', "
            f"pattern: '{_escape(pattern)}', artist_id: '{_escape(artist_slug)}', "
            f"frequency: {freq}, description: 'Song structure: {_escape(pattern)}'}})")
        stats["structure_templates"] += 1

    # Link songs to structure templates
    for song_data in graph_data.get("songs", []):
        sec_types = [sec["section_type"] for sec in song_data.get("sections", [])]
        pattern = "-".join(sec_types)
        template_id = f"{artist_slug}:structure:{pattern[:60].replace('-', '_')}"
        _safe_execute(conn,
            f"MATCH (s:Song), (t:StructureTemplate) "
            f"WHERE s.id = '{_escape(song_data['id'])}' "
            f"AND t.id = '{_escape(template_id)}' "
            f"CREATE (s)-[:USES_STRUCTURE {{type: 'USES_STRUCTURE'}}]->(t)")
        stats["rel_uses_structure"] += 1

    print(f"\nGraph ingestion complete for {artist_slug}:")
    for key, val in stats.items():
        if val > 0:
            print(f"  {key}: {val}")

    return stats


# ---------------------------------------------------------------------------
# Node creation helpers
# ---------------------------------------------------------------------------

def _safe_execute(conn: kuzu.Connection, query: str) -> None:
    """Execute a query, silently handling duplicate/constraint errors."""
    try:
        conn.execute(query)
    except Exception as e:
        err = str(e).lower()
        if "already exists" in err or "duplicate" in err or "violate" in err:
            pass
        else:
            print(f"  Query error: {e}\n  Query: {query[:200]}")


def _create_artist_node(conn, artist_slug, config, graph_data):
    total_songs = graph_data.get("stats", {}).get("total_songs", 0)
    total_lines = graph_data.get("stats", {}).get("total_lines", 0)
    _safe_execute(conn,
        f"CREATE (a:Artist {{id: '{_escape(artist_slug)}', "
        f"name: '{_escape(config.get('name', artist_slug))}', "
        f"slug: '{_escape(artist_slug)}', "
        f"language: '{_escape(config.get('language', ''))}', "
        f"musical_style: '{_escape(config.get('musical_style', ''))}', "
        f"vocal_style: '{_escape(config.get('vocal_style', ''))}', "
        f"vocabulary_level: '{_escape(config.get('vocabulary_level', ''))}', "
        f"song_count: {total_songs}, "
        f"total_line_count: {total_lines}}})")


def _create_album_node(conn, artist_slug, album_name):
    album_id = f"{artist_slug}:album:{album_name.lower().replace(' ', '_')[:50]}"
    _safe_execute(conn,
        f"CREATE (a:Album {{id: '{_escape(album_id)}', "
        f"name: '{_escape(album_name)}', "
        f"release_date: '', artist_id: '{_escape(artist_slug)}', "
        f"song_count: 0}})")


def _create_mood_nodes(conn) -> set[str]:
    """Create the fixed set of mood nodes. Returns set of mood IDs."""
    moods = {
        "nostalgic":   (0.2,  0.3, "Warm longing for the past"),
        "romantic":    (0.7,  0.5, "Love, desire, tenderness"),
        "melancholic": (-0.3, 0.2, "Sadness, grief, sorrow"),
        "hopeful":     (0.6,  0.6, "Optimism, looking forward"),
        "peaceful":    (0.4,  0.1, "Calm, serene, contented"),
        "neutral":     (0.0,  0.3, "Balanced, neither positive nor negative"),
        "energetic":   (0.5,  0.9, "High energy, excitement, passion"),
        "angry":       (-0.6, 0.8, "Frustration, rage, intensity"),
        "devotional":  (0.5,  0.4, "Spiritual, surrendered, sufi"),
    }
    for mood_id, (valence, arousal, desc) in moods.items():
        _safe_execute(conn,
            f"CREATE (m:Mood {{id: '{mood_id}', name: '{mood_id}', "
            f"valence: {valence}, arousal: {arousal}, "
            f"description: '{_escape(desc)}'}})")
    return set(moods.keys())


def _create_song_node(conn, song_data):
    _safe_execute(conn,
        f"CREATE (s:Song {{id: '{_escape(song_data['id'])}', "
        f"title: '{_escape(song_data['title'])}', "
        f"artist_id: '{_escape(song_data['artist_id'])}', "
        f"album_id: '', "
        f"language: '{_escape(song_data.get('language', ''))}', "
        f"mood: '{_escape(song_data.get('mood', ''))}', "
        f"full_lyrics: '{_escape(song_data.get('full_lyrics_clean', '')[:5000])}', "
        f"url: '{_escape(song_data.get('url', ''))}', "
        f"line_count: {song_data.get('line_count', 0)}, "
        f"section_count: {song_data.get('section_count', 0)}, "
        f"word_count: {song_data.get('word_count', 0)}}})")


def _create_section_node(conn, sec_data):
    _safe_execute(conn,
        f"CREATE (sec:Section {{id: '{_escape(sec_data['id'])}', "
        f"song_id: '{_escape(sec_data['song_id'])}', "
        f"section_type: '{_escape(sec_data['section_type'])}', "
        f"section_index: {sec_data.get('section_index', 0)}, "
        f"text: '{_escape(sec_data.get('text', '')[:3000])}', "
        f"line_count: {sec_data.get('line_count', 0)}, "
        f"word_count: {sec_data.get('word_count', 0)}, "
        f"language: '{_escape(sec_data.get('language', ''))}', "
        f"mood: '{_escape(sec_data.get('mood', ''))}'}})")


def _create_line_node(conn, line_data):
    _safe_execute(conn,
        f"CREATE (l:Line {{id: '{_escape(line_data['id'])}', "
        f"section_id: '{_escape(line_data['section_id'])}', "
        f"song_id: '{_escape(line_data['song_id'])}', "
        f"line_index: {line_data.get('line_index', 0)}, "
        f"global_line_index: {line_data.get('global_line_index', 0)}, "
        f"text: '{_escape(line_data.get('text', '')[:1000])}', "
        f"romanized: '{_escape(line_data.get('romanized', ''))}', "
        f"word_count: {line_data.get('word_count', 0)}, "
        f"syllable_count: {line_data.get('syllable_count', 0)}, "
        f"language: '{_escape(line_data.get('language', ''))}', "
        f"has_code_switch: {str(line_data.get('has_code_switch', False)).lower()}, "
        f"end_word: '{_escape(line_data.get('end_word', ''))}'}})")


def _create_phrase_node(conn, phrase_data):
    _safe_execute(conn,
        f"CREATE (p:Phrase {{id: '{_escape(phrase_data['id'])}', "
        f"text: '{_escape(phrase_data['text'])}', "
        f"romanized: '{_escape(phrase_data.get('romanized', ''))}', "
        f"language: '{_escape(phrase_data.get('language', ''))}', "
        f"frequency: {phrase_data.get('frequency', 0)}, "
        f"artist_id: '{_escape(phrase_data.get('artist_id', ''))}', "
        f"is_signature: {str(phrase_data.get('is_signature', False)).lower()}'}})")


def _create_cultural_ref_node(conn, cr_data):
    _safe_execute(conn,
        f"CREATE (cr:CulturalReference {{id: '{_escape(cr_data['id'])}', "
        f"reference_text: '{_escape(cr_data['reference_text'])}', "
        f"category: '{_escape(cr_data['category'])}', "
        f"cultural_context: '{_escape(cr_data['cultural_context'])}', "
        f"artist_id: '{_escape(cr_data.get('artist_id', ''))}', "
        f"frequency: {cr_data.get('frequency', 0)}}})")


def _create_meter_pattern_node(conn, mp_data):
    _safe_execute(conn,
        f"CREATE (mp:MeterPattern {{id: '{_escape(mp_data['id'])}', "
        f"pattern: '{_escape(mp_data['pattern'])}', "
        f"artist_id: '{_escape(mp_data.get('artist_id', ''))}', "
        f"frequency: {mp_data.get('frequency', 0)}, "
        f"description: '{_escape(mp_data.get('description', ''))}'}})")
