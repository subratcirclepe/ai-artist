"""Cypher query templates for the Music Knowledge Graph.

Centralized query definitions for all retrieval stages and graph operations.
"""


# ---------------------------------------------------------------------------
# Graph statistics
# ---------------------------------------------------------------------------

GRAPH_STATS = """
MATCH (s:Song) WHERE s.artist_id = $artist_id
RETURN count(s) AS song_count
"""

SECTION_STATS = """
MATCH (sec:Section) WHERE sec.song_id STARTS WITH $prefix
RETURN sec.section_type AS type, count(sec) AS count, avg(sec.line_count) AS avg_lines
"""

LINE_STATS = """
MATCH (l:Line) WHERE l.song_id STARTS WITH $prefix
RETURN count(l) AS line_count, avg(l.word_count) AS avg_word_count
"""

# ---------------------------------------------------------------------------
# Retrieval queries
# ---------------------------------------------------------------------------

# Stage 2: Vocabulary patterns
TOP_PHRASES = """
MATCH (p:Phrase) WHERE p.artist_id = $artist_id AND p.frequency >= 3
RETURN p.text AS text, p.frequency AS freq
ORDER BY p.frequency DESC LIMIT 30
"""

# Stage 3: Rhyme pairs
TOP_RHYME_PAIRS = """
MATCH (rp:RhymePair) WHERE rp.artist_id = $artist_id
RETURN rp.word_a AS word_a, rp.word_b AS word_b,
       rp.rhyme_type AS rhyme_type, rp.frequency AS freq
ORDER BY rp.frequency DESC LIMIT 20
"""

# Stage 4: Emotional arcs
EMOTIONAL_ARCS = """
MATCH (a:EmotionalArc) WHERE a.song_id STARTS WITH $prefix
RETURN a.arc_type AS arc_type, a.description AS description
LIMIT 20
"""

# Stage 5: Metaphors
ARTIST_METAPHORS = """
MATCH (m:Metaphor) WHERE m.artist_id = $artist_id
RETURN m.source_text AS source_text, m.source_domain AS source_domain,
       m.target_domain AS target_domain, m.frequency AS freq
ORDER BY m.frequency DESC LIMIT 15
"""

# Stage 6: Cultural references
CULTURAL_REFS = """
MATCH (cr:CulturalReference) WHERE cr.artist_id = $artist_id
RETURN cr.reference_text AS reference, cr.category AS category,
       cr.cultural_context AS context, cr.frequency AS freq
ORDER BY cr.frequency DESC LIMIT 15
"""

# Stage 7: Structure templates
STRUCTURE_TEMPLATES = """
MATCH (t:StructureTemplate) WHERE t.artist_id = $artist_id
RETURN t.pattern AS pattern, t.frequency AS freq, t.description AS description
ORDER BY t.frequency DESC LIMIT 5
"""

SECTION_AVG_LINES = """
MATCH (sec:Section) WHERE sec.song_id STARTS WITH $prefix
RETURN sec.section_type AS section_type, avg(sec.line_count) AS avg_lines
"""

# Fingerprint
STYLE_FINGERPRINT = """
MATCH (f:StyleFingerprint) WHERE f.artist_id = $artist_id
RETURN f
"""

# ---------------------------------------------------------------------------
# Validation queries
# ---------------------------------------------------------------------------

ALL_LINES_FOR_ARTIST = """
MATCH (l:Line) WHERE l.song_id STARTS WITH $prefix
RETURN l.text AS text
"""

# ---------------------------------------------------------------------------
# Exploration queries
# ---------------------------------------------------------------------------

ARTIST_OVERVIEW = """
MATCH (a:Artist) WHERE a.id = $artist_id
OPTIONAL MATCH (a)-[:HAS_FINGERPRINT]->(f:StyleFingerprint)
RETURN a, f
"""

SONGS_FOR_ARTIST = """
MATCH (s:Song) WHERE s.artist_id = $artist_id
RETURN s.id AS id, s.title AS title, s.mood AS mood,
       s.language AS language, s.section_count AS sections,
       s.line_count AS lines, s.word_count AS words
ORDER BY s.title
"""

THEMES_FOR_ARTIST = """
MATCH (t:Theme) WHERE t.artist_id = $artist_id
RETURN t.name AS name, t.description AS description, t.song_count AS song_count
ORDER BY t.song_count DESC
"""

CLUSTERS_FOR_ARTIST = """
MATCH (tc:ThematicCluster) WHERE tc.artist_id = $artist_id
RETURN tc.label AS label, tc.description AS description,
       tc.song_count AS song_count, tc.cohesion AS cohesion
ORDER BY tc.song_count DESC
"""
