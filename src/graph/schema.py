"""KuzuDB schema definitions for the Music Knowledge Graph.

Modeled after GitNexus's schema pattern: typed node tables, single relation table
with type property, separate embedding table with HNSW vector index.
"""

# ---------------------------------------------------------------------------
# Node table definitions
# ---------------------------------------------------------------------------

NODE_TABLE_QUERIES = [
    # --- Structural nodes ---
    """CREATE NODE TABLE IF NOT EXISTS Artist (
        id STRING,
        name STRING,
        slug STRING,
        language STRING,
        musical_style STRING,
        vocal_style STRING,
        vocabulary_level STRING,
        song_count INT32,
        total_line_count INT32,
        PRIMARY KEY (id)
    )""",

    """CREATE NODE TABLE IF NOT EXISTS Album (
        id STRING,
        name STRING,
        release_date STRING,
        artist_id STRING,
        song_count INT32,
        PRIMARY KEY (id)
    )""",

    """CREATE NODE TABLE IF NOT EXISTS Song (
        id STRING,
        title STRING,
        artist_id STRING,
        album_id STRING,
        language STRING,
        mood STRING,
        full_lyrics STRING,
        url STRING,
        line_count INT32,
        section_count INT32,
        word_count INT32,
        PRIMARY KEY (id)
    )""",

    """CREATE NODE TABLE IF NOT EXISTS Section (
        id STRING,
        song_id STRING,
        section_type STRING,
        section_index INT32,
        text STRING,
        line_count INT32,
        word_count INT32,
        language STRING,
        mood STRING,
        PRIMARY KEY (id)
    )""",

    """CREATE NODE TABLE IF NOT EXISTS Line (
        id STRING,
        section_id STRING,
        song_id STRING,
        line_index INT32,
        global_line_index INT32,
        text STRING,
        romanized STRING,
        word_count INT32,
        syllable_count INT32,
        language STRING,
        has_code_switch BOOL,
        end_word STRING,
        PRIMARY KEY (id)
    )""",

    # --- Linguistic nodes ---
    """CREATE NODE TABLE IF NOT EXISTS Phrase (
        id STRING,
        text STRING,
        romanized STRING,
        language STRING,
        frequency INT32,
        artist_id STRING,
        is_signature BOOL,
        PRIMARY KEY (id)
    )""",

    """CREATE NODE TABLE IF NOT EXISTS Metaphor (
        id STRING,
        source_text STRING,
        source_domain STRING,
        target_domain STRING,
        artist_id STRING,
        frequency INT32,
        PRIMARY KEY (id)
    )""",

    """CREATE NODE TABLE IF NOT EXISTS CulturalReference (
        id STRING,
        reference_text STRING,
        category STRING,
        cultural_context STRING,
        artist_id STRING,
        frequency INT32,
        PRIMARY KEY (id)
    )""",

    """CREATE NODE TABLE IF NOT EXISTS RhymePair (
        id STRING,
        word_a STRING,
        word_b STRING,
        rhyme_type STRING,
        language STRING,
        frequency INT32,
        artist_id STRING,
        PRIMARY KEY (id)
    )""",

    # --- Stylistic nodes ---
    """CREATE NODE TABLE IF NOT EXISTS Theme (
        id STRING,
        name STRING,
        description STRING,
        artist_id STRING,
        song_count INT32,
        PRIMARY KEY (id)
    )""",

    """CREATE NODE TABLE IF NOT EXISTS Mood (
        id STRING,
        name STRING,
        valence DOUBLE,
        arousal DOUBLE,
        description STRING,
        PRIMARY KEY (id)
    )""",

    """CREATE NODE TABLE IF NOT EXISTS MeterPattern (
        id STRING,
        pattern STRING,
        artist_id STRING,
        frequency INT32,
        description STRING,
        PRIMARY KEY (id)
    )""",

    """CREATE NODE TABLE IF NOT EXISTS StructureTemplate (
        id STRING,
        pattern STRING,
        artist_id STRING,
        frequency INT32,
        description STRING,
        PRIMARY KEY (id)
    )""",

    # --- Analytical / Computed nodes ---
    """CREATE NODE TABLE IF NOT EXISTS ThematicCluster (
        id STRING,
        label STRING,
        heuristic_label STRING,
        description STRING,
        enriched_by STRING,
        cohesion DOUBLE,
        song_count INT32,
        artist_id STRING,
        PRIMARY KEY (id)
    )""",

    """CREATE NODE TABLE IF NOT EXISTS EmotionalArc (
        id STRING,
        song_id STRING,
        arc_type STRING,
        description STRING,
        PRIMARY KEY (id)
    )""",

    """CREATE NODE TABLE IF NOT EXISTS StyleFingerprint (
        id STRING,
        artist_id STRING,
        avg_line_length DOUBLE,
        avg_section_length DOUBLE,
        vocabulary_richness DOUBLE,
        code_switch_frequency DOUBLE,
        metaphor_density DOUBLE,
        repetition_index DOUBLE,
        avg_mood_valence DOUBLE,
        avg_mood_arousal DOUBLE,
        PRIMARY KEY (id)
    )""",

    """CREATE NODE TABLE IF NOT EXISTS VocabularyCluster (
        id STRING,
        label STRING,
        artist_id STRING,
        frequency INT32,
        distinctiveness DOUBLE,
        PRIMARY KEY (id)
    )""",

    # --- Embedding table (separate, following GitNexus CodeEmbedding pattern) ---
    """CREATE NODE TABLE IF NOT EXISTS LyricEmbedding (
        node_id STRING,
        embedding FLOAT[384],
        PRIMARY KEY (node_id)
    )""",
]

# ---------------------------------------------------------------------------
# Relationship table definitions
# ---------------------------------------------------------------------------

# KuzuDB requires explicit FROM/TO type pairs for each relation table.
# We use a single LyricRelation table with a type property (GitNexus pattern).
REL_TABLE_QUERIES = [
    """CREATE REL TABLE IF NOT EXISTS WRITTEN_BY (
        FROM Song TO Artist,
        type STRING DEFAULT 'WRITTEN_BY'
    )""",

    """CREATE REL TABLE IF NOT EXISTS BELONGS_TO (
        FROM Song TO Album,
        track_number INT32,
        type STRING DEFAULT 'BELONGS_TO'
    )""",

    """CREATE REL TABLE IF NOT EXISTS CONTAINS_SECTION (
        FROM Song TO Section,
        type STRING DEFAULT 'CONTAINS_SECTION'
    )""",

    """CREATE REL TABLE IF NOT EXISTS CONTAINS_LINE (
        FROM Section TO Line,
        type STRING DEFAULT 'CONTAINS_LINE'
    )""",

    """CREATE REL TABLE IF NOT EXISTS SECTION_FOLLOWS (
        FROM Section TO Section,
        gap INT32,
        type STRING DEFAULT 'FOLLOWS'
    )""",

    """CREATE REL TABLE IF NOT EXISTS LINE_FOLLOWS (
        FROM Line TO Line,
        gap INT32,
        type STRING DEFAULT 'FOLLOWS'
    )""",

    """CREATE REL TABLE IF NOT EXISTS USES_PHRASE (
        FROM Line TO Phrase,
        position INT32,
        type STRING DEFAULT 'USES_PHRASE'
    )""",

    """CREATE REL TABLE IF NOT EXISTS CONTAINS_METAPHOR (
        FROM Line TO Metaphor,
        type STRING DEFAULT 'CONTAINS_METAPHOR'
    )""",

    """CREATE REL TABLE IF NOT EXISTS LINE_REFERENCES_CULTURE (
        FROM Line TO CulturalReference,
        type STRING DEFAULT 'REFERENCES_CULTURE'
    )""",

    """CREATE REL TABLE IF NOT EXISTS SONG_REFERENCES_CULTURE (
        FROM Song TO CulturalReference,
        type STRING DEFAULT 'REFERENCES_CULTURE'
    )""",

    """CREATE REL TABLE IF NOT EXISTS HAS_THEME (
        FROM Song TO Theme,
        strength DOUBLE,
        type STRING DEFAULT 'HAS_THEME'
    )""",

    """CREATE REL TABLE IF NOT EXISTS SECTION_EXPRESSES_MOOD (
        FROM Section TO Mood,
        intensity DOUBLE,
        type STRING DEFAULT 'EXPRESSES_MOOD'
    )""",

    """CREATE REL TABLE IF NOT EXISTS SONG_EXPRESSES_MOOD (
        FROM Song TO Mood,
        intensity DOUBLE,
        type STRING DEFAULT 'EXPRESSES_MOOD'
    )""",

    """CREATE REL TABLE IF NOT EXISTS RHYMES_WITH (
        FROM Line TO Line,
        rhyme_type STRING,
        pair_id STRING,
        type STRING DEFAULT 'RHYMES_WITH'
    )""",

    """CREATE REL TABLE IF NOT EXISTS MEMBER_OF_CLUSTER (
        FROM Song TO ThematicCluster,
        type STRING DEFAULT 'MEMBER_OF_CLUSTER'
    )""",

    """CREATE REL TABLE IF NOT EXISTS SONG_SIMILAR_TO (
        FROM Song TO Song,
        score DOUBLE,
        basis STRING,
        type STRING DEFAULT 'SIMILAR_TO'
    )""",

    """CREATE REL TABLE IF NOT EXISTS MOOD_TRANSITIONS_TO (
        FROM Mood TO Mood,
        frequency INT32,
        artist_id STRING,
        type STRING DEFAULT 'TRANSITIONS_TO'
    )""",

    """CREATE REL TABLE IF NOT EXISTS USES_STRUCTURE (
        FROM Song TO StructureTemplate,
        type STRING DEFAULT 'USES_STRUCTURE'
    )""",

    """CREATE REL TABLE IF NOT EXISTS HAS_ARC (
        FROM Song TO EmotionalArc,
        type STRING DEFAULT 'HAS_ARC'
    )""",

    """CREATE REL TABLE IF NOT EXISTS HAS_FINGERPRINT (
        FROM Artist TO StyleFingerprint,
        type STRING DEFAULT 'HAS_FINGERPRINT'
    )""",

    """CREATE REL TABLE IF NOT EXISTS HAS_VOCAB_CLUSTER (
        FROM Artist TO VocabularyCluster,
        type STRING DEFAULT 'HAS_VOCAB_CLUSTER'
    )""",

    """CREATE REL TABLE IF NOT EXISTS HAS_METER (
        FROM Section TO MeterPattern,
        type STRING DEFAULT 'HAS_METER'
    )""",
]

# ---------------------------------------------------------------------------
# Index definitions
# ---------------------------------------------------------------------------

INDEX_QUERIES = [
    # HNSW vector index on embeddings
    "CALL CREATE_VECTOR_INDEX('LyricEmbedding', 'lyric_embedding_idx', 'embedding', metric := 'cosine')",
]

# All node table names for iteration
NODE_TABLE_NAMES = [
    "Artist", "Album", "Song", "Section", "Line",
    "Phrase", "Metaphor", "CulturalReference", "RhymePair",
    "Theme", "Mood", "MeterPattern", "StructureTemplate",
    "ThematicCluster", "EmotionalArc", "StyleFingerprint",
    "VocabularyCluster", "LyricEmbedding",
]
