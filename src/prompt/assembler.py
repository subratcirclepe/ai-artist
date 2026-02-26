"""Context assembly and dynamic prompt construction.

Assembles graph-derived insights into a two-part prompt architecture:
- System prompt: artist identity + constraints (computed from graph, not static YAML)
- User prompt: reference examples + generation task
"""

from src.retrieval.pipeline import RetrievalResult, RequestAnalysis


# ---------------------------------------------------------------------------
# Token budget constants
# ---------------------------------------------------------------------------

MAX_SYSTEM_TOKENS = 3500
MAX_USER_TOKENS = 8000


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def build_graph_system_prompt(
    artist_name: str,
    artist_slug: str,
    retrieval: RetrievalResult,
) -> str:
    """Build the system prompt from graph-derived data.

    This replaces the static YAML-based prompt with computed data.
    """
    parts = []

    # --- Core identity ---
    parts.append(
        f"You are {artist_name}'s creative consciousness. "
        f"You do not imitate — you ARE the creative process that produces {artist_name}'s music.\n"
    )

    # --- Structural instincts ---
    parts.append("## YOUR CREATIVE DNA\n")
    parts.append("### Structural Instincts")
    if retrieval.structures:
        top_structures = [s.get("pattern", "") for s in retrieval.structures[:3]]
        parts.append(f"You build songs as: {', '.join(top_structures)}")
    if retrieval.avg_lines_per_section:
        line_info = ", ".join(
            f"{k}: ~{v} lines" for k, v in retrieval.avg_lines_per_section.items()
        )
        parts.append(f"Section sizes: {line_info}")

    # --- Fingerprint stats ---
    fp = retrieval.fingerprint
    if fp:
        avg_ll = fp.get("avg_line_length", 0)
        avg_sl = fp.get("avg_section_length", 0)
        vocab_r = fp.get("vocabulary_richness", 0)
        csf = fp.get("code_switch_frequency", 0)
        if avg_ll > 0:
            parts.append(f"Your average line length: {avg_ll:.1f} words")
        if csf > 0.05:
            parts.append(f"You code-switch (mix Hindi/English) in ~{csf*100:.0f}% of lines")

    # --- Language rules ---
    parts.append("\n### Language Rules")
    if retrieval.vocabulary_clusters:
        vocab_sample = ", ".join(retrieval.vocabulary_clusters[:30])
        parts.append(f"Your vocabulary space includes: {vocab_sample}")
    if retrieval.anti_vocabulary:
        anti_sample = ", ".join(retrieval.anti_vocabulary[:20])
        parts.append(f"NEVER use these words: {anti_sample}")
    if retrieval.signature_phrases:
        phrase_sample = "; ".join(retrieval.signature_phrases[:10])
        parts.append(f"Your signature expressions: {phrase_sample}")

    # --- Rhyme DNA ---
    parts.append("\n### Rhyme DNA")
    if retrieval.rhyme_schemes.get("preferred_patterns"):
        patterns = ", ".join(retrieval.rhyme_schemes["preferred_patterns"][:5])
        parts.append(f"Preferred rhyme patterns: {patterns}")
    if retrieval.top_rhyme_pairs:
        pairs = ", ".join(
            f"{p['word_a']}/{p['word_b']}" for p in retrieval.top_rhyme_pairs[:8]
        )
        parts.append(f"Common rhyming pairs: {pairs}")

    # --- Emotional architecture ---
    parts.append("\n### Emotional Architecture")
    if retrieval.common_arcs:
        from collections import Counter
        arc_counter = Counter(a.get("arc_type", "") for a in retrieval.common_arcs)
        top_arcs = [f"{arc} ({count}x)" for arc, count in arc_counter.most_common(3)]
        parts.append(f"Typical emotional arcs: {', '.join(top_arcs)}")

    # --- Metaphor palette ---
    parts.append("\n### Metaphor Palette")
    if retrieval.metaphors:
        metaphor_lines = []
        for m in retrieval.metaphors[:8]:
            src = m.get("source_domain", "")
            tgt = m.get("target_domain", "")
            metaphor_lines.append(f"{src} → {tgt}")
        parts.append(f"Your signature metaphor domains: {'; '.join(metaphor_lines)}")

    # --- Cultural anchors ---
    if retrieval.cultural_references:
        refs = ", ".join(
            f"{r['reference']} ({r['category']})"
            for r in retrieval.cultural_references[:10]
        )
        parts.append(f"Cultural anchors you draw from: {refs}")

    # --- Absolute rules ---
    parts.append("\n## ABSOLUTE RULES")
    parts.append(f"1. Every line must pass this test: \"Would {artist_name} actually write this?\"")
    parts.append("2. NEVER use generic filler phrases or cliche Bollywood lines")
    parts.append("3. NEVER copy or closely paraphrase any of the reference lyrics shown to you")
    parts.append(f"4. Match {artist_name}'s EXACT emotional register — if they whisper, you whisper; if they build, you crescendo")
    parts.append("5. Maintain linguistic authenticity — use the RIGHT mix of Hindi/English/Urdu for this artist")

    # --- Chat rules ---
    parts.append("\n## CHAT RULES")
    parts.append(f"- When asked to write a song: produce complete lyrics with section labels")
    parts.append(f"- When asked about your creative process: respond as {artist_name} would")
    parts.append(f"- Always stay in character as {artist_name}'s creative persona")
    parts.append("- Be warm, genuine, and passionate about music")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# User prompt builder (for song generation)
# ---------------------------------------------------------------------------

def build_graph_generation_prompt(
    topic: str,
    artist_name: str,
    request: RequestAnalysis,
    retrieval: RetrievalResult,
) -> str:
    """Build the user-facing generation prompt with reference examples.

    This replaces the static generation_prompt from prompts.yaml.
    """
    parts = []

    # --- Reference lyrics ---
    if retrieval.thematic_sections:
        parts.append("## REFERENCE LYRICS (absorb the STYLE, never copy the WORDS)\n")

        # Select diverse section types for examples
        seen_types = set()
        shown = 0
        for sec in retrieval.thematic_sections:
            sec_type = sec.get("section_type", "section")
            if shown >= 4:
                break
            # Prefer diversity: one verse, one chorus, one bridge
            if sec_type in seen_types and shown >= 2:
                continue

            # Get song title from node_id
            node_id = sec.get("node_id", "")
            song_part = ":".join(node_id.split(":")[:2]) if ":" in node_id else ""
            song_title = song_part.split(":")[-1].replace("_", " ").title() if song_part else "Unknown"

            text_preview = sec.get("text", "")[:400]
            mood = sec.get("mood", "")
            line_count = sec.get("line_count", 0)

            parts.append(f"### Example {sec_type.replace('_', ' ').title()} (from \"{song_title}\"):")
            parts.append(text_preview)
            parts.append(f"[Structure: {line_count} lines, mood: {mood}]\n")

            seen_types.add(sec_type)
            shown += 1

    # --- Generation task ---
    parts.append("## YOUR TASK\n")
    parts.append(f"Write an original song about \"{topic}\" in the voice of {artist_name}.\n")

    # Structure recommendation
    if retrieval.structures:
        top_structure = retrieval.structures[0].get("pattern", "verse-chorus-verse-chorus")
        parts.append(f"Structure: {top_structure}")

    # Emotional arc recommendation
    if retrieval.common_arcs:
        from collections import Counter
        arc_counter = Counter(a.get("arc_type", "") for a in retrieval.common_arcs)
        top_arc = arc_counter.most_common(1)[0][0] if arc_counter else "gentle_rise"
        mood_signals = request.mood_signals
        if mood_signals:
            parts.append(
                f"Emotional arc: Start with {mood_signals[0]}, "
                f"evolve naturally (typical pattern: {top_arc})"
            )

    # Section targets
    if retrieval.avg_lines_per_section:
        verse_lines = retrieval.avg_lines_per_section.get("verse", 4)
        chorus_lines = retrieval.avg_lines_per_section.get("chorus", 4)
        parts.append(f"Target: ~{verse_lines} lines per verse, ~{chorus_lines} lines per chorus")

    # Format requirements
    parts.append("\nRequirements:")
    parts.append("- Include proper section labels [Verse 1], [Chorus], [Bridge], etc.")
    parts.append("- Include [emotional/delivery directions] in brackets: [softly], [building], [whispered]")
    parts.append("- If lyrics are in Hindi/Urdu: provide BOTH Devanagari script AND romanized transliteration")
    parts.append("- The song should feel like a genuine unreleased track, not an AI approximation")
    parts.append(f"- Approximate length: 200-300 words (3-4 minutes when sung)")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Chat prompt builder
# ---------------------------------------------------------------------------

def build_graph_chat_prompt(
    user_message: str,
    artist_name: str,
) -> str:
    """Build the user prompt for chat mode."""
    return (
        f"The user is chatting with you as {artist_name}'s creative persona. "
        f"Respond naturally and in character.\n\n"
        f"User: {user_message}"
    )
