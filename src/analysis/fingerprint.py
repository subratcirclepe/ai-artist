"""Artist StyleFingerprint computation and LLM-powered analysis.

Computes statistical fingerprints from the structural analysis,
and uses LLM batch calls for metaphor extraction and theme assignment.
"""

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field, asdict

from src.utils import PROCESSED_DIR, load_artist_config
from src.preprocessor import estimate_mood, MOOD_KEYWORDS
from src.analysis.phonetics import detect_section_rhyme_scheme


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StyleFingerprintData:
    id: str
    artist_id: str
    avg_line_length: float          # avg words per line
    avg_section_length: float       # avg lines per section
    vocabulary_richness: float      # type-token ratio
    code_switch_frequency: float    # fraction of lines with code-switching
    metaphor_density: float         # metaphors per 100 words (set after LLM analysis)
    repetition_index: float         # fraction of repeated lines
    avg_mood_valence: float
    avg_mood_arousal: float
    top_rhyme_types: list[str] = field(default_factory=list)
    preferred_structures: list[str] = field(default_factory=list)
    vocabulary_set: list[str] = field(default_factory=list)       # top 500 words
    anti_vocabulary: list[str] = field(default_factory=list)      # words never used


@dataclass
class ThemeData:
    id: str
    name: str
    description: str
    artist_id: str
    song_count: int


@dataclass
class MetaphorData:
    id: str
    source_text: str
    source_domain: str
    target_domain: str
    artist_id: str
    frequency: int


@dataclass
class EmotionalArcData:
    id: str
    song_id: str
    arc_type: str  # gentle_rise, crescendo_crash, steady_melancholy, oscillating, slow_build
    mood_sequence: list[str] = field(default_factory=list)
    intensity_sequence: list[float] = field(default_factory=list)
    description: str = ""


# ---------------------------------------------------------------------------
# Mood valence/arousal mapping
# ---------------------------------------------------------------------------

MOOD_VA = {
    "nostalgic":   (0.2, 0.3),
    "romantic":    (0.7, 0.5),
    "melancholic": (-0.3, 0.2),
    "hopeful":     (0.6, 0.6),
    "peaceful":    (0.4, 0.1),
    "neutral":     (0.0, 0.3),
    "energetic":   (0.5, 0.9),
    "angry":       (-0.6, 0.8),
    "devotional":  (0.5, 0.4),
}


# ---------------------------------------------------------------------------
# StyleFingerprint computation
# ---------------------------------------------------------------------------

def compute_fingerprint(
    graph_data: dict,
    artist_slug: str,
) -> StyleFingerprintData:
    """Compute a statistical StyleFingerprint from analyzed graph data.

    Args:
        graph_data: Full analysis dict from lyric_analyzer.analyze_artist().
        artist_slug: Artist identifier.

    Returns:
        StyleFingerprintData with computed statistics.
    """
    songs = graph_data.get("songs", [])
    if not songs:
        return _empty_fingerprint(artist_slug)

    # Collect statistics across all songs
    all_line_lengths = []
    all_section_lengths = []
    all_words: list[str] = []
    code_switch_count = 0
    total_lines = 0
    line_texts: list[str] = []
    section_moods: list[str] = []
    rhyme_type_counter: Counter = Counter()
    structure_counter: Counter = Counter()

    for song in songs:
        sec_types = []
        for section in song.get("sections", []):
            section_lines = section.get("lines", [])
            all_section_lengths.append(len(section_lines))
            sec_types.append(section.get("section_type", ""))
            section_moods.append(section.get("mood", "neutral"))

            # Detect rhyme scheme for this section
            scheme = detect_section_rhyme_scheme(section_lines)
            rhyme_type_counter[scheme] += 1

            for line in section_lines:
                wc = line.get("word_count", 0)
                all_line_lengths.append(wc)
                total_lines += 1
                if line.get("has_code_switch", False):
                    code_switch_count += 1
                text = line.get("text", "")
                line_texts.append(text)
                words = text.lower().split()
                all_words.extend(words)

        pattern = "-".join(sec_types)
        structure_counter[pattern] += 1

    # Vocabulary richness (type-token ratio)
    unique_words = set(all_words)
    vocabulary_richness = len(unique_words) / max(len(all_words), 1)

    # Top vocabulary (most frequent 500 words)
    word_counter = Counter(all_words)
    top_words = [w for w, _ in word_counter.most_common(500)]

    # Repetition index (fraction of lines that appear more than once)
    line_counter = Counter(ln.strip().lower() for ln in line_texts if ln.strip())
    repeated = sum(1 for count in line_counter.values() if count > 1)
    repetition_index = repeated / max(len(line_counter), 1)

    # Average mood valence and arousal
    valences = [MOOD_VA.get(m, (0, 0.3))[0] for m in section_moods]
    arousals = [MOOD_VA.get(m, (0, 0.3))[1] for m in section_moods]
    avg_valence = sum(valences) / max(len(valences), 1)
    avg_arousal = sum(arousals) / max(len(arousals), 1)

    # Top rhyme types and structures
    top_rhyme_types = [rt for rt, _ in rhyme_type_counter.most_common(5)]
    preferred_structures = [st for st, _ in structure_counter.most_common(5)]

    return StyleFingerprintData(
        id=f"{artist_slug}:fingerprint",
        artist_id=artist_slug,
        avg_line_length=sum(all_line_lengths) / max(len(all_line_lengths), 1),
        avg_section_length=sum(all_section_lengths) / max(len(all_section_lengths), 1),
        vocabulary_richness=vocabulary_richness,
        code_switch_frequency=code_switch_count / max(total_lines, 1),
        metaphor_density=0.0,  # Updated after LLM analysis
        repetition_index=repetition_index,
        avg_mood_valence=avg_valence,
        avg_mood_arousal=avg_arousal,
        top_rhyme_types=top_rhyme_types,
        preferred_structures=preferred_structures,
        vocabulary_set=top_words,
        anti_vocabulary=[],  # Computed via cross-artist comparison
    )


def _empty_fingerprint(artist_slug: str) -> StyleFingerprintData:
    return StyleFingerprintData(
        id=f"{artist_slug}:fingerprint",
        artist_id=artist_slug,
        avg_line_length=0, avg_section_length=0, vocabulary_richness=0,
        code_switch_frequency=0, metaphor_density=0, repetition_index=0,
        avg_mood_valence=0, avg_mood_arousal=0,
    )


# ---------------------------------------------------------------------------
# Emotional arc computation
# ---------------------------------------------------------------------------

def compute_emotional_arcs(
    graph_data: dict,
    artist_slug: str,
) -> list[EmotionalArcData]:
    """Compute emotional arcs for each song based on section moods.

    Args:
        graph_data: Full analysis dict.
        artist_slug: Artist identifier.

    Returns:
        List of EmotionalArcData, one per song.
    """
    arcs = []
    for song in graph_data.get("songs", []):
        sections = song.get("sections", [])
        if len(sections) < 2:
            continue

        mood_seq = [sec.get("mood", "neutral") for sec in sections]
        intensity_seq = [MOOD_VA.get(m, (0, 0.3))[1] for m in mood_seq]

        arc_type = _classify_arc(intensity_seq)

        arc = EmotionalArcData(
            id=f"{song['id']}:arc",
            song_id=song["id"],
            arc_type=arc_type,
            mood_sequence=mood_seq,
            intensity_sequence=intensity_seq,
            description=f"{arc_type}: {' -> '.join(mood_seq)}",
        )
        arcs.append(arc)

    return arcs


def _classify_arc(intensities: list[float]) -> str:
    """Classify the shape of an emotional arc."""
    if not intensities or len(intensities) < 2:
        return "steady_melancholy"

    # Check for crescendo (steady increase then drop)
    peak_idx = intensities.index(max(intensities))
    valley_idx = intensities.index(min(intensities))

    diff = max(intensities) - min(intensities)
    if diff < 0.15:
        return "steady_melancholy"

    # Crescendo-crash: peak is in the latter half, then drops
    if peak_idx >= len(intensities) * 0.5 and peak_idx < len(intensities) - 1:
        if intensities[-1] < intensities[peak_idx] - 0.1:
            return "crescendo_crash"

    # Gentle rise: generally increasing
    increasing = sum(1 for i in range(1, len(intensities)) if intensities[i] >= intensities[i-1])
    if increasing >= len(intensities) * 0.6:
        return "gentle_rise"

    # Slow build: starts low, ends high
    if intensities[-1] > intensities[0] + 0.15:
        return "slow_build"

    # Oscillating: multiple direction changes
    changes = sum(1 for i in range(2, len(intensities))
                  if (intensities[i] - intensities[i-1]) * (intensities[i-1] - intensities[i-2]) < 0)
    if changes >= 2:
        return "oscillating"

    return "steady_melancholy"


# ---------------------------------------------------------------------------
# Theme extraction (keyword + LLM-assisted)
# ---------------------------------------------------------------------------

# Expanded theme keywords beyond the mood keywords
THEME_KEYWORDS = {
    "nostalgia_and_memories": [
        "yaad", "yaadein", "remember", "purana", "purani", "bachpan",
        "childhood", "woh din", "those days", "memory", "memories",
    ],
    "love_and_romance": [
        "pyaar", "love", "ishq", "dil", "heart", "tere", "tumse",
        "saath", "together", "humsafar", "mohabbat", "chahat",
    ],
    "heartbreak_and_separation": [
        "dard", "pain", "rona", "cry", "tears", "aansu", "tanha",
        "alone", "lonely", "akela", "judai", "separation", "alvida",
    ],
    "hope_and_dreams": [
        "umeed", "hope", "naya", "new", "sapna", "dream", "light",
        "roshni", "tomorrow", "kal", "sunrise", "udaan",
    ],
    "nature_and_seasons": [
        "baarish", "rain", "monsoon", "sawan", "hawa", "wind",
        "badal", "clouds", "aasmaan", "sky", "sitare", "stars",
        "dhoop", "sunshine", "patthar", "stone", "nadi", "river",
    ],
    "spiritual_and_sufi": [
        "khuda", "allah", "rab", "ishq", "rooh", "soul", "fana",
        "qawwali", "dargah", "sufi", "maula", "dervish",
    ],
    "urban_life": [
        "sheher", "city", "sadak", "road", "traffic", "crowd",
        "building", "apartment", "metro", "cafe",
    ],
    "home_and_belonging": [
        "ghar", "home", "maa", "mother", "papa", "father",
        "family", "gali", "street", "gaon", "village",
    ],
    "journey_and_travel": [
        "safar", "journey", "raasta", "path", "musafir", "traveler",
        "train", "station", "chal", "walk", "door", "far",
    ],
    "self_reflection": [
        "main", "khud", "self", "mirror", "aaina", "soch",
        "thought", "zindagi", "life", "waqt", "time",
    ],
}


def extract_themes(
    graph_data: dict,
    artist_slug: str,
) -> list[ThemeData]:
    """Extract themes from songs using keyword matching."""
    theme_song_count: Counter = Counter()

    for song in graph_data.get("songs", []):
        text_lower = song.get("full_lyrics_clean", "").lower()
        for theme_name, keywords in THEME_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score >= 2:  # At least 2 keyword matches
                theme_song_count[theme_name] += 1

    themes = []
    for theme_name, count in theme_song_count.most_common():
        theme_id = f"{artist_slug}:theme:{theme_name}"
        display_name = theme_name.replace("_", " ").title()
        themes.append(ThemeData(
            id=theme_id,
            name=display_name,
            description=f"Songs about {display_name.lower()}",
            artist_id=artist_slug,
            song_count=count,
        ))

    return themes


# ---------------------------------------------------------------------------
# LLM-powered metaphor extraction
# ---------------------------------------------------------------------------

def extract_metaphors_with_llm(
    graph_data: dict,
    artist_slug: str,
) -> list[MetaphorData]:
    """Extract metaphors from lyrics using LLM batch analysis.

    Sends batches of song sections to Claude for metaphor identification.
    Falls back to keyword-based extraction if no LLM API key is available.

    Returns:
        List of MetaphorData.
    """
    # Try LLM-based extraction first
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key and not api_key.startswith("your_"):
        return _llm_metaphor_extraction(graph_data, artist_slug, api_key)

    # Fallback: keyword-based metaphor detection
    return _keyword_metaphor_extraction(graph_data, artist_slug)


def _llm_metaphor_extraction(
    graph_data: dict,
    artist_slug: str,
    api_key: str,
) -> list[MetaphorData]:
    """Use Claude to extract metaphors from lyrics."""
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        return _keyword_metaphor_extraction(graph_data, artist_slug)

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        anthropic_api_key=api_key,
        temperature=0.0,
        max_tokens=2000,
    )

    all_metaphors: list[MetaphorData] = []
    metaphor_counter: Counter = Counter()

    # Batch sections (10 at a time)
    sections_batch = []
    for song in graph_data.get("songs", []):
        for section in song.get("sections", []):
            if section.get("text", "").strip():
                sections_batch.append({
                    "song": song["title"],
                    "type": section["section_type"],
                    "text": section["text"][:500],
                })

    # Process in batches of 10
    for i in range(0, min(len(sections_batch), 100), 10):
        batch = sections_batch[i:i+10]
        batch_text = "\n\n".join(
            f"[{s['song']} - {s['type']}]\n{s['text']}" for s in batch
        )

        try:
            response = llm.invoke([
                SystemMessage(content=(
                    "You are a literary analyst specializing in Hindi/English song lyrics. "
                    "Identify metaphors in the following lyrics. For each metaphor, output EXACTLY "
                    "one line in this format:\n"
                    "METAPHOR: <source text> | <source domain> | <target domain>\n\n"
                    "Example:\n"
                    "METAPHOR: dil ka musafir | travel/journey | love/emotion\n"
                    "METAPHOR: baarish mein bheega | rain/weather | longing/sadness\n\n"
                    "Only list clear metaphors, not literal descriptions. "
                    "If no metaphors found, output: NONE"
                )),
                HumanMessage(content=batch_text),
            ])

            # Parse response
            for line in response.content.split("\n"):
                line = line.strip()
                if line.startswith("METAPHOR:"):
                    parts = line[len("METAPHOR:"):].strip().split("|")
                    if len(parts) == 3:
                        source_text = parts[0].strip()
                        source_domain = parts[1].strip()
                        target_domain = parts[2].strip()
                        key = (source_domain, target_domain)
                        metaphor_counter[key] += 1

                        if key not in {(m.source_domain, m.target_domain) for m in all_metaphors}:
                            mid = f"{artist_slug}:metaphor:{len(all_metaphors)}"
                            all_metaphors.append(MetaphorData(
                                id=mid,
                                source_text=source_text,
                                source_domain=source_domain,
                                target_domain=target_domain,
                                artist_id=artist_slug,
                                frequency=1,
                            ))
        except Exception as e:
            print(f"  LLM metaphor extraction batch error: {e}")
            continue

    # Update frequencies
    for m in all_metaphors:
        key = (m.source_domain, m.target_domain)
        m.frequency = metaphor_counter.get(key, 1)

    return all_metaphors


# Keyword-based metaphor detection patterns
_METAPHOR_PATTERNS = {
    ("rain/weather", "longing/sadness"): ["baarish", "barish", "barsat", "sawan", "bheeg"],
    ("journey/travel", "love/life"): ["safar", "raasta", "musafir", "door", "manzil"],
    ("light/darkness", "hope/despair"): ["roshni", "andhera", "ujala", "suraj", "chaand"],
    ("ocean/shore", "love/distance"): ["samandar", "kinara", "lehren", "wave", "ocean"],
    ("fire/flame", "passion"): ["aag", "jalana", "sholay", "flame", "burn"],
    ("wind/breeze", "freedom/change"): ["hawa", "breeze", "toofan", "storm"],
    ("mirror/reflection", "self/truth"): ["aaina", "mirror", "reflection", "shadow"],
    ("flowers/garden", "beauty/love"): ["phool", "gulab", "garden", "bagh", "khilna"],
    ("chains/cage", "imprisonment/restriction"): ["zanjeer", "pinjra", "cage", "bandhan"],
    ("stars/sky", "dreams/destiny"): ["sitare", "aasmaan", "sky", "stars", "chaand"],
}


def _keyword_metaphor_extraction(
    graph_data: dict,
    artist_slug: str,
) -> list[MetaphorData]:
    """Fallback keyword-based metaphor detection."""
    metaphor_counter: Counter = Counter()

    for song in graph_data.get("songs", []):
        text_lower = song.get("full_lyrics_clean", "").lower()
        for (source, target), keywords in _METAPHOR_PATTERNS.items():
            for kw in keywords:
                if kw in text_lower:
                    metaphor_counter[(source, target)] += 1
                    break  # Count once per domain pair per song

    results = []
    for (source, target), freq in metaphor_counter.most_common():
        mid = f"{artist_slug}:metaphor:{source.split('/')[0]}_{target.split('/')[0]}"
        results.append(MetaphorData(
            id=mid,
            source_text=f"{source} as {target}",
            source_domain=source,
            target_domain=target,
            artist_id=artist_slug,
            frequency=freq,
        ))

    return results


# ---------------------------------------------------------------------------
# Full analysis pipeline (Phase 4)
# ---------------------------------------------------------------------------

def run_advanced_analysis(artist_slug: str, graph_data: dict | None = None) -> dict:
    """Run LLM-powered analysis: metaphors, themes, emotional arcs, fingerprint.

    Args:
        artist_slug: Artist identifier.
        graph_data: Pre-loaded graph data. If None, loads from disk.

    Returns:
        Dict with themes, metaphors, emotional_arcs, fingerprint.
    """
    if graph_data is None:
        data_path = PROCESSED_DIR / f"{artist_slug}_graph_data.json"
        with open(data_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

    print(f"Running advanced analysis for {artist_slug}...")

    # Extract themes
    themes = extract_themes(graph_data, artist_slug)
    print(f"  Extracted {len(themes)} themes")

    # Extract metaphors (LLM or keyword fallback)
    metaphors = extract_metaphors_with_llm(graph_data, artist_slug)
    print(f"  Extracted {len(metaphors)} metaphors")

    # Compute emotional arcs
    arcs = compute_emotional_arcs(graph_data, artist_slug)
    print(f"  Computed {len(arcs)} emotional arcs")

    # Compute fingerprint
    fingerprint = compute_fingerprint(graph_data, artist_slug)
    # Update metaphor density
    total_words = graph_data.get("stats", {}).get("total_words", 1)
    fingerprint.metaphor_density = len(metaphors) / max(total_words / 100, 1)
    print(f"  Computed style fingerprint")

    result = {
        "themes": [asdict(t) for t in themes],
        "metaphors": [asdict(m) for m in metaphors],
        "emotional_arcs": [asdict(a) for a in arcs],
        "fingerprint": asdict(fingerprint),
    }

    # Save to disk
    output_path = PROCESSED_DIR / f"{artist_slug}_advanced_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved advanced analysis to {output_path}")

    return result
