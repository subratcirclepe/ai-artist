"""Structural decomposition and linguistic analysis of lyrics.

Decomposes raw lyrics into Song -> Section -> Line hierarchy and extracts
linguistic features (phrases, cultural references, meter patterns).
"""

import re
import json
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

from src.utils import slugify, RAW_DIR, PROCESSED_DIR, ensure_dirs
from src.preprocessor import detect_language, estimate_mood


# ---------------------------------------------------------------------------
# Data classes for the structural hierarchy
# ---------------------------------------------------------------------------

@dataclass
class LineData:
    id: str
    section_id: str
    song_id: str
    line_index: int
    global_line_index: int
    text: str
    romanized: str
    word_count: int
    syllable_count: int
    language: str
    has_code_switch: bool
    end_word: str


@dataclass
class SectionData:
    id: str
    song_id: str
    section_type: str  # verse, chorus, bridge, pre_chorus, outro, interlude, mukhda, antara
    section_index: int
    text: str
    line_count: int
    word_count: int
    language: str
    mood: str
    lines: list[LineData] = field(default_factory=list)


@dataclass
class SongData:
    id: str
    title: str
    artist_id: str
    album: str
    year: int | None
    language: str
    mood: str
    full_lyrics: str
    full_lyrics_clean: str  # without section headers
    url: str
    line_count: int
    section_count: int
    word_count: int
    sections: list[SectionData] = field(default_factory=list)


@dataclass
class PhraseData:
    id: str
    text: str
    romanized: str
    language: str
    frequency: int
    artist_id: str
    is_signature: bool


@dataclass
class CulturalReferenceData:
    id: str
    reference_text: str
    category: str  # food, weather, literature, place, festival, daily_life, emotion
    cultural_context: str
    artist_id: str
    frequency: int


@dataclass
class MeterPatternData:
    id: str
    pattern: str  # e.g. "5-7-5-7" (syllable counts per line)
    artist_id: str
    frequency: int
    description: str


# ---------------------------------------------------------------------------
# Section header parsing
# ---------------------------------------------------------------------------

# Map Genius section header text to normalized types
SECTION_TYPE_MAP = {
    "verse": "verse",
    "chorus": "chorus",
    "bridge": "bridge",
    "pre-chorus": "pre_chorus",
    "pre chorus": "pre_chorus",
    "prechorus": "pre_chorus",
    "hook": "chorus",
    "refrain": "chorus",
    "outro": "outro",
    "intro": "intro",
    "interlude": "interlude",
    "post-chorus": "post_chorus",
    "post chorus": "post_chorus",
    # Bollywood/Indian music terms
    "mukhda": "mukhda",
    "mukhra": "mukhda",
    "antara": "antara",
    "sthayi": "mukhda",
    "sanchari": "bridge",
    "abhog": "outro",
}

SECTION_HEADER_RE = re.compile(r"^\[([^\]]+)\]$", re.MULTILINE)


def _normalize_section_type(header_text: str) -> tuple[str, int]:
    """Parse a section header like 'Verse 2' into (type, index).

    Returns:
        (section_type, section_number) — number defaults to 1.
    """
    text = header_text.strip().lower()

    # Extract trailing number if present: "Verse 2" -> ("verse", 2)
    num_match = re.search(r"(\d+)\s*$", text)
    number = int(num_match.group(1)) if num_match else 0
    text_no_num = re.sub(r"\s*\d+\s*$", "", text).strip()

    # Try direct match
    for pattern, stype in SECTION_TYPE_MAP.items():
        if pattern in text_no_num:
            return stype, number

    # Fallback: treat as generic section
    return "section", number


# ---------------------------------------------------------------------------
# Syllable estimation
# ---------------------------------------------------------------------------

def _estimate_syllables(text: str) -> int:
    """Estimate syllable count for Hindi/English mixed text."""
    # For Devanagari: each vowel matra or standalone vowel is roughly one syllable
    devanagari_vowels = len(re.findall(
        r"[\u0904-\u0914\u093E-\u094C\u0962\u0963]", text
    ))
    devanagari_consonants = len(re.findall(r"[\u0915-\u0939]", text))

    # For Latin text: simple English syllable heuristic
    latin_words = re.findall(r"[a-zA-Z]+", text)
    latin_syllables = 0
    for word in latin_words:
        word = word.lower()
        count = len(re.findall(r"[aeiouy]+", word))
        if word.endswith("e") and count > 1:
            count -= 1
        latin_syllables += max(count, 1)

    # Devanagari syllable estimate: vowels + consonants without virama
    hindi_syllables = max(devanagari_vowels, devanagari_consonants // 2)

    return hindi_syllables + latin_syllables


def _detect_code_switch(text: str) -> bool:
    """Detect if a line switches between Hindi and English mid-line."""
    # Find segments of Devanagari and Latin
    segments = re.findall(r"[\u0900-\u097F]+|[a-zA-Z]+", text)
    if len(segments) < 2:
        return False

    has_hindi = any(re.match(r"[\u0900-\u097F]", s) for s in segments)
    has_english = any(re.match(r"[a-zA-Z]", s) for s in segments)
    return has_hindi and has_english


def _get_end_word(text: str) -> str:
    """Extract the last word of a line for rhyme analysis."""
    words = re.findall(r"[\u0900-\u097F]+|[a-zA-Z]+", text.strip())
    return words[-1].lower() if words else ""


# ---------------------------------------------------------------------------
# Structural decomposition
# ---------------------------------------------------------------------------

def decompose_song(
    raw_song: dict,
    artist_slug: str,
) -> SongData:
    """Decompose a raw song dict into structured Song -> Section -> Line hierarchy.

    Args:
        raw_song: Dict with keys: title, album, year, lyrics, url.
        artist_slug: The artist identifier slug.

    Returns:
        Fully decomposed SongData with sections and lines.
    """
    song_slug = slugify(raw_song["title"])
    song_id = f"{artist_slug}:{song_slug}"
    lyrics = raw_song["lyrics"]

    # Split lyrics into sections using [Header] markers
    parts = SECTION_HEADER_RE.split(lyrics)
    # parts alternates: [text_before_first_header, header1, text1, header2, text2, ...]

    sections: list[SectionData] = []
    global_line_idx = 0
    section_type_counters: Counter = Counter()

    # If lyrics start with text before any header, treat as implicit first section
    start_idx = 0
    if parts and parts[0].strip():
        # Text before first header — treat as implicit verse
        section_type_counters["verse"] += 1
        sec, global_line_idx = _build_section(
            header_type="verse",
            header_number=section_type_counters["verse"],
            section_index=len(sections),
            text=parts[0],
            song_id=song_id,
            artist_slug=artist_slug,
            global_line_start=global_line_idx,
        )
        if sec.line_count > 0:
            sections.append(sec)
        start_idx = 1
    elif parts and not parts[0].strip():
        start_idx = 1

    # Process header-text pairs
    for i in range(start_idx, len(parts) - 1, 2):
        header_text = parts[i]
        section_text = parts[i + 1] if (i + 1) < len(parts) else ""

        sec_type, sec_num = _normalize_section_type(header_text)
        if sec_num == 0:
            section_type_counters[sec_type] += 1
            sec_num = section_type_counters[sec_type]
        else:
            section_type_counters[sec_type] = max(
                section_type_counters[sec_type], sec_num
            )

        sec, global_line_idx = _build_section(
            header_type=sec_type,
            header_number=sec_num,
            section_index=len(sections),
            text=section_text,
            song_id=song_id,
            artist_slug=artist_slug,
            global_line_start=global_line_idx,
        )
        if sec.line_count > 0:
            sections.append(sec)

    # If no sections were found (no headers at all), treat entire lyrics as one section
    if not sections:
        section_type_counters["verse"] += 1
        sec, global_line_idx = _build_section(
            header_type="verse",
            header_number=1,
            section_index=0,
            text=lyrics,
            song_id=song_id,
            artist_slug=artist_slug,
            global_line_start=0,
        )
        if sec.line_count > 0:
            sections.append(sec)

    # Build clean lyrics (without section headers) for embedding
    clean_lyrics = re.sub(r"\[.*?\]", "", lyrics)
    clean_lyrics = re.sub(r"\n{3,}", "\n\n", clean_lyrics).strip()

    total_lines = sum(s.line_count for s in sections)
    total_words = sum(s.word_count for s in sections)

    album_name = ""
    if raw_song.get("album"):
        if isinstance(raw_song["album"], dict):
            album_name = raw_song["album"].get("name", "")
        else:
            album_name = str(raw_song["album"])

    return SongData(
        id=song_id,
        title=raw_song["title"],
        artist_id=artist_slug,
        album=album_name,
        year=raw_song.get("year"),
        language=detect_language(clean_lyrics),
        mood=estimate_mood(clean_lyrics),
        full_lyrics=lyrics,
        full_lyrics_clean=clean_lyrics,
        url=raw_song.get("url", ""),
        line_count=total_lines,
        section_count=len(sections),
        word_count=total_words,
        sections=sections,
    )


def _build_section(
    header_type: str,
    header_number: int,
    section_index: int,
    text: str,
    song_id: str,
    artist_slug: str,
    global_line_start: int,
) -> tuple[SectionData, int]:
    """Build a SectionData from raw text.

    Returns:
        (SectionData, next_global_line_index)
    """
    section_id = f"{song_id}:{header_type}_{header_number}"

    raw_lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    lines: list[LineData] = []
    global_idx = global_line_start

    for i, line_text in enumerate(raw_lines):
        line_id = f"{section_id}:line_{i}"
        word_count = len(line_text.split())
        lines.append(LineData(
            id=line_id,
            section_id=section_id,
            song_id=song_id,
            line_index=i,
            global_line_index=global_idx,
            text=line_text,
            romanized="",  # Filled later by transliteration step
            word_count=word_count,
            syllable_count=_estimate_syllables(line_text),
            language=detect_language(line_text),
            has_code_switch=_detect_code_switch(line_text),
            end_word=_get_end_word(line_text),
        ))
        global_idx += 1

    section_text = "\n".join(ln.text for ln in lines)
    section_word_count = sum(ln.word_count for ln in lines)

    return SectionData(
        id=section_id,
        song_id=song_id,
        section_type=header_type,
        section_index=section_index,
        text=section_text,
        line_count=len(lines),
        word_count=section_word_count,
        language=detect_language(section_text) if section_text else "unknown",
        mood=estimate_mood(section_text) if section_text else "neutral",
        lines=lines,
    ), global_idx


# ---------------------------------------------------------------------------
# Phrase extraction
# ---------------------------------------------------------------------------

def extract_phrases(
    songs: list[SongData],
    artist_slug: str,
    min_frequency: int = 3,
    ngram_range: tuple[int, int] = (2, 5),
) -> list[PhraseData]:
    """Extract recurring multi-word phrases across an artist's catalog.

    Args:
        songs: List of decomposed SongData.
        artist_slug: Artist identifier.
        min_frequency: Minimum occurrences to consider a phrase.
        ngram_range: (min_n, max_n) for n-gram extraction.

    Returns:
        List of PhraseData for phrases meeting the frequency threshold.
    """
    ngram_counter: Counter = Counter()

    for song in songs:
        for section in song.sections:
            for line in section.lines:
                words = line.text.lower().split()
                for n in range(ngram_range[0], min(ngram_range[1] + 1, len(words) + 1)):
                    for i in range(len(words) - n + 1):
                        ngram = " ".join(words[i : i + n])
                        ngram_counter[ngram] += 1

    phrases = []
    for text, freq in ngram_counter.most_common():
        if freq < min_frequency:
            break
        # Skip very common Hindi/English stop-word-only phrases
        if _is_stopword_phrase(text):
            continue
        phrase_id = f"{artist_slug}:phrase:{slugify(text[:50])}"
        phrases.append(PhraseData(
            id=phrase_id,
            text=text,
            romanized="",
            language=detect_language(text),
            frequency=freq,
            artist_id=artist_slug,
            is_signature=False,  # Set later when cross-artist comparison is done
        ))

    return phrases


# Common Hindi/English stop words to filter from phrase extraction
_STOP_WORDS = {
    "the", "a", "an", "is", "was", "are", "were", "in", "on", "at", "to",
    "of", "and", "or", "but", "for", "with", "that", "this", "it", "i",
    "you", "he", "she", "we", "they", "me", "my", "your", "his", "her",
    "hai", "hain", "ka", "ki", "ke", "ko", "se", "ne", "mein", "par",
    "ye", "wo", "yeh", "woh", "aur", "ya", "nahi", "na", "ho", "tha",
    "thi", "the", "hum", "tum", "main", "tu", "mere", "tera", "tere",
}


def _is_stopword_phrase(text: str) -> bool:
    """Check if a phrase is composed entirely of stop words."""
    words = set(text.lower().split())
    return words.issubset(_STOP_WORDS)


# ---------------------------------------------------------------------------
# Cultural reference detection (keyword-based)
# ---------------------------------------------------------------------------

CULTURAL_REFERENCES = {
    "food": {
        "chai": "Tea — represents comfort, nostalgia, warmth in Indian culture",
        "coffee": "Coffee — represents modern, urban connection",
        "mithai": "Sweets — represents celebration, sweetness of life",
    },
    "weather": {
        "baarish": "Rain — represents longing, romance, monsoon melancholy",
        "barish": "Rain — represents longing, romance, monsoon melancholy",
        "barsat": "Rainy season — represents emotional cleansing, renewal",
        "monsoon": "Monsoon — represents intense emotion, romance",
        "sawan": "Monsoon month — represents romantic longing in Indian tradition",
        "dhoop": "Sunshine — represents warmth, happiness, clarity",
        "hawa": "Wind/breeze — represents freedom, gentle emotion",
        "badal": "Clouds — represents uncertainty, hidden emotion",
    },
    "daily_life": {
        "gali": "Street/lane — represents neighborhood, belonging",
        "galiyan": "Streets/lanes — represents childhood, home",
        "chaabiyan": "Keys — represents access, secrets, home",
        "chitthi": "Letter — represents old-fashioned connection, distance",
        "khat": "Letter — represents poetic, Urdu tradition of correspondence",
        "train": "Train — represents journey, departure, distance",
        "khidki": "Window — represents perspective, longing to see outside",
        "sheher": "City — represents modern life, distance from roots",
        "gaon": "Village — represents simplicity, roots, nostalgia",
    },
    "emotion": {
        "ishq": "Deep love — Urdu/Sufi tradition of divine/passionate love",
        "junoon": "Obsession/passion — intensity beyond reason",
        "dard": "Pain — poetic suffering, heartache",
        "sukoon": "Peace/tranquility — inner calm",
        "yaadein": "Memories — nostalgic recollections",
        "mohabbat": "Love — tender, literary Hindi/Urdu love",
        "intezaar": "Waiting — patient longing, anticipation",
        "judai": "Separation — distance from beloved",
        "alvida": "Farewell — goodbye with emotional weight",
    },
    "literature": {
        "shayari": "Urdu poetry tradition",
        "ghazal": "Urdu/Persian poetic form — love and loss",
        "nazm": "Free verse Urdu poem",
        "qissa": "Story/tale — narrative tradition",
    },
    "place": {
        "dilli": "Delhi — capital, urban life",
        "mumbai": "Mumbai — dreams, Bollywood, ambition",
        "kashmir": "Kashmir — paradise, beauty, longing",
        "ganga": "River Ganges — spirituality, purification",
    },
    "festival": {
        "diwali": "Festival of lights — celebration, hope, new beginnings",
        "holi": "Festival of colors — joy, freedom, playfulness",
        "eid": "Islamic festival — celebration, togetherness",
    },
}


def extract_cultural_references(
    songs: list[SongData],
    artist_slug: str,
) -> list[CulturalReferenceData]:
    """Extract cultural references from lyrics using keyword matching."""
    ref_counts: Counter = Counter()

    for song in songs:
        text_lower = song.full_lyrics_clean.lower()
        for category, refs in CULTURAL_REFERENCES.items():
            for keyword in refs:
                count = text_lower.count(keyword.lower())
                if count > 0:
                    ref_counts[(keyword, category)] += count

    results = []
    for (keyword, category), freq in ref_counts.most_common():
        ref_id = f"{artist_slug}:cultural:{slugify(keyword)}"
        context = CULTURAL_REFERENCES[category][keyword]
        results.append(CulturalReferenceData(
            id=ref_id,
            reference_text=keyword,
            category=category,
            cultural_context=context,
            artist_id=artist_slug,
            frequency=freq,
        ))

    return results


# ---------------------------------------------------------------------------
# Meter pattern extraction
# ---------------------------------------------------------------------------

def extract_meter_patterns(
    songs: list[SongData],
    artist_slug: str,
    min_frequency: int = 2,
) -> list[MeterPatternData]:
    """Extract syllable-count-per-line patterns from sections."""
    pattern_counter: Counter = Counter()

    for song in songs:
        for section in song.sections:
            if section.line_count < 2:
                continue
            # Create pattern from syllable counts
            syllable_counts = [ln.syllable_count for ln in section.lines]
            pattern = "-".join(str(s) for s in syllable_counts)
            pattern_counter[(pattern, section.section_type)] += 1

    results = []
    for (pattern, sec_type), freq in pattern_counter.most_common():
        if freq < min_frequency:
            break
        pattern_id = f"{artist_slug}:meter:{slugify(pattern[:30])}_{sec_type}"
        results.append(MeterPatternData(
            id=pattern_id,
            pattern=pattern,
            artist_id=artist_slug,
            frequency=freq,
            description=f"{sec_type} with syllable pattern {pattern}",
        ))

    return results


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------

def analyze_artist(artist_slug: str) -> dict:
    """Run full structural decomposition and linguistic analysis for an artist.

    Reads from data/raw/{artist_slug}.json (must already exist from scraper).
    Saves structured analysis to data/processed/{artist_slug}_graph_data.json.

    Returns:
        Dict containing songs, phrases, cultural_references, meter_patterns.
    """
    raw_path = RAW_DIR / f"{artist_slug}.json"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"No raw data at {raw_path}. Run the scraper first."
        )

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_songs = json.load(f)

    print(f"Decomposing {len(raw_songs)} songs for {artist_slug}...")

    # Stage 1: Structural decomposition
    songs = []
    for raw_song in raw_songs:
        song = decompose_song(raw_song, artist_slug)
        if song.section_count > 0:
            songs.append(song)

    print(f"  Decomposed into {sum(s.section_count for s in songs)} sections, "
          f"{sum(s.line_count for s in songs)} lines")

    # Stage 2: Phrase extraction
    phrases = extract_phrases(songs, artist_slug)
    print(f"  Extracted {len(phrases)} recurring phrases")

    # Stage 3: Cultural reference detection
    cultural_refs = extract_cultural_references(songs, artist_slug)
    print(f"  Found {len(cultural_refs)} cultural references")

    # Stage 4: Meter patterns
    meter_patterns = extract_meter_patterns(songs, artist_slug)
    print(f"  Found {len(meter_patterns)} meter patterns")

    result = {
        "artist_slug": artist_slug,
        "songs": [_song_to_dict(s) for s in songs],
        "phrases": [asdict(p) for p in phrases],
        "cultural_references": [asdict(cr) for cr in cultural_refs],
        "meter_patterns": [asdict(mp) for mp in meter_patterns],
        "stats": {
            "total_songs": len(songs),
            "total_sections": sum(s.section_count for s in songs),
            "total_lines": sum(s.line_count for s in songs),
            "total_words": sum(s.word_count for s in songs),
            "total_phrases": len(phrases),
            "total_cultural_refs": len(cultural_refs),
            "total_meter_patterns": len(meter_patterns),
        },
    }

    # Save to disk
    ensure_dirs()
    output_path = PROCESSED_DIR / f"{artist_slug}_graph_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved graph analysis data to {output_path}")

    return result


def _song_to_dict(song: SongData) -> dict:
    """Convert SongData to a serializable dict."""
    d = asdict(song)
    return d
