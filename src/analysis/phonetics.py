"""Hindi/English phonetic analysis for rhyme detection.

Provides rhyme pair detection using suffix matching and phonetic heuristics
for Hindi (Devanagari + romanized) and English text.
"""

import re
from collections import Counter
from dataclasses import dataclass


@dataclass
class RhymePairData:
    id: str
    word_a: str
    word_b: str
    rhyme_type: str  # perfect, slant, assonance, internal, cross_language
    language: str
    frequency: int
    artist_id: str


# ---------------------------------------------------------------------------
# Phonetic suffix extraction
# ---------------------------------------------------------------------------

# Hindi vowel matras (vowel signs) and their approximate sounds
_HINDI_VOWEL_SUFFIXES = {
    "\u093E": "aa",   # ा
    "\u093F": "i",    # ि
    "\u0940": "ee",   # ी
    "\u0941": "u",    # ु
    "\u0942": "oo",   # ू
    "\u0947": "e",    # े
    "\u0948": "ai",   # ै
    "\u094B": "o",    # ो
    "\u094C": "au",   # ौ
}


def _get_hindi_suffix(word: str, length: int = 3) -> str:
    """Extract a phonetic suffix from a Hindi (Devanagari) word.

    Takes the last `length` characters and normalizes vowel matras
    to their approximate Latin phonetic equivalents.
    """
    if not word:
        return ""
    suffix = word[-length:]
    # Replace Devanagari vowel matras with their phonetic equivalent
    result = []
    for ch in suffix:
        if ch in _HINDI_VOWEL_SUFFIXES:
            result.append(_HINDI_VOWEL_SUFFIXES[ch])
        elif "\u0915" <= ch <= "\u0939":
            # Consonant without matra — implicit "a" sound
            result.append("a")
        else:
            result.append(ch)
    return "".join(result)


def _get_english_suffix(word: str, length: int = 3) -> str:
    """Extract suffix from an English word for rhyme comparison."""
    return word.lower()[-length:] if word else ""


def _get_romanized_suffix(word: str, length: int = 3) -> str:
    """Extract suffix from a romanized Hindi word."""
    return word.lower()[-length:] if word else ""


def _is_hindi(word: str) -> bool:
    """Check if a word is Devanagari."""
    return bool(re.search(r"[\u0900-\u097F]", word))


def _is_english(word: str) -> bool:
    """Check if a word is Latin script."""
    return bool(re.match(r"^[a-zA-Z]+$", word))


# ---------------------------------------------------------------------------
# Rhyme comparison
# ---------------------------------------------------------------------------

def get_word_suffix(word: str) -> str:
    """Get the phonetic suffix of a word regardless of script."""
    word = word.strip()
    if _is_hindi(word):
        return _get_hindi_suffix(word)
    elif _is_english(word):
        return _get_english_suffix(word)
    else:
        # Romanized Hindi or mixed
        return _get_romanized_suffix(word)


def classify_rhyme(word_a: str, word_b: str) -> str | None:
    """Classify the rhyme type between two words.

    Returns:
        Rhyme type string or None if no rhyme detected.
    """
    if not word_a or not word_b:
        return None

    word_a = word_a.strip().lower()
    word_b = word_b.strip().lower()

    if word_a == word_b:
        return None  # Same word is not a rhyme

    a_hindi = _is_hindi(word_a)
    b_hindi = _is_hindi(word_b)
    a_english = _is_english(word_a)
    b_english = _is_english(word_b)

    # Cross-language rhyme (Hindi word rhymes with English word)
    if (a_hindi and b_english) or (a_english and b_hindi):
        suffix_a = get_word_suffix(word_a)
        suffix_b = get_word_suffix(word_b)
        if suffix_a and suffix_b and suffix_a[-2:] == suffix_b[-2:]:
            return "cross_language"
        return None

    # Same-language comparison
    suffix_a = get_word_suffix(word_a)
    suffix_b = get_word_suffix(word_b)

    if not suffix_a or not suffix_b:
        return None

    # Perfect rhyme: last 3 sounds match
    if len(suffix_a) >= 3 and len(suffix_b) >= 3 and suffix_a[-3:] == suffix_b[-3:]:
        return "perfect"

    # Slant rhyme: last 2 sounds match
    if len(suffix_a) >= 2 and len(suffix_b) >= 2 and suffix_a[-2:] == suffix_b[-2:]:
        return "slant"

    # Assonance: last vowel sound matches
    vowels_a = re.findall(r"[aeiou]+", suffix_a)
    vowels_b = re.findall(r"[aeiou]+", suffix_b)
    if vowels_a and vowels_b and vowels_a[-1] == vowels_b[-1]:
        return "assonance"

    return None


# ---------------------------------------------------------------------------
# Rhyme pair extraction from songs
# ---------------------------------------------------------------------------

def extract_rhyme_pairs(
    songs: list[dict],
    artist_slug: str,
) -> list[RhymePairData]:
    """Extract rhyme pairs from an artist's catalog by analyzing line endings.

    Checks adjacent lines and alternating lines within each section
    for rhyme patterns (AABB, ABAB, ABBA).

    Args:
        songs: List of SongData dicts (from lyric_analyzer output).
        artist_slug: Artist identifier.

    Returns:
        List of RhymePairData with deduplicated rhyme pairs.
    """
    pair_counter: Counter = Counter()
    pair_info: dict[tuple, str] = {}  # (word_a, word_b) -> rhyme_type

    for song in songs:
        for section in song.get("sections", []):
            lines = section.get("lines", [])
            if len(lines) < 2:
                continue

            end_words = [ln.get("end_word", "").strip() for ln in lines]

            # Check adjacent pairs (AABB pattern): lines 0-1, 2-3, etc.
            for i in range(0, len(end_words) - 1, 1):
                w_a = end_words[i]
                w_b = end_words[i + 1]
                rhyme_type = classify_rhyme(w_a, w_b)
                if rhyme_type:
                    key = tuple(sorted([w_a.lower(), w_b.lower()]))
                    pair_counter[key] += 1
                    pair_info[key] = rhyme_type

            # Check alternating pairs (ABAB pattern): lines 0-2, 1-3, etc.
            for i in range(0, len(end_words) - 2):
                w_a = end_words[i]
                w_b = end_words[i + 2]
                rhyme_type = classify_rhyme(w_a, w_b)
                if rhyme_type:
                    key = tuple(sorted([w_a.lower(), w_b.lower()]))
                    pair_counter[key] += 1
                    pair_info[key] = rhyme_type

    # Build RhymePairData list
    results = []
    for (w_a, w_b), freq in pair_counter.most_common():
        if freq < 1:
            break
        rhyme_type = pair_info.get((w_a, w_b), "slant")
        pair_id = f"{artist_slug}:rhyme:{w_a}_{w_b}"

        # Determine language
        if _is_hindi(w_a) or _is_hindi(w_b):
            lang = "hindi"
        elif _is_english(w_a) and _is_english(w_b):
            lang = "english"
        else:
            lang = "hinglish"

        results.append(RhymePairData(
            id=pair_id,
            word_a=w_a,
            word_b=w_b,
            rhyme_type=rhyme_type,
            language=lang,
            frequency=freq,
            artist_id=artist_slug,
        ))

    return results


def detect_section_rhyme_scheme(lines: list[dict]) -> str:
    """Detect the rhyme scheme of a section (e.g., AABB, ABAB, ABBA, FREE).

    Args:
        lines: List of line dicts with 'end_word' key.

    Returns:
        Rhyme scheme string like "AABB", "ABAB", or "FREE".
    """
    if len(lines) < 2:
        return "FREE"

    end_words = [ln.get("end_word", "").strip().lower() for ln in lines]
    n = len(end_words)

    # Assign letters based on rhyme matching
    scheme = []
    letter_map: dict[str, str] = {}
    next_letter = ord("A")

    for i, word in enumerate(end_words):
        assigned = False
        for j in range(i):
            rhyme = classify_rhyme(word, end_words[j])
            if rhyme:
                scheme.append(scheme[j])
                assigned = True
                break
        if not assigned:
            letter = chr(next_letter)
            scheme.append(letter)
            next_letter += 1

    pattern = "".join(scheme)

    # Classify common patterns
    if n >= 4:
        first_four = pattern[:4]
        if first_four in ("AABB", "ABAB", "ABBA", "ABCB"):
            return first_four

    return pattern if len(set(pattern)) < len(pattern) else "FREE"
