"""Post-generation validation layer.

Runs 5 checks on generated output:
1. Vocabulary authenticity
2. Originality (anti-plagiarism)
3. Rhyme scheme compliance
4. Emotional arc consistency
5. Structural compliance

Produces a ValidationReport and triggers re-generation if needed.
"""

import re
from collections import Counter
from dataclasses import dataclass, field

from src.preprocessor import estimate_mood
from src.analysis.phonetics import classify_rhyme, detect_section_rhyme_scheme


@dataclass
class FlaggedLine:
    line_index: int
    line_text: str
    reason: str
    score: float


@dataclass
class ValidationReport:
    overall_pass: bool
    overall_score: float  # 0.0 to 1.0
    vocabulary_score: float
    originality_score: float
    rhyme_score: float
    emotional_arc_score: float
    structure_score: float
    flagged_lines: list[FlaggedLine] = field(default_factory=list)
    recommendation: str = "accept"  # "accept" | "regenerate_partial" | "regenerate_full"
    details: dict = field(default_factory=dict)


def validate_output(
    output_text: str,
    artist_slug: str,
    vocabulary_set: list[str] | None = None,
    anti_vocabulary: list[str] | None = None,
    expected_structure: str | None = None,
    expected_mood_arc: list[str] | None = None,
    existing_lines: list[str] | None = None,
) -> ValidationReport:
    """Run all 5 validation checks on generated output.

    Args:
        output_text: The generated song lyrics.
        artist_slug: Artist identifier.
        vocabulary_set: Top words the artist uses (for vocabulary check).
        anti_vocabulary: Words the artist never uses.
        expected_structure: Expected section pattern e.g. "verse-chorus-verse-chorus".
        expected_mood_arc: Expected mood sequence per section.
        existing_lines: All existing lines from the artist's catalog (for originality).

    Returns:
        ValidationReport with scores and recommendations.
    """
    # Parse output into sections and lines
    sections = _parse_output_sections(output_text)
    all_lines = [ln for sec in sections for ln in sec["lines"]]

    # Run checks
    vocab_score = _check_vocabulary(all_lines, vocabulary_set or [], anti_vocabulary or [])
    orig_score, orig_flags = _check_originality(all_lines, existing_lines or [])
    rhyme_score = _check_rhyme_compliance(sections)
    arc_score = _check_emotional_arc(sections, expected_mood_arc or [])
    struct_score = _check_structural_compliance(sections, expected_structure or "")

    # Aggregate
    weights = {
        "vocabulary": 0.25,
        "originality": 0.30,
        "rhyme": 0.15,
        "emotional_arc": 0.15,
        "structure": 0.15,
    }
    overall = (
        vocab_score * weights["vocabulary"]
        + orig_score * weights["originality"]
        + rhyme_score * weights["rhyme"]
        + arc_score * weights["emotional_arc"]
        + struct_score * weights["structure"]
    )

    # Determine recommendation
    if overall >= 0.8 and not orig_flags:
        recommendation = "accept"
    elif overall >= 0.6 and len(orig_flags) <= 2:
        recommendation = "regenerate_partial"
    else:
        recommendation = "regenerate_full"

    return ValidationReport(
        overall_pass=recommendation == "accept",
        overall_score=overall,
        vocabulary_score=vocab_score,
        originality_score=orig_score,
        rhyme_score=rhyme_score,
        emotional_arc_score=arc_score,
        structure_score=struct_score,
        flagged_lines=orig_flags,
        recommendation=recommendation,
        details={
            "total_lines": len(all_lines),
            "total_sections": len(sections),
            "vocabulary_overlap": vocab_score,
            "anti_vocab_violations": _count_anti_vocab(all_lines, anti_vocabulary or []),
        },
    )


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _parse_output_sections(text: str) -> list[dict]:
    """Parse generated output into sections with section types and lines."""
    sections = []
    current_section = {"type": "intro", "lines": [], "text": ""}

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        # Check for section header
        header_match = re.match(r"^\[([^\]]+)\]$", line)
        if header_match:
            # Save current section if non-empty
            if current_section["lines"]:
                current_section["text"] = "\n".join(current_section["lines"])
                sections.append(current_section)
            # Start new section
            header = header_match.group(1).lower()
            sec_type = "verse"
            if "chorus" in header or "hook" in header:
                sec_type = "chorus"
            elif "bridge" in header:
                sec_type = "bridge"
            elif "outro" in header:
                sec_type = "outro"
            elif "intro" in header:
                sec_type = "intro"
            elif "pre" in header:
                sec_type = "pre_chorus"
            current_section = {"type": sec_type, "lines": [], "text": ""}
        elif not line.startswith("["):
            # Strip delivery directions for analysis but keep them
            clean_line = re.sub(r"\[.*?\]", "", line).strip()
            if clean_line:
                current_section["lines"].append(clean_line)

    # Save last section
    if current_section["lines"]:
        current_section["text"] = "\n".join(current_section["lines"])
        sections.append(current_section)

    return sections


# ---------------------------------------------------------------------------
# Check 1: Vocabulary authenticity
# ---------------------------------------------------------------------------

def _check_vocabulary(
    lines: list[str],
    vocabulary_set: list[str],
    anti_vocabulary: list[str],
) -> float:
    """Check if output vocabulary matches artist's fingerprint."""
    if not vocabulary_set:
        return 0.8  # No vocabulary data, assume reasonable

    all_words = []
    for line in lines:
        words = re.findall(r"[\u0900-\u097F]+|[a-zA-Z]+", line.lower())
        all_words.extend(words)

    if not all_words:
        return 0.0

    vocab_set = set(w.lower() for w in vocabulary_set)
    anti_set = set(w.lower() for w in anti_vocabulary)

    in_vocab = sum(1 for w in all_words if w in vocab_set)
    in_anti = sum(1 for w in all_words if w in anti_set)

    vocab_score = in_vocab / len(all_words)
    penalty = (in_anti / len(all_words)) * 2  # Double penalty for anti-vocabulary

    return max(0.0, min(1.0, vocab_score - penalty))


def _count_anti_vocab(lines: list[str], anti_vocabulary: list[str]) -> int:
    """Count anti-vocabulary violations."""
    anti_set = set(w.lower() for w in anti_vocabulary)
    count = 0
    for line in lines:
        words = re.findall(r"[\u0900-\u097F]+|[a-zA-Z]+", line.lower())
        count += sum(1 for w in words if w in anti_set)
    return count


# ---------------------------------------------------------------------------
# Check 2: Originality (anti-plagiarism)
# ---------------------------------------------------------------------------

def _check_originality(
    lines: list[str],
    existing_lines: list[str],
) -> tuple[float, list[FlaggedLine]]:
    """Check that no lines are copied from the artist's real catalog."""
    if not existing_lines:
        return 1.0, []  # No catalog to compare against

    existing_set = set(ln.strip().lower() for ln in existing_lines if ln.strip())
    existing_ngrams = {}
    for ln in existing_set:
        ngrams = _get_ngrams(ln, 4)
        if ngrams:
            existing_ngrams[ln] = ngrams

    flagged = []
    for i, line in enumerate(lines):
        line_lower = line.strip().lower()
        if not line_lower:
            continue

        # Exact match check
        if line_lower in existing_set:
            flagged.append(FlaggedLine(
                line_index=i,
                line_text=line,
                reason="Exact copy of existing line",
                score=1.0,
            ))
            continue

        # 4-gram overlap check
        line_ngrams = _get_ngrams(line_lower, 4)
        if not line_ngrams:
            continue

        for existing_ln, ex_ngrams in existing_ngrams.items():
            overlap = len(line_ngrams & ex_ngrams) / max(len(line_ngrams), 1)
            if overlap > 0.6:
                flagged.append(FlaggedLine(
                    line_index=i,
                    line_text=line,
                    reason=f"High n-gram overlap ({overlap:.0%}) with: {existing_ln[:80]}",
                    score=overlap,
                ))
                break

    originality = 1.0 - (len(flagged) / max(len(lines), 1))
    return max(0.0, originality), flagged


def _get_ngrams(text: str, n: int) -> set[str]:
    """Get character n-grams from text."""
    text = text.lower().strip()
    if len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


# ---------------------------------------------------------------------------
# Check 3: Rhyme scheme compliance
# ---------------------------------------------------------------------------

def _check_rhyme_compliance(sections: list[dict]) -> float:
    """Check if sections have consistent rhyme patterns."""
    if not sections:
        return 0.5

    rhyming_sections = 0
    total_sections = 0

    for section in sections:
        lines = section["lines"]
        if len(lines) < 2:
            continue
        total_sections += 1

        # Check adjacent lines for rhymes
        end_words = []
        for line in lines:
            words = re.findall(r"[\u0900-\u097F]+|[a-zA-Z]+", line)
            if words:
                end_words.append(words[-1])

        if len(end_words) < 2:
            continue

        # Count rhyming pairs
        rhyme_count = 0
        total_pairs = 0
        for i in range(0, len(end_words) - 1):
            total_pairs += 1
            if classify_rhyme(end_words[i], end_words[i+1]):
                rhyme_count += 1
            # Also check ABAB
            if i + 2 < len(end_words):
                total_pairs += 1
                if classify_rhyme(end_words[i], end_words[i+2]):
                    rhyme_count += 1

        if total_pairs > 0 and rhyme_count / total_pairs > 0.2:
            rhyming_sections += 1

    if total_sections == 0:
        return 0.5

    return rhyming_sections / total_sections


# ---------------------------------------------------------------------------
# Check 4: Emotional arc consistency
# ---------------------------------------------------------------------------

def _check_emotional_arc(
    sections: list[dict],
    expected_arc: list[str],
) -> float:
    """Check if mood progression matches expected emotional arc."""
    if not expected_arc or not sections:
        return 0.7  # No arc specified, assume reasonable

    detected_moods = []
    for section in sections:
        text = section.get("text", "")
        mood = estimate_mood(text)
        detected_moods.append(mood)

    if not detected_moods:
        return 0.5

    # Compare detected vs expected (allow one-step difference)
    matches = 0
    comparisons = min(len(detected_moods), len(expected_arc))
    for i in range(comparisons):
        if detected_moods[i] == expected_arc[i]:
            matches += 1
        elif _mood_distance(detected_moods[i], expected_arc[i]) <= 1:
            matches += 0.5

    return matches / max(comparisons, 1)


def _mood_distance(mood_a: str, mood_b: str) -> int:
    """Simple mood distance: 0 if same, 1 if adjacent, 2 otherwise."""
    if mood_a == mood_b:
        return 0

    # Adjacent mood groups
    adjacent = {
        "nostalgic": {"melancholic", "peaceful", "romantic"},
        "romantic": {"nostalgic", "hopeful", "peaceful"},
        "melancholic": {"nostalgic", "peaceful"},
        "hopeful": {"romantic", "peaceful", "energetic"},
        "peaceful": {"nostalgic", "hopeful", "melancholic"},
        "neutral": {"nostalgic", "peaceful", "hopeful"},
        "energetic": {"hopeful", "romantic"},
    }

    if mood_b in adjacent.get(mood_a, set()):
        return 1
    return 2


# ---------------------------------------------------------------------------
# Check 5: Structural compliance
# ---------------------------------------------------------------------------

def _check_structural_compliance(
    sections: list[dict],
    expected_structure: str,
) -> float:
    """Check if output structure matches expected template."""
    if not expected_structure or not sections:
        return 0.7  # No structure specified

    expected_types = expected_structure.split("-")
    actual_types = [sec["type"] for sec in sections]

    if not expected_types:
        return 0.7

    # Compare lengths
    length_score = 1.0 - abs(len(actual_types) - len(expected_types)) / max(len(expected_types), 1)
    length_score = max(0.0, length_score)

    # Compare types (in order)
    matches = 0
    for i in range(min(len(actual_types), len(expected_types))):
        if actual_types[i] == expected_types[i]:
            matches += 1
    type_score = matches / max(len(expected_types), 1)

    # Must have at least one verse and one chorus
    has_verse = any(t in ("verse", "mukhda") for t in actual_types)
    has_chorus = any(t in ("chorus", "hook") for t in actual_types)
    has_basics = 1.0 if (has_verse and has_chorus) else 0.5

    return (length_score * 0.3 + type_score * 0.4 + has_basics * 0.3)
