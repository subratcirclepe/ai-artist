"""Re-generation strategy for failed validation.

When the validation layer rejects generated output, this module
handles partial or full re-generation with strengthened constraints.
"""

from src.validation.validator import ValidationReport


def build_repair_prompt(
    original_output: str,
    report: ValidationReport,
    artist_name: str,
) -> str:
    """Build a targeted repair prompt for partial re-generation.

    Used when only 1-2 lines are flagged and the rest is good.
    """
    flagged_info = []
    for fl in report.flagged_lines:
        flagged_info.append(f"- Line {fl.line_index + 1}: \"{fl.line_text}\" — {fl.reason}")

    flagged_text = "\n".join(flagged_info)

    return (
        f"The following song was generated but has some issues that need fixing:\n\n"
        f"{original_output}\n\n"
        f"## ISSUES TO FIX\n"
        f"{flagged_text}\n\n"
        f"## INSTRUCTIONS\n"
        f"Rewrite the ENTIRE song, keeping the good parts but fixing the flagged lines.\n"
        f"- Replace any lines that are too similar to existing lyrics with completely original ones\n"
        f"- Maintain the same structure, mood progression, and style\n"
        f"- Keep everything in {artist_name}'s authentic voice\n"
        f"- Do NOT change lines that were not flagged unless necessary for flow"
    )


def build_enhanced_regeneration_prompt(
    original_output: str,
    report: ValidationReport,
    artist_name: str,
    topic: str,
) -> str:
    """Build a strengthened prompt for full re-generation.

    Used when the overall score is too low or multiple issues exist.
    """
    issues = []
    if report.vocabulary_score < 0.7:
        issues.append("vocabulary doesn't match the artist's register")
    if report.originality_score < 0.8:
        issues.append("some lines are too similar to existing lyrics")
    if report.rhyme_score < 0.5:
        issues.append("rhyme scheme is inconsistent or missing")
    if report.emotional_arc_score < 0.6:
        issues.append("emotional progression doesn't feel natural")
    if report.structure_score < 0.6:
        issues.append("song structure is wrong")

    issues_text = "; ".join(issues) if issues else "general quality is too low"

    return (
        f"Write a completely NEW original song about \"{topic}\" in the voice of {artist_name}.\n\n"
        f"IMPORTANT: A previous attempt was rejected because: {issues_text}.\n\n"
        f"This time, pay EXTRA attention to:\n"
        f"- Use ONLY words and expressions that {artist_name} would actually use\n"
        f"- Create completely original lines — do NOT recycle any phrases from known songs\n"
        f"- Follow a clear rhyme pattern within each section\n"
        f"- Build a natural emotional arc across the song\n"
        f"- Include proper section labels [Verse 1], [Chorus], [Bridge], etc.\n"
        f"- Include [emotional/delivery directions] in brackets\n"
        f"- If Hindi/Urdu: provide both Devanagari and romanized transliteration\n\n"
        f"Make this feel like an authentic unreleased {artist_name} track."
    )


def select_best_attempt(attempts: list[tuple[str, ValidationReport]]) -> tuple[str, ValidationReport]:
    """Select the best attempt from multiple generation attempts.

    Args:
        attempts: List of (output_text, validation_report) tuples.

    Returns:
        The (output, report) with the highest overall score.
    """
    if not attempts:
        return "", ValidationReport(
            overall_pass=False, overall_score=0.0,
            vocabulary_score=0, originality_score=0,
            rhyme_score=0, emotional_arc_score=0,
            structure_score=0, recommendation="regenerate_full",
        )

    return max(attempts, key=lambda x: x[1].overall_score)
