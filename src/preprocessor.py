"""Data preprocessor for lyrics chunks."""

import argparse
import json
import re
from pathlib import Path

from src.utils import (
    slugify,
    ensure_dirs,
    RAW_DIR,
    PROCESSED_DIR,
)

# Mood keywords for simple mood detection
MOOD_KEYWORDS = {
    "nostalgic": [
        "yaad", "remember", "purana", "purani", "bachpan", "childhood",
        "woh din", "those days", "memory", "memories", "yaadein",
    ],
    "romantic": [
        "pyaar", "love", "ishq", "dil", "heart", "tere", "tumse",
        "saath", "together", "humsafar", "mohabbat",
    ],
    "melancholic": [
        "dard", "pain", "rona", "cry", "tears", "aansu", "tanha",
        "alone", "lonely", "akela", "judai", "separation", "sad",
    ],
    "hopeful": [
        "umeed", "hope", "naya", "new", "sapna", "dream", "light",
        "roshni", "tomorrow", "kal", "sunrise",
    ],
    "peaceful": [
        "sukoon", "peace", "calm", "shanti", "quiet", "silence",
        "baarish", "rain", "breeze", "hawa",
    ],
}


def detect_language(text: str) -> str:
    """Detect if text is Hindi, English, or Hinglish using simple heuristics."""
    # Check for Devanagari characters
    devanagari_count = len(re.findall(r"[\u0900-\u097F]", text))
    # Check for Latin characters (words)
    latin_words = len(re.findall(r"[a-zA-Z]+", text))

    total = devanagari_count + latin_words
    if total == 0:
        return "unknown"

    hindi_ratio = devanagari_count / total

    if hindi_ratio > 0.7:
        return "hindi"
    elif hindi_ratio < 0.1:
        return "english"
    else:
        return "hinglish"


def estimate_mood(text: str) -> str:
    """Estimate the mood of lyrics using keyword matching."""
    text_lower = text.lower()
    scores = {}
    for mood, keywords in MOOD_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[mood] = score

    if max(scores.values()) == 0:
        return "neutral"
    return max(scores, key=scores.get)


def chunk_lyrics(
    lyrics: str,
    chunk_size: int = 200,
    overlap: int = 50,
) -> list[str]:
    """
    Split lyrics into overlapping word-based chunks.

    Args:
        lyrics: Full lyrics text.
        chunk_size: Target words per chunk.
        overlap: Overlap words between chunks.

    Returns:
        List of text chunks.
    """
    words = lyrics.split()
    if len(words) <= chunk_size:
        return [lyrics]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def clean_processed_text(text: str) -> str:
    """Additional cleaning for processed text."""
    # Remove any remaining Genius artifacts
    text = re.sub(r"EmbedShare.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\d+ Contributors?", "", text)
    text = re.sub(r"You might also like", "", text)
    # Collapse multiple spaces
    text = re.sub(r" +", " ", text)
    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def preprocess_artist(artist_slug: str) -> list[dict]:
    """
    Preprocess raw lyrics data into chunks for embedding.

    Args:
        artist_slug: The artist's slug identifier.

    Returns:
        List of processed chunk dictionaries.
    """
    raw_path = RAW_DIR / f"{artist_slug}.json"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"No raw data found at {raw_path}. Run the scraper first."
        )

    with open(raw_path, "r", encoding="utf-8") as f:
        songs = json.load(f)

    all_chunks = []
    for song in songs:
        lyrics = clean_processed_text(song["lyrics"])
        if not lyrics:
            continue

        language = detect_language(lyrics)
        mood = estimate_mood(lyrics)
        chunks = chunk_lyrics(lyrics)

        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{artist_slug}_{slugify(song['title'])}_chunk_{idx}"
            all_chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "song_title": song["title"],
                    "artist": artist_slug,
                    "album": song.get("album"),
                    "year": song.get("year"),
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "language": language,
                    "estimated_mood": mood,
                },
            })

    # Print stats
    unique_songs = len(set(c["metadata"]["song_title"] for c in all_chunks))
    languages = {}
    for c in all_chunks:
        lang = c["metadata"]["language"]
        languages[lang] = languages.get(lang, 0) + 1

    print(f"Preprocessing complete for {artist_slug}:")
    print(f"  Total songs: {unique_songs}")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Language distribution: {languages}")

    return all_chunks


def save_processed_data(artist_slug: str, chunks: list[dict]) -> Path:
    """Save processed chunks as JSON."""
    ensure_dirs()
    output_path = PROCESSED_DIR / f"{artist_slug}_processed.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved processed data to {output_path}")
    return output_path


def run_preprocessor(artist_slug: str) -> Path:
    """Full preprocessing pipeline."""
    chunks = preprocess_artist(artist_slug)
    if not chunks:
        raise RuntimeError(f"No chunks produced for {artist_slug}")
    return save_processed_data(artist_slug, chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess scraped lyrics")
    parser.add_argument("--artist", required=True, help="Artist slug")
    args = parser.parse_args()
    run_preprocessor(args.artist)
