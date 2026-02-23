"""Genius API lyrics scraper for AI Artist Agent."""

import argparse
import json
import re
import time
from pathlib import Path

import lyricsgenius

from src.utils import (
    get_api_key,
    load_artist_config,
    slugify,
    ensure_dirs,
    RAW_DIR,
)


def clean_lyrics(lyrics: str) -> str:
    """Clean raw Genius lyrics text."""
    if not lyrics:
        return ""

    # Remove "Embed" text at the end
    lyrics = re.sub(r"\d*Embed$", "", lyrics)
    # Remove "You might also like" text
    lyrics = re.sub(r"You might also like", "", lyrics)
    # Remove contributor info
    lyrics = re.sub(r"\d+ Contributors?.*?\n", "", lyrics)
    # Remove "See .* Live" promotions
    lyrics = re.sub(r"See .* LiveGet tickets.*?\n", "", lyrics, flags=re.IGNORECASE)
    # Remove section headers like [Verse 1], [Chorus], etc.
    lyrics = re.sub(r"\[.*?\]", "", lyrics)
    # Remove extra blank lines
    lyrics = re.sub(r"\n{3,}", "\n\n", lyrics)
    # Strip leading/trailing whitespace
    lyrics = lyrics.strip()

    return lyrics


def scrape_artist(
    artist_name: str,
    max_songs: int = 50,
    genius_token: str | None = None,
) -> list[dict]:
    """
    Scrape lyrics for an artist from Genius.

    Args:
        artist_name: The artist's name as it appears on Genius.
        max_songs: Maximum number of songs to scrape.
        genius_token: Genius API token. If None, reads from env.

    Returns:
        List of song dictionaries.
    """
    if genius_token is None:
        genius_token = get_api_key("GENIUS_API_TOKEN")

    genius = lyricsgenius.Genius(
        genius_token,
        timeout=15,
        retries=3,
        remove_section_headers=False,  # We clean them ourselves
    )
    genius.verbose = False
    genius.excluded_terms = [
        "(Remix)",
        "(Live)",
        "(Instrumental)",
        "(Karaoke)",
    ]

    print(f"Searching for {artist_name} on Genius...")
    artist = genius.search_artist(
        artist_name,
        max_songs=max_songs,
        sort="popularity",
    )

    if artist is None:
        print(f"Could not find artist: {artist_name}")
        return []

    songs = []
    total = len(artist.songs)
    for i, song in enumerate(artist.songs, 1):
        print(f"Processing song {i}/{total}: {song.title}...")

        raw_lyrics = song.lyrics or ""
        cleaned = clean_lyrics(raw_lyrics)

        if not cleaned or len(cleaned) < 50:
            print(f"  Skipping (no/short lyrics): {song.title}")
            continue

        songs.append({
            "title": song.title,
            "album": getattr(song, "album", None),
            "year": getattr(song, "year", None),
            "lyrics": cleaned,
            "url": song.url,
        })

        # Rate limit: be gentle with the API
        time.sleep(0.5)

    print(f"Successfully scraped {len(songs)} songs for {artist_name}.")
    return songs


def save_raw_data(artist_slug: str, songs: list[dict]) -> Path:
    """Save scraped songs as JSON."""
    ensure_dirs()
    output_path = RAW_DIR / f"{artist_slug}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(songs, f, ensure_ascii=False, indent=2)
    print(f"Saved raw data to {output_path}")
    return output_path


def run_scraper(artist_name: str, max_songs: int = 50) -> Path:
    """Full scraping pipeline: scrape and save."""
    artist_slug = slugify(artist_name)
    songs = scrape_artist(artist_name, max_songs=max_songs)
    if not songs:
        raise RuntimeError(f"No songs scraped for {artist_name}")
    return save_raw_data(artist_slug, songs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape lyrics from Genius")
    parser.add_argument("--artist", required=True, help="Artist name")
    parser.add_argument(
        "--max-songs", type=int, default=50, help="Max songs to scrape"
    )
    args = parser.parse_args()
    run_scraper(args.artist, args.max_songs)
