"""Tests for the lyrics scraper."""

import pytest
from src.scraper import clean_lyrics


class TestCleanLyrics:
    """Test lyrics cleaning logic."""

    def test_removes_embed_text(self):
        raw = "Some lyrics here\n123Embed"
        assert "Embed" not in clean_lyrics(raw)

    def test_removes_you_might_also_like(self):
        raw = "Line one\nYou might also like\nLine two"
        cleaned = clean_lyrics(raw)
        assert "You might also like" not in cleaned

    def test_removes_contributor_info(self):
        raw = "5 Contributors\nActual lyrics here"
        cleaned = clean_lyrics(raw)
        assert "Contributors" not in cleaned

    def test_removes_section_headers(self):
        raw = "[Verse 1]\nSome lyrics\n[Chorus]\nMore lyrics"
        cleaned = clean_lyrics(raw)
        assert "[Verse 1]" not in cleaned
        assert "[Chorus]" not in cleaned
        assert "Some lyrics" in cleaned
        assert "More lyrics" in cleaned

    def test_collapses_blank_lines(self):
        raw = "Line 1\n\n\n\n\nLine 2"
        cleaned = clean_lyrics(raw)
        assert "\n\n\n" not in cleaned

    def test_empty_input(self):
        assert clean_lyrics("") == ""
        assert clean_lyrics(None) == ""

    def test_preserves_actual_lyrics(self):
        raw = "Baarish ke baad woh smell aati hai na\nMitti ki yaad ki"
        cleaned = clean_lyrics(raw)
        assert "Baarish ke baad" in cleaned
        assert "Mitti ki yaad ki" in cleaned
