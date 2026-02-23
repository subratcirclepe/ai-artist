"""Tests for the RAG pipeline."""

import pytest
from src.rag_chain import format_context, build_system_prompt
from src.utils import load_artist_config


class TestFormatContext:
    """Test context formatting for prompts."""

    def test_empty_references(self):
        result = format_context([])
        assert "No reference lyrics available" in result

    def test_formats_references(self):
        refs = [
            {
                "text": "Some lyrics text here",
                "metadata": {"song_title": "Test Song"},
                "score": 0.5,
            }
        ]
        result = format_context(refs)
        assert "Reference Song 1: Test Song" in result
        assert "Some lyrics text here" in result

    def test_multiple_references(self):
        refs = [
            {
                "text": "Lyrics one",
                "metadata": {"song_title": "Song A"},
                "score": 0.3,
            },
            {
                "text": "Lyrics two",
                "metadata": {"song_title": "Song B"},
                "score": 0.5,
            },
        ]
        result = format_context(refs)
        assert "Reference Song 1: Song A" in result
        assert "Reference Song 2: Song B" in result


class TestBuildSystemPrompt:
    """Test system prompt assembly."""

    def test_includes_artist_name(self):
        prompt = build_system_prompt("anuv_jain", "No context")
        assert "Anuv Jain" in prompt

    def test_includes_context(self):
        context = "Reference Song 1: Baarishein\nSome lyrics"
        prompt = build_system_prompt("anuv_jain", context)
        assert "Baarishein" in prompt

    def test_includes_style_rules(self):
        prompt = build_system_prompt("anuv_jain", "No context")
        assert "acoustic" in prompt.lower() or "folk" in prompt.lower()

    def test_all_artists_build_successfully(self):
        for slug in ["anuv_jain", "arijit_singh", "prateek_kuhad"]:
            prompt = build_system_prompt(slug, "test context")
            assert len(prompt) > 100
