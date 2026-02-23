"""Tests for the artist agent."""

import pytest
from src.agent import ArtistAgent, is_song_request, extract_topic


class TestIntentDetection:
    """Test song request detection."""

    def test_detects_write_song(self):
        assert is_song_request("Write a song about rain")
        assert is_song_request("write song about love")

    def test_detects_sing_about(self):
        assert is_song_request("Sing about monsoon")

    def test_detects_create_lyrics(self):
        assert is_song_request("Create lyrics for heartbreak")
        assert is_song_request("Create a song about summer")

    def test_detects_compose(self):
        assert is_song_request("Compose a song about memories")

    def test_detects_hindi_requests(self):
        assert is_song_request("gana likho baarish ke baare mein")
        assert is_song_request("gaana banao pyaar ke baare mein")

    def test_regular_chat_not_song(self):
        assert not is_song_request("What inspires you?")
        assert not is_song_request("Tell me about your music")
        assert not is_song_request("Hello!")


class TestTopicExtraction:
    """Test topic extraction from messages."""

    def test_extracts_topic_from_write_song(self):
        topic = extract_topic("Write a song about monsoon rain")
        assert "monsoon" in topic.lower() or "rain" in topic.lower()

    def test_extracts_topic_from_sing_about(self):
        topic = extract_topic("Sing about lost love")
        assert "lost love" in topic.lower()

    def test_extracts_topic_from_lyrics_for(self):
        topic = extract_topic("Create lyrics about childhood memories")
        assert "childhood" in topic.lower() or "memories" in topic.lower()


class TestArtistAgent:
    """Test ArtistAgent class."""

    def test_initialization(self):
        agent = ArtistAgent("anuv_jain")
        assert agent.artist_name == "Anuv Jain"
        assert agent.artist_slug == "anuv_jain"
        assert agent.chat_history == []

    def test_greeting(self):
        agent = ArtistAgent("anuv_jain")
        greeting = agent.get_greeting()
        assert "Anuv Jain" in greeting

    def test_switch_artist(self):
        agent = ArtistAgent("anuv_jain")
        agent.chat_history = [("user", "hello"), ("assistant", "hi")]
        agent.switch_artist("arijit_singh")
        assert agent.artist_name == "Arijit Singh"
        assert agent.chat_history == []

    def test_clear_history(self):
        agent = ArtistAgent("anuv_jain")
        agent.chat_history = [("user", "hello")]
        agent.clear_history()
        assert agent.chat_history == []

    def test_history_window(self):
        agent = ArtistAgent("anuv_jain")
        # Add more than max_history exchanges
        for i in range(15):
            agent._add_to_history("user", f"msg {i}")
            agent._add_to_history("assistant", f"reply {i}")
        # Should be trimmed to max_history * 2
        assert len(agent.chat_history) == agent.max_history * 2

    def test_all_artists_init(self):
        for slug in ["anuv_jain", "arijit_singh", "prateek_kuhad"]:
            agent = ArtistAgent(slug)
            assert agent.artist_name is not None
