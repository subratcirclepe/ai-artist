"""Artist persona agent with memory and intent detection."""

import re

from src.utils import load_artist_config, vectorstore_exists
from src.graph.connection import graph_exists


# Patterns that indicate a song generation request
SONG_PATTERNS = [
    r"write\s+(a\s+)?song",
    r"sing\s+about",
    r"create\s+(a\s+)?(song|lyrics)",
    r"compose\s+(a\s+)?(song|lyrics)",
    r"make\s+(a\s+)?song",
    r"lyrics\s+(for|about|on)",
    r"gana\s+(likho|banao|sunao)",
    r"gaana\s+(likho|banao|sunao)",
]


def is_song_request(message: str) -> bool:
    """Detect if the user is asking for song generation."""
    message_lower = message.lower()
    return any(re.search(p, message_lower) for p in SONG_PATTERNS)


def extract_topic(message: str) -> str:
    """Extract the song topic from the user's message."""
    message_lower = message.lower()

    # Try to extract topic after common phrases
    topic_patterns = [
        r"(?:write|sing|create|compose|make)\s+(?:a\s+)?(?:song|lyrics)?\s*(?:about|on|for)\s+(.+)",
        r"(?:lyrics)\s+(?:about|on|for)\s+(.+)",
        r"(?:gana|gaana)\s+(?:likho|banao|sunao)\s+(.+)",
    ]

    for pattern in topic_patterns:
        match = re.search(pattern, message_lower)
        if match:
            return match.group(1).strip().rstrip(".")

    # Fallback: use the full message minus the command words
    cleaned = re.sub(
        r"(write|sing|create|compose|make|lyrics|song|a|please|can you|could you)",
        "",
        message_lower,
    ).strip()
    return cleaned if cleaned else message


class ArtistAgent:
    """Stateful artist persona agent with conversation memory."""

    def __init__(self, artist_slug: str):
        self.artist_slug = artist_slug
        self.artist_config = load_artist_config(artist_slug)
        self.chat_history: list[tuple[str, str]] = []
        self.max_history = 10  # Keep last 10 exchanges

    @property
    def artist_name(self) -> str:
        return self.artist_config["name"]

    def chat(
        self,
        user_message: str,
        k: int = 5,
        temperature: float | None = None,
    ) -> dict:
        """
        Main entry point for user interactions.

        Detects intent and routes to song generation or chat.

        Args:
            user_message: The user's message.
            k: Number of reference chunks to retrieve.
            temperature: Override temperature (uses defaults if None).

        Returns:
            Dict with response content and metadata.
        """
        # Use graph-powered pipeline if available, fallback to flat RAG
        use_graph = graph_exists(self.artist_slug)

        if is_song_request(user_message):
            topic = extract_topic(user_message)
            temp = temperature if temperature is not None else 0.85

            if use_graph:
                from src.graph_rag_chain import generate_song_with_graph
                result = generate_song_with_graph(
                    self.artist_slug, topic, k=k, temperature=temp,
                )
            else:
                from src.rag_chain import generate_song
                result = generate_song(
                    self.artist_slug, topic, k=k, temperature=temp,
                )

            # Store in history
            self._add_to_history("user", user_message)
            self._add_to_history("assistant", result["song"])

            response_dict = {
                "type": "song",
                "response": result["song"],
                "references": result["references"],
                "topic": result["topic"],
                "artist": result["artist"],
            }
            # Include validation info if graph-powered
            if result.get("validation"):
                response_dict["validation"] = result["validation"]
            if result.get("graph_powered"):
                response_dict["graph_powered"] = True
            return response_dict
        else:
            temp = temperature if temperature is not None else 0.7

            if use_graph:
                from src.graph_rag_chain import chat_with_artist_graph
                result = chat_with_artist_graph(
                    self.artist_slug, user_message,
                    chat_history=self.chat_history, k=3, temperature=temp,
                )
            else:
                from src.rag_chain import chat_with_artist
                result = chat_with_artist(
                    self.artist_slug, user_message,
                    chat_history=self.chat_history, k=3, temperature=temp,
                )

            # Store in history
            self._add_to_history("user", user_message)
            self._add_to_history("assistant", result["response"])

            response_dict = {
                "type": "chat",
                "response": result["response"],
                "references": result["references"],
                "artist": self.artist_name,
            }
            if result.get("graph_powered"):
                response_dict["graph_powered"] = True
            return response_dict

    def _add_to_history(self, role: str, content: str):
        """Add a message to chat history, maintaining window size."""
        self.chat_history.append((role, content))
        # Keep only last N exchanges (2 messages per exchange)
        max_messages = self.max_history * 2
        if len(self.chat_history) > max_messages:
            self.chat_history = self.chat_history[-max_messages:]

    def switch_artist(self, new_artist_slug: str):
        """Switch to a different artist, clearing conversation history."""
        self.artist_slug = new_artist_slug
        self.artist_config = load_artist_config(new_artist_slug)
        self.chat_history = []

    def get_history(self) -> list[tuple[str, str]]:
        """Return the conversation history."""
        return list(self.chat_history)

    def clear_history(self):
        """Reset the conversation history."""
        self.chat_history = []

    def get_greeting(self) -> str:
        """Generate an initial greeting from the artist persona."""
        return (
            f"Hey! I'm {self.artist_name}'s AI creative twin. "
            f"Ask me to write a song about anything, or just chat about music!"
        )
