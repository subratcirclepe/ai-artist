"""Chat interface component."""

import streamlit as st


def render_chat_history():
    """Render the chat message history."""
    for msg in st.session_state.get("messages", []):
        role = msg["role"]
        content = msg["content"]
        references = msg.get("references", [])

        with st.chat_message(role):
            st.markdown(content)

            # Show references for assistant messages
            if role == "assistant" and references:
                render_references(references)


def render_references(references: list[dict]):
    """Render the reference songs panel under an AI response."""
    if not references:
        return

    # Deduplicate by song title
    seen_titles = set()
    unique_refs = []
    for ref in references:
        title = ref.get("metadata", {}).get("song_title", "Unknown")
        if title not in seen_titles:
            seen_titles.add(title)
            unique_refs.append(ref)

    with st.expander(f"📚 Reference songs used ({len(unique_refs)})"):
        for ref in unique_refs:
            meta = ref.get("metadata", {})
            title = meta.get("song_title", "Unknown")
            album = meta.get("album", "")
            mood = meta.get("estimated_mood", "")
            text_preview = ref.get("text", "")[:200]

            st.markdown(f"**{title}**")
            if album:
                st.caption(f"Album: {album}")
            if mood:
                st.caption(f"Mood: {mood}")
            st.text(text_preview + "...")
            st.divider()


def add_message(role: str, content: str, references: list | None = None):
    """Add a message to the session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    msg = {"role": role, "content": content}
    if references:
        msg["references"] = references
    st.session_state.messages.append(msg)
