"""Reference panel component for showing retrieved lyrics context."""

import streamlit as st


def render_reference_panel(references: list[dict]):
    """
    Render a standalone reference panel.
    Used when the user explicitly wants to see what songs are in the database.
    """
    if not references:
        st.info("No reference songs available. Set up artist data first.")
        return

    st.subheader("📚 Retrieved Reference Songs")

    for i, ref in enumerate(references, 1):
        meta = ref.get("metadata", {})
        title = meta.get("song_title", "Unknown")
        album = meta.get("album", "N/A")
        year = meta.get("year", "N/A")
        language = meta.get("language", "N/A")
        mood = meta.get("estimated_mood", "N/A")
        score = ref.get("score", 0)

        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{i}. {title}**")
                st.caption(f"Album: {album} | Year: {year}")
                st.caption(f"Language: {language} | Mood: {mood}")
            with col2:
                st.metric("Relevance", f"{(1 - score) * 100:.0f}%")

            with st.expander("View lyrics excerpt"):
                st.text(ref.get("text", "No text available"))

            st.divider()
