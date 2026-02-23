# AI Artist Agent

An AI that writes songs in the exact style of your favorite Indian singer-songwriters. Powered by RAG (Retrieval-Augmented Generation) with real lyrics as style context.

## Quick Start

1. **Clone and setup:**
   ```bash
   git clone <repo>
   cd ai-artist
   bash setup.sh
   ```

   Or manually:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Add your API keys** — copy `.env.example` to `.env` and fill in:
   ```
   GENIUS_API_TOKEN=your_genius_token
   ANTHROPIC_API_KEY=your_anthropic_key
   ```
   - Get Genius token: https://genius.com/api-clients
   - Get Anthropic key: https://console.anthropic.com

3. **Run the app:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

4. **First time:** Click "Setup Artist Data" in the sidebar to scrape and embed lyrics.

5. **Start chatting!** Ask it to write songs or just chat about music.

## Supported Artists

- Anuv Jain — indie folk, Hinglish
- Arijit Singh — Bollywood ballads, Hindi/Urdu
- Prateek Kuhad — indie folk, English/Hindi

## Adding New Artists

1. Edit `config/artists.yaml` and add a new artist profile
2. Restart the app and click "Setup Artist Data" for the new artist

## How It Works

1. **Scraping** — Downloads lyrics from Genius API
2. **Preprocessing** — Cleans, chunks, and annotates lyrics with mood/language
3. **Embedding** — Creates vector embeddings using sentence-transformers
4. **RAG** — Retrieves relevant lyrics as style context when generating
5. **Generation** — Claude writes original songs matching the artist's exact style

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Claude (Anthropic API) |
| Framework | LangChain |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Lyrics Source | Genius API (lyricsgenius) |
| Web UI | Streamlit |

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
ai-artist/
├── config/          # Artist profiles and prompt templates
├── data/            # Raw lyrics, processed chunks, vectorstore
├── src/             # Core modules (scraper, preprocessor, embeddings, RAG, agent)
├── app/             # Streamlit web app and components
└── tests/           # Unit tests
```
