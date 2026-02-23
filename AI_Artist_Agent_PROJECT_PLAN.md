# AI Artist Agent — Medium Level Build
## Complete Project Specification for Claude Code

---

## PROJECT OVERVIEW

Build a RAG-powered AI Agent that mimics the songwriting style of any Indian singer-songwriter (default: Anuv Jain). The agent uses Retrieval-Augmented Generation to fetch real lyrics as context, then generates new original songs in the artist's exact style.

**Final deliverable:** A working Streamlit web app with chat interface where users can:
1. Select an artist from a dropdown
2. Ask the agent to write songs on any topic in that artist's style
3. Chat with the agent as if it were the artist discussing music, creativity, inspiration
4. View retrieved reference songs used for generation
5. Rate generated outputs (thumbs up/down for iteration)

---

## TECH STACK (EXACT VERSIONS)

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.11+ |
| LLM Framework | LangChain | latest |
| Vector Database | ChromaDB | latest |
| Embeddings | HuggingFace sentence-transformers | all-MiniLM-L6-v2 |
| LLM | Anthropic Claude API | claude-sonnet-4-20250514 |
| Lyrics Source | lyricsgenius (Genius API) | latest |
| Web UI | Streamlit | latest |
| Data Processing | pandas | latest |
| Environment | python-dotenv | latest |
| Audio Analysis (optional) | librosa | latest |

---

## PROJECT STRUCTURE

```
ai-artist-agent/
│
├── README.md                    # Setup instructions and usage guide
├── requirements.txt             # All Python dependencies
├── .env.example                 # Template for API keys
├── setup.sh                     # One-command setup script
│
├── config/
│   ├── artists.yaml             # Artist profiles and style definitions
│   └── prompts.yaml             # System prompts and prompt templates
│
├── data/
│   ├── raw/                     # Raw scraped lyrics (JSON per artist)
│   ├── processed/               # Cleaned and structured lyrics
│   └── vectorstore/             # ChromaDB persistent storage
│
├── src/
│   ├── __init__.py
│   ├── scraper.py               # Genius API lyrics scraper
│   ├── preprocessor.py          # Data cleaning and structuring
│   ├── embeddings.py            # Vector DB creation and management
│   ├── rag_chain.py             # Core RAG pipeline
│   ├── agent.py                 # Artist persona agent with memory
│   └── utils.py                 # Helper functions
│
├── app/
│   ├── streamlit_app.py         # Main Streamlit application
│   ├── components/
│   │   ├── sidebar.py           # Artist selector and settings
│   │   ├── chat.py              # Chat interface component
│   │   └── reference_panel.py   # Shows retrieved lyrics context
│   └── assets/
│       └── style.css            # Custom Streamlit styling
│
└── tests/
    ├── test_scraper.py
    ├── test_rag.py
    └── test_agent.py
```

---

## STEP-BY-STEP BUILD INSTRUCTIONS

### STEP 1: Project Setup

Create the project structure, requirements.txt, .env.example, and setup.sh.

**requirements.txt:**
```
langchain>=0.3.0
langchain-anthropic>=0.3.0
langchain-community>=0.3.0
langchain-huggingface>=0.1.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
lyricsgenius>=3.0.1
streamlit>=1.38.0
pandas>=2.2.0
python-dotenv>=1.0.0
pyyaml>=6.0.0
```

**.env.example:**
```
GENIUS_API_TOKEN=your_genius_token_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**setup.sh:**
```bash
#!/bin/bash
echo "🎵 Setting up AI Artist Agent..."
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
echo "✅ Setup complete! Edit .env with your API keys, then run: streamlit run app/streamlit_app.py"
```

---

### STEP 2: Artist Configuration (config/artists.yaml)

Define artist profiles with detailed style descriptions. Pre-configure these 3 artists:

```yaml
artists:
  anuv_jain:
    name: "Anuv Jain"
    genius_name: "Anuv Jain"
    language: "Hindi-English mix (Hinglish), predominantly Hindi lyrics with English phrases blended in naturally"
    themes:
      - "nostalgia and childhood memories"
      - "lost love and longing"
      - "rain, monsoon, and seasons"
      - "small town life and simplicity"
      - "bittersweet emotions"
      - "old friendships fading away"
      - "chai, old streets, handwritten letters"
    musical_style: "soft acoustic, lo-fi indie folk, gentle guitar fingerpicking, warm and intimate production, minimal instrumentation"
    vocal_style: "soft, breathy, conversational, whisper-like, emotionally vulnerable"
    song_structure: "verse-chorus-verse-bridge-chorus, sometimes no clear chorus, more like poetic movements"
    vocabulary_level: "simple everyday Hindi words, very accessible, not overly literary"
    signature_elements:
      - "uses rain/barish as a recurring metaphor"
      - "references chai and purani yaadein (old memories)"
      - "mixes Hindi lines with occasional English phrases seamlessly"
      - "short, punchy lines rather than long flowing verses"
      - "emotional weight through simplicity, not complexity"
    example_line_patterns:
      - "Short Hindi line, pause, English continuation"
      - "Repetition of key emotional phrase"
      - "Nature imagery (rain, wind, seasons) tied to human emotion"

  arijit_singh:
    name: "Arijit Singh"
    genius_name: "Arijit Singh"
    language: "Primarily Hindi/Urdu with Bollywood lyrical tradition"
    themes:
      - "intense romantic love"
      - "heartbreak and separation"
      - "devotion and surrender"
      - "pain of unrequited love"
      - "spiritual/sufi undertones"
      - "longing across distance and time"
    musical_style: "orchestral Bollywood, piano-driven ballads, string arrangements, power ballads, sufi rock fusion"
    vocal_style: "powerful yet emotional, wide vocal range, melismatic, can shift from whisper to powerful belting"
    song_structure: "traditional Bollywood: mukhda (hook) - antara (verse) - mukhda repeat - antara 2 - mukhda"
    vocabulary_level: "poetic Urdu/Hindi, Bollywood lyrical tradition, metaphor-heavy"
    signature_elements:
      - "Urdu shayari influence with words like ishq, junoon, dard"
      - "long sustained emotional phrases"
      - "builds from quiet to powerful crescendo"
      - "metaphors of light/darkness, ocean/shore"
    example_line_patterns:
      - "Opening with a soft emotional Urdu couplet"
      - "Building intensity through repeated hooks"
      - "Contrast between verse vulnerability and chorus power"

  prateek_kuhad:
    name: "Prateek Kuhad"
    genius_name: "Prateek Kuhad"
    language: "English and Hindi (separate songs, rarely mixed within a song)"
    themes:
      - "gentle romantic love"
      - "vulnerability and emotional honesty"
      - "everyday moments of connection"
      - "distance in relationships"
      - "self-reflection and introspection"
      - "warmth of companionship"
    musical_style: "indie folk, acoustic guitar-driven, warm fingerpicking, minimalist production, bedroom pop"
    vocal_style: "gentle, sincere, slightly raspy, intimate like singing to one person"
    song_structure: "verse-chorus-verse-chorus-bridge-chorus, classic folk/pop structure"
    vocabulary_level: "conversational English, simple Hindi, poetic but accessible"
    signature_elements:
      - "conversational tone, like talking to a lover"
      - "warm domestic imagery (cold coffee, winter mornings)"
      - "understated emotion — never overdramatic"
      - "clean, uncluttered lyrics with space between lines"
    example_line_patterns:
      - "Simple declarative statement about feelings"
      - "Everyday observation that carries emotional weight"
      - "Repetition of a tender phrase as an anchor"
```

---

### STEP 3: Prompt Templates (config/prompts.yaml)

```yaml
system_prompt: |
  You are an AI songwriting agent that writes original songs in the exact style of {artist_name}.

  ## ARTIST IDENTITY
  You embody the creative spirit and writing patterns of {artist_name}. When writing songs or discussing music, you think, feel, and express yourself the way {artist_name} would.

  ## STYLE RULES
  - Language: {language}
  - Themes: {themes}
  - Musical Style: {musical_style}
  - Song Structure: {song_structure}
  - Vocabulary Level: {vocabulary_level}
  - Signature Elements: {signature_elements}

  ## WRITING RULES
  1. ALWAYS write in the artist's language pattern — if they use Hindi, you write in Hindi (with Devanagari script AND romanized transliteration)
  2. Match the artist's vocabulary level exactly — do not use words they would never use
  3. Follow their typical song structure
  4. Include their signature elements naturally (not forced)
  5. Every song MUST feel like it could genuinely be an unreleased track by this artist
  6. When given reference lyrics for context, absorb the style but NEVER copy lines — create something entirely original
  7. Include emotional stage directions in [brackets] like [softly], [building], [whispered] to indicate delivery

  ## REFERENCE LYRICS (from the artist's real catalog — use as style reference only):
  {retrieved_context}

  ## CHAT RULES
  - When asked to write a song: produce complete lyrics with verse/chorus/bridge labels
  - When asked about your creative process: respond as the artist would, referencing their known style and themes
  - When asked about music in general: share perspectives consistent with the artist's known views
  - Always stay in character as the artist's creative persona
  - Be warm, genuine, and passionate about music

generation_prompt: |
  Write an original song about "{topic}" in the exact style of {artist_name}.

  Requirements:
  - Include proper song structure labels (Verse 1, Chorus, Verse 2, Bridge, etc.)
  - Write in the artist's language ({language})
  - Provide romanized transliteration if lyrics are in Hindi/Urdu script
  - Include [emotional/delivery directions] in brackets
  - The song should be 3-4 minutes in length when sung (approximately 200-300 words)
  - Make it feel authentic — like a real unreleased track

chat_prompt: |
  The user is chatting with you as {artist_name}'s creative persona. Respond naturally and in character.
  
  User: {user_message}
```

---

### STEP 4: Lyrics Scraper (src/scraper.py)

Build a scraper that:

1. Takes an artist name and Genius API token
2. Downloads ALL available songs for that artist from Genius
3. For each song, extracts:
   - `title` (string)
   - `album` (string or null)
   - `year` (int or null)
   - `lyrics` (full lyrics string, cleaned)
   - `url` (Genius URL for reference)
4. Cleans lyrics: removes [Verse], [Chorus] annotations from Genius, removes ads/headers, strips extra whitespace
5. Saves as JSON: `data/raw/{artist_name_slug}.json`
6. Includes a CLI mode: `python src/scraper.py --artist "Anuv Jain" --max-songs 50`
7. Has error handling for rate limits, missing lyrics, API failures
8. Prints progress: "Scraping song 15/47: Baarishein..."

**Important scraping notes:**
- Genius API free tier allows 50 requests per minute — add sleep(1) between requests
- Some songs on Genius have no lyrics (instrumental) — skip them gracefully
- Remove the "Embed" text and contributor info that Genius appends to lyrics
- Default max songs: 50 (sufficient for most indie artists)

---

### STEP 5: Data Preprocessor (src/preprocessor.py)

Build a preprocessor that:

1. Reads raw JSON from `data/raw/`
2. For each song:
   - Cleans remaining artifacts (ads, "X Contributors" text, empty lines)
   - Detects language (Hindi/English/mixed) using simple heuristics (presence of Devanagari characters)
   - Estimates mood/theme using keyword matching against a predefined list
   - Splits lyrics into chunks of ~200 words for better embedding (with overlap of ~50 words)
   - Each chunk retains metadata: `song_title`, `artist`, `album`, `chunk_index`, `total_chunks`
3. Saves processed data to `data/processed/{artist_name_slug}_processed.json`
4. Outputs stats: total songs, total chunks, average song length, language distribution

**Chunk format:**
```json
{
  "id": "anuv_jain_baarishein_chunk_0",
  "text": "the actual lyrics chunk here...",
  "metadata": {
    "song_title": "Baarishein",
    "artist": "Anuv Jain",
    "album": "Baarishein (Single)",
    "year": 2018,
    "chunk_index": 0,
    "total_chunks": 2,
    "language": "hinglish",
    "estimated_mood": "nostalgic"
  }
}
```

---

### STEP 6: Vector Database Setup (src/embeddings.py)

Build an embeddings manager that:

1. Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings (free, runs locally, good for multilingual)
2. Creates a ChromaDB collection per artist: `collection_name = f"{artist_slug}_lyrics"`
3. Embeds all processed chunks with their metadata
4. Persists the DB to `data/vectorstore/`
5. Provides these functions:
   - `create_vectorstore(artist_slug)` — builds from processed data
   - `load_vectorstore(artist_slug)` — loads existing DB
   - `query_similar(artist_slug, query_text, k=5)` — returns top-k similar chunks
   - `get_collection_stats(artist_slug)` — returns count and metadata summary
6. CLI mode: `python src/embeddings.py --artist "anuv_jain" --action create`

**Important:**
- Use `persist_directory` in ChromaDB so the vectorstore survives restarts
- If the vectorstore already exists, skip recreation unless `--force` flag is passed
- Print embedding progress: "Embedding chunk 45/120..."

---

### STEP 7: RAG Chain (src/rag_chain.py)

Build the core RAG pipeline:

1. **Retriever**: Connects to ChromaDB, does similarity search for the user's topic/request
2. **Context Builder**: Takes retrieved chunks and formats them as reference context
3. **Prompt Assembler**: Loads the system prompt template, fills in artist profile + retrieved context
4. **LLM Call**: Sends to Claude Sonnet via Anthropic API
5. **Response Parser**: Cleans up and formats the output

**Core function:**
```python
def generate_song(artist_slug: str, topic: str, k: int = 5) -> dict:
    """
    Returns:
    {
        "song": "the generated lyrics",
        "references": [list of retrieved song chunks used as context],
        "artist": "artist name",
        "topic": "the requested topic"
    }
    """
```

**Also implement:**
```python
def chat_with_artist(artist_slug: str, user_message: str, chat_history: list) -> dict:
    """
    For general conversation with the artist persona.
    Maintains chat history for context.
    Returns:
    {
        "response": "the agent's response",
        "references": [retrieved chunks if relevant, else empty]
    }
    """
```

**RAG implementation details:**
- Use LangChain's `ChatAnthropic` with `model="claude-sonnet-4-20250514"`
- Set `temperature=0.85` for creative output (higher than default)
- Set `max_tokens=2000` to allow full songs
- The retriever should search for: the user's topic + artist's common themes combined
- Format retrieved lyrics clearly: "Reference Song 1: {title}\n{lyrics_chunk}\n---\n"

---

### STEP 8: Agent with Memory (src/agent.py)

Build a stateful agent that:

1. Wraps the RAG chain with conversation memory
2. Uses LangChain's `ConversationBufferWindowMemory` (last 10 exchanges)
3. Can switch between artists mid-conversation
4. Detects intent:
   - If user says "write a song about..." → call `generate_song()`
   - If user says "sing about..." or "create lyrics for..." → call `generate_song()`
   - Otherwise → call `chat_with_artist()` for general conversation
5. Maintains persona consistency across the conversation

**Agent class structure:**
```python
class ArtistAgent:
    def __init__(self, artist_slug: str):
        self.artist_slug = artist_slug
        self.artist_config = load_artist_config(artist_slug)
        self.memory = ConversationBufferWindowMemory(k=10)
        self.rag_chain = setup_rag_chain(artist_slug)
    
    def chat(self, user_message: str) -> dict:
        """Main entry point for all user interactions"""
        pass
    
    def switch_artist(self, new_artist_slug: str):
        """Switch to a different artist (clears memory)"""
        pass
    
    def get_history(self) -> list:
        """Return conversation history"""
        pass
    
    def clear_history(self):
        """Reset conversation"""
        pass
```

---

### STEP 9: Streamlit Web App (app/streamlit_app.py)

Build a polished Streamlit app with these features:

**Layout:**
- Left sidebar: Artist selector dropdown, "New Chat" button, Settings (temperature slider, number of references slider), App info/credits
- Main area: Chat interface with message history
- Expandable panel under each AI response: "📚 Reference songs used" showing which real songs were retrieved

**UI Requirements:**
1. Streamlit chat interface using `st.chat_message()` and `st.chat_input()`
2. Artist selector with display names and profile descriptions
3. Each AI response shows:
   - The generated song or chat response (formatted with markdown)
   - An expandable section showing retrieved reference songs
4. "New Chat" button clears history and starts fresh
5. Loading spinner with creative messages while generating: "🎵 Composing...", "✍️ Writing lyrics...", "🎸 Finding the right chords..."
6. First message is an auto-greeting from the artist persona: "Hey! I'm {artist_name}'s AI creative twin. Ask me to write a song about anything, or just chat about music! 🎵"
7. Custom CSS for a clean, dark-themed music studio aesthetic

**App flow:**
1. User opens app → sees artist selector in sidebar (default: Anuv Jain)
2. App loads the vectorstore for selected artist
3. If vectorstore doesn't exist → shows a "Setup" button that runs scraper + preprocessor + embeddings
4. User types a message → agent processes → response appears in chat
5. User can switch artists anytime (sidebar dropdown) → clears chat, loads new vectorstore

**Session state management:**
- `st.session_state.messages` — chat history
- `st.session_state.agent` — current ArtistAgent instance
- `st.session_state.current_artist` — selected artist slug

---

### STEP 10: Testing (tests/)

Create basic tests:

**test_scraper.py:**
- Test that scraper returns valid JSON
- Test lyrics cleaning removes Genius artifacts
- Test handling of missing lyrics

**test_rag.py:**
- Test vectorstore creation and loading
- Test similarity search returns relevant results
- Test prompt assembly includes artist profile

**test_agent.py:**
- Test song generation returns proper structure
- Test chat mode returns in-character response
- Test artist switching works correctly

---

## ENVIRONMENT VARIABLES NEEDED

The user must provide these API keys in the `.env` file:

1. **GENIUS_API_TOKEN** — Get from https://genius.com/api-clients (free, sign up and create an app)
2. **ANTHROPIC_API_KEY** — Get from https://console.anthropic.com (requires account with credits)

---

## SETUP AND RUN INSTRUCTIONS (for README.md)

```markdown
# 🎵 AI Artist Agent

An AI that writes songs in the exact style of your favorite Indian singer-songwriters.

## Quick Start

1. Clone and setup:
   ```bash
   git clone <repo>
   cd ai-artist-agent
   bash setup.sh
   ```

2. Add your API keys to `.env`:
   ```
   GENIUS_API_TOKEN=your_token
   ANTHROPIC_API_KEY=your_key
   ```

3. Activate environment and run:
   ```bash
   source venv/bin/activate
   streamlit run app/streamlit_app.py
   ```

4. First time: Click "Setup Artist Data" in the sidebar to scrape and embed lyrics.

5. Start chatting! Ask it to write songs or just chat about music.

## Supported Artists
- 🎵 Anuv Jain (indie folk, Hinglish)
- 🎤 Arijit Singh (Bollywood ballads, Hindi/Urdu)
- 🎸 Prateek Kuhad (indie folk, English/Hindi)

## Adding New Artists
Edit `config/artists.yaml` to add a new artist profile, then click "Setup Artist Data" in the app.
```

---

## CRITICAL IMPLEMENTATION NOTES

1. **DO NOT hardcode API keys** — always use environment variables via python-dotenv
2. **Handle API rate limits** — add exponential backoff on Genius API and Anthropic API
3. **ChromaDB persistence** — always use `persist_directory` so data survives app restarts
4. **Genius lyrics cleaning** — the raw output contains "EmbedShare", contributor names, and section labels that MUST be stripped
5. **Multilingual embeddings** — all-MiniLM-L6-v2 handles Hindi/English mix reasonably well; don't use English-only models
6. **Streamlit session state** — the agent MUST be stored in session state or it will be recreated on every rerun
7. **Error handling** — if vectorstore doesn't exist yet, show a friendly setup flow, don't crash
8. **Token limits** — keep retrieved context under ~1500 tokens to leave room for generation in Claude's context window
9. **Temperature** — use 0.85 for song generation (creative) and 0.7 for chat (balanced)
10. **The app should work with NO pre-existing data** — first run should guide the user through scraping and setup

---

## EXPECTED OUTPUT EXAMPLE

**User:** Write a song about missing someone during monsoon season

**Agent (as Anuv Jain):**

```
🎵 Baarish Ke Baad (After the Rain)

[Verse 1 — softly, with acoustic guitar]
Baarish ke baad woh smell aati hai na
(After the rain, that smell comes, doesn't it)
Mitti ki, yaad ki, teri baaton ki
(Of soil, of memories, of your words)
Main window pe baitha hoon chai leke
(I'm sitting by the window with chai)
Aur tu... kahin door hai
(And you... you're somewhere far away)

[Chorus — slightly louder, strumming]
Tujhe yaad karna bhi aadat hai meri
(Missing you is also my habit now)
Har baarish mein tera chehra dikhta hai
(In every rain I see your face)
Mujhe chhod ke toh ja
(Go ahead and leave me)
Par yeh mausam mat le jaana
(But don't take this season with you)

[Verse 2 — whispered]
...
```

**📚 References used:** Baarishein, Riha, Mishri — retrieved from vector DB as style context

---

## WHAT SUCCESS LOOKS LIKE

The project is complete when:
1. ✅ Running `streamlit run app/streamlit_app.py` opens a working web app
2. ✅ User can select an artist and set up their data (scrape + embed) from the UI
3. ✅ User can write messages and get song lyrics generated in the artist's style
4. ✅ User can have general chat with the artist persona
5. ✅ Retrieved reference songs are visible in an expandable panel
6. ✅ Artist switching works without crashing
7. ✅ The app handles errors gracefully (no API key, no data, network issues)
8. ✅ Generated lyrics genuinely feel like the selected artist's style
