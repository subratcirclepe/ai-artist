"""Graph-powered RAG pipeline for song generation and artist chat.

Replaces the flat vector-based rag_chain.py with a knowledge graph-driven
multi-stage retrieval pipeline. Falls back to the original RAG chain
when the knowledge graph is not available for an artist.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.utils import load_artist_config, PROCESSED_DIR
from src.graph.connection import graph_exists
from src.retrieval.pipeline import (
    analyze_request,
    execute_retrieval_pipeline,
    RequestAnalysis,
    RetrievalResult,
)
from src.prompt.assembler import (
    build_graph_system_prompt,
    build_graph_generation_prompt,
    build_graph_chat_prompt,
)
from src.validation.validator import validate_output, ValidationReport
from src.validation.regenerator import (
    build_repair_prompt,
    build_enhanced_regeneration_prompt,
    select_best_attempt,
)
from src.rag_chain import invoke_with_retry  # Reuse the multi-provider LLM fallback


MAX_REGENERATION_ATTEMPTS = 3


def generate_song_with_graph(
    artist_slug: str,
    topic: str,
    k: int = 5,
    temperature: float = 0.85,
) -> dict:
    """Generate a song using the knowledge graph-powered pipeline.

    Falls back to the original RAG chain if no graph exists.

    Args:
        artist_slug: Artist identifier.
        topic: Song topic.
        k: Number of reference sections (for hybrid search).
        temperature: LLM temperature.

    Returns:
        Dict with song, references, artist, topic, validation_report.
    """
    # Fallback to original RAG if no graph
    if not graph_exists(artist_slug):
        from src.rag_chain import generate_song
        return generate_song(artist_slug, topic, k=k, temperature=temperature)

    artist_config = load_artist_config(artist_slug)
    artist_name = artist_config["name"]

    # Stage 0: Analyze request
    request = analyze_request(topic)

    # Execute 7-stage retrieval
    retrieval = execute_retrieval_pipeline(artist_slug, request)

    # Build two-part prompt
    system_prompt = build_graph_system_prompt(artist_name, artist_slug, retrieval)
    generation_prompt = build_graph_generation_prompt(topic, artist_name, request, retrieval)

    # Load existing lines for originality check
    existing_lines = _load_existing_lines(artist_slug)

    # Generation + validation loop
    attempts = []
    for attempt in range(MAX_REGENERATION_ATTEMPTS):
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=generation_prompt if attempt == 0 else generation_prompt),
        ]

        # On retry, use enhanced prompts
        if attempt > 0 and attempts:
            last_output, last_report = attempts[-1]
            if last_report.recommendation == "regenerate_partial":
                repair = build_repair_prompt(last_output, last_report, artist_name)
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=repair),
                ]
            else:
                enhanced = build_enhanced_regeneration_prompt(
                    last_output, last_report, artist_name, topic
                )
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=enhanced),
                ]

        response = invoke_with_retry(messages, temperature=temperature)
        output_text = response.content

        # Validate
        report = validate_output(
            output_text=output_text,
            artist_slug=artist_slug,
            vocabulary_set=retrieval.fingerprint.get("vocabulary_set", []) if isinstance(retrieval.fingerprint, dict) else [],
            anti_vocabulary=retrieval.fingerprint.get("anti_vocabulary", []) if isinstance(retrieval.fingerprint, dict) else [],
            expected_structure=retrieval.structures[0]["pattern"] if retrieval.structures else "",
            existing_lines=existing_lines,
        )

        attempts.append((output_text, report))

        if report.overall_pass:
            print(f"[Graph RAG] Validated on attempt {attempt + 1} (score: {report.overall_score:.2f})")
            break
        else:
            print(f"[Graph RAG] Attempt {attempt + 1} failed validation "
                  f"(score: {report.overall_score:.2f}, rec: {report.recommendation})")

    # Select best attempt
    best_output, best_report = select_best_attempt(attempts)

    # Build reference info
    references = [
        {
            "text": sec.get("text", "")[:200],
            "metadata": {
                "song_title": sec.get("node_id", "").split(":")[1].replace("_", " ").title()
                              if ":" in sec.get("node_id", "") else "Unknown",
                "section_type": sec.get("section_type", ""),
                "mood": sec.get("mood", ""),
            },
            "score": sec.get("score", 0),
        }
        for sec in retrieval.thematic_sections[:5]
    ]

    return {
        "song": best_output,
        "references": references,
        "artist": artist_name,
        "topic": topic,
        "validation": {
            "overall_score": best_report.overall_score,
            "passed": best_report.overall_pass,
            "attempts": len(attempts),
            "vocabulary_score": best_report.vocabulary_score,
            "originality_score": best_report.originality_score,
            "rhyme_score": best_report.rhyme_score,
        },
        "graph_powered": True,
    }


def chat_with_artist_graph(
    artist_slug: str,
    user_message: str,
    chat_history: list | None = None,
    k: int = 3,
    temperature: float = 0.7,
) -> dict:
    """Chat with the artist persona using graph-powered context.

    Falls back to original RAG chain if no graph exists.
    """
    if not graph_exists(artist_slug):
        from src.rag_chain import chat_with_artist
        return chat_with_artist(artist_slug, user_message, chat_history, k, temperature)

    artist_config = load_artist_config(artist_slug)
    artist_name = artist_config["name"]

    # Analyze request for context retrieval
    request = analyze_request(user_message)
    retrieval = execute_retrieval_pipeline(artist_slug, request)

    # Build prompts
    system_prompt = build_graph_system_prompt(artist_name, artist_slug, retrieval)
    chat_prompt = build_graph_chat_prompt(user_message, artist_name)

    # Build message history
    messages = [SystemMessage(content=system_prompt)]
    if chat_history:
        for role, content in chat_history:
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=chat_prompt))

    response = invoke_with_retry(messages, temperature=temperature)

    references = [
        {
            "text": sec.get("text", "")[:200],
            "metadata": {
                "song_title": sec.get("node_id", "").split(":")[1].replace("_", " ").title()
                              if ":" in sec.get("node_id", "") else "Unknown",
                "section_type": sec.get("section_type", ""),
            },
            "score": sec.get("score", 0),
        }
        for sec in retrieval.thematic_sections[:3]
    ]

    return {
        "response": response.content,
        "references": references,
        "graph_powered": True,
    }


def _load_existing_lines(artist_slug: str) -> list[str]:
    """Load all existing lines for an artist from the graph data file."""
    data_path = PROCESSED_DIR / f"{artist_slug}_graph_data.json"
    if not data_path.exists():
        return []

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        lines = []
        for song in graph_data.get("songs", []):
            for section in song.get("sections", []):
                for line in section.get("lines", []):
                    text = line.get("text", "").strip()
                    if text:
                        lines.append(text)
        return lines
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Full graph pipeline setup (one-time per artist)
# ---------------------------------------------------------------------------

def setup_graph_pipeline(artist_slug: str) -> dict:
    """Run the complete graph ingestion pipeline for an artist.

    This is the one-time setup that replaces the flat preprocessing pipeline.
    Runs: structural decomposition → linguistic analysis → advanced analysis →
    clustering → graph ingestion → embedding generation.

    Args:
        artist_slug: Artist identifier.

    Returns:
        Stats dict with counts of all created entities.
    """
    from src.analysis.lyric_analyzer import analyze_artist
    from src.analysis.phonetics import extract_rhyme_pairs
    from src.analysis.fingerprint import run_advanced_analysis
    from src.analysis.thematic_clustering import run_thematic_clustering
    from src.graph.ingestion import ingest_artist
    from src.graph.loader import generate_and_load_embeddings, ingest_advanced_data

    all_stats = {}

    # Phase 1-3: Structural decomposition + linguistic analysis
    print("\n=== Phase 1-3: Structural & Linguistic Analysis ===")
    graph_data = analyze_artist(artist_slug)
    all_stats["analysis"] = graph_data.get("stats", {})

    # Extract rhyme pairs
    print("\n=== Rhyme Pair Extraction ===")
    rhyme_pairs = extract_rhyme_pairs(graph_data["songs"], artist_slug)
    print(f"  Found {len(rhyme_pairs)} rhyme pairs")

    # Phase 4: Advanced analysis (LLM-powered)
    print("\n=== Phase 4: Advanced Analysis ===")
    advanced_data = run_advanced_analysis(artist_slug, graph_data)
    all_stats["advanced"] = {
        "themes": len(advanced_data.get("themes", [])),
        "metaphors": len(advanced_data.get("metaphors", [])),
        "emotional_arcs": len(advanced_data.get("emotional_arcs", [])),
    }

    # Phase 5: Thematic clustering
    print("\n=== Phase 5: Thematic Clustering ===")
    clusters = run_thematic_clustering(artist_slug, graph_data, advanced_data)
    all_stats["clusters"] = len(clusters)

    # Phase 2 continued: Graph ingestion (structural data)
    print("\n=== Graph Ingestion: Structural Data ===")
    ingestion_stats = ingest_artist(artist_slug, graph_data)
    all_stats["ingestion"] = ingestion_stats

    # Phase 6: Advanced data ingestion + embeddings
    print("\n=== Graph Ingestion: Advanced Data ===")
    from dataclasses import asdict
    adv_stats = ingest_advanced_data(
        artist_slug,
        advanced_data,
        clusters=[asdict(c) for c in clusters] if clusters else None,
        rhyme_pairs=[asdict(rp) for rp in rhyme_pairs] if rhyme_pairs else None,
    )
    all_stats["advanced_ingestion"] = adv_stats

    print("\n=== Embedding Generation ===")
    embed_stats = generate_and_load_embeddings(artist_slug, graph_data)
    all_stats["embeddings"] = embed_stats

    print("\n=== Graph Pipeline Complete ===")
    print(f"Total stats: {json.dumps(all_stats, indent=2, default=str)}")

    return all_stats
