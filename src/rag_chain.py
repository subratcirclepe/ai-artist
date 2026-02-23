"""Core RAG pipeline for song generation and artist chat."""

import os
import time

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.utils import (
    get_api_key,
    load_artist_config,
    load_prompts,
)
from src.embeddings import load_vectorstore, vectorstore_exists


def _get_available_providers() -> list[dict]:
    """
    Build an ordered list of LLM providers based on which API keys are set.
    Tries: Claude -> Gemini -> Groq -> Cohere
    """
    providers = []

    # 1. Anthropic Claude (primary)
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key and not anthropic_key.startswith("your_"):
        providers.append({"name": "anthropic", "key": anthropic_key})

    # 2. Google Gemini (fallback)
    google_key = os.getenv("GOOGLE_API_KEY", "")
    if google_key and not google_key.startswith("your_"):
        providers.append({"name": "gemini", "key": google_key})

    # 3. Groq (free, fast — Llama 3.3 70B)
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key and not groq_key.startswith("your_"):
        providers.append({"name": "groq", "key": groq_key})

    # 4. Cohere (free tier — Command-R)
    cohere_key = os.getenv("COHERE_API_KEY", "")
    if cohere_key and not cohere_key.startswith("your_"):
        providers.append({"name": "cohere", "key": cohere_key})

    return providers


def _create_llm(provider: dict, temperature: float, max_tokens: int, model_override: str | None = None):
    """Create an LLM instance for a given provider."""
    name = provider["name"]
    key = provider["key"]

    if name == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_override or "claude-sonnet-4-20250514",
            anthropic_api_key=key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif name == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_override or "gemini-2.0-flash",
            google_api_key=key,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    elif name == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_override or "llama-3.3-70b-versatile",
            groq_api_key=key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif name == "cohere":
        from langchain_cohere import ChatCohere
        return ChatCohere(
            model=model_override or "command-r-plus",
            cohere_api_key=key,
            temperature=temperature,
            max_tokens=max_tokens,
        )


# Gemini sub-models to try before moving to next provider
GEMINI_MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash"]


def invoke_with_retry(messages: list, temperature: float = 0.85, max_tokens: int = 2000):
    """
    Call LLM with retry + multi-provider fallback.
    Order: Gemini (3 models) -> Groq -> Cohere
    """
    providers = _get_available_providers()
    if not providers:
        raise RuntimeError(
            "No LLM API keys configured. Add at least one to your .env file:\n"
            "  GOOGLE_API_KEY   — https://aistudio.google.com/apikey\n"
            "  GROQ_API_KEY     — https://console.groq.com/keys (free)\n"
            "  COHERE_API_KEY   — https://dashboard.cohere.com/api-keys (free)"
        )

    errors = []

    for provider in providers:
        # For Gemini, try multiple sub-models
        if provider["name"] == "gemini":
            models_to_try = GEMINI_MODELS
        else:
            models_to_try = [None]  # use default model

        for model in models_to_try:
            for attempt in range(2):
                try:
                    llm = _create_llm(provider, temperature, max_tokens, model_override=model)
                    response = llm.invoke(messages)
                    label = model or provider["name"]
                    print(f"[LLM] Response from {label}")
                    return response
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err or "rate" in err.lower():
                        wait = (attempt + 1) * 10
                        label = model or provider["name"]
                        print(f"[LLM] Rate limited on {label}, waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        errors.append(f"{provider['name']}: {err[:100]}")
                        break  # non-rate-limit error, skip to next

        print(f"[LLM] {provider['name']} exhausted, trying next provider...")

    raise RuntimeError(
        "All LLM providers failed. Errors:\n" +
        "\n".join(f"  - {e}" for e in errors) +
        "\n\nTips:\n"
        "  - Add more free API keys to .env (GROQ_API_KEY, COHERE_API_KEY)\n"
        "  - Wait a few minutes for rate limits to reset"
    )


def retrieve_context(artist_slug: str, query: str, k: int = 5) -> list[dict]:
    """Retrieve relevant lyrics chunks from the vectorstore."""
    if not vectorstore_exists(artist_slug):
        return []

    vectorstore = load_vectorstore(artist_slug)
    results = vectorstore.similarity_search_with_score(query, k=k)

    return [
        {
            "text": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score),
        }
        for doc, score in results
    ]


def format_context(references: list[dict]) -> str:
    """Format retrieved chunks into a context string for the prompt."""
    if not references:
        return "No reference lyrics available yet. Generate based on your knowledge of the artist's style."

    parts = []
    for i, ref in enumerate(references, 1):
        title = ref["metadata"].get("song_title", "Unknown")
        text = ref["text"]
        parts.append(f"Reference Song {i}: {title}\n{text}\n---")

    return "\n\n".join(parts)


def build_system_prompt(artist_slug: str, retrieved_context: str) -> str:
    """Build the full system prompt with artist profile and context."""
    artist_config = load_artist_config(artist_slug)
    prompts = load_prompts()

    system_template = prompts["system_prompt"]

    themes = ", ".join(artist_config.get("themes", []))
    signature_elements = "\n  ".join(
        f"- {e}" for e in artist_config.get("signature_elements", [])
    )

    return system_template.format(
        artist_name=artist_config["name"],
        language=artist_config.get("language", ""),
        themes=themes,
        musical_style=artist_config.get("musical_style", ""),
        song_structure=artist_config.get("song_structure", ""),
        vocabulary_level=artist_config.get("vocabulary_level", ""),
        signature_elements=signature_elements,
        retrieved_context=retrieved_context,
    )


def generate_song(
    artist_slug: str,
    topic: str,
    k: int = 5,
    temperature: float = 0.85,
) -> dict:
    """Generate a song in the artist's style using RAG."""
    artist_config = load_artist_config(artist_slug)
    prompts = load_prompts()

    references = retrieve_context(artist_slug, topic, k=k)
    context_str = format_context(references)

    system_prompt = build_system_prompt(artist_slug, context_str)
    generation_prompt = prompts["generation_prompt"].format(
        topic=topic,
        artist_name=artist_config["name"],
        language=artist_config.get("language", ""),
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=generation_prompt),
    ]
    response = invoke_with_retry(messages, temperature=temperature)

    return {
        "song": response.content,
        "references": references,
        "artist": artist_config["name"],
        "topic": topic,
    }


def chat_with_artist(
    artist_slug: str,
    user_message: str,
    chat_history: list | None = None,
    k: int = 3,
    temperature: float = 0.7,
) -> dict:
    """Chat with the artist persona using RAG for context."""
    artist_config = load_artist_config(artist_slug)

    references = retrieve_context(artist_slug, user_message, k=k)
    context_str = format_context(references)

    system_prompt = build_system_prompt(artist_slug, context_str)

    messages = [SystemMessage(content=system_prompt)]

    if chat_history:
        for role, content in chat_history:
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=user_message))

    response = invoke_with_retry(messages, temperature=temperature)

    return {
        "response": response.content,
        "references": references,
    }
