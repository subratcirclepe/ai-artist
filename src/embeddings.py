"""Vector database creation and management using ChromaDB."""

import argparse
import json

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.utils import (
    ensure_dirs,
    PROCESSED_DIR,
    VECTORSTORE_DIR,
    vectorstore_exists,
)

# Embedding model — multilingual, runs locally
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_function() -> HuggingFaceEmbeddings:
    """Create the HuggingFace embedding function."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vectorstore(artist_slug: str, force: bool = False) -> Chroma:
    """
    Create a ChromaDB vectorstore from processed lyrics data.

    Args:
        artist_slug: Artist identifier slug.
        force: If True, recreate even if vectorstore exists.

    Returns:
        Chroma vectorstore instance.
    """
    ensure_dirs()
    persist_dir = str(VECTORSTORE_DIR / artist_slug)

    if vectorstore_exists(artist_slug) and not force:
        print(f"Vectorstore already exists for {artist_slug}. Use --force to recreate.")
        return load_vectorstore(artist_slug)

    # Load processed data
    processed_path = PROCESSED_DIR / f"{artist_slug}_processed.json"
    if not processed_path.exists():
        raise FileNotFoundError(
            f"No processed data at {processed_path}. Run preprocessor first."
        )

    with open(processed_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        raise ValueError(f"No chunks found for {artist_slug}")

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [c["id"] for c in chunks]

    print(f"Embedding {len(texts)} chunks for {artist_slug}...")
    embedding_fn = get_embedding_function()

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embedding_fn,
        metadatas=metadatas,
        ids=ids,
        persist_directory=persist_dir,
        collection_name=f"{artist_slug}_lyrics",
    )

    print(f"Vectorstore created with {len(texts)} documents at {persist_dir}")
    return vectorstore


def load_vectorstore(artist_slug: str) -> Chroma:
    """Load an existing ChromaDB vectorstore."""
    persist_dir = str(VECTORSTORE_DIR / artist_slug)

    if not vectorstore_exists(artist_slug):
        raise FileNotFoundError(
            f"No vectorstore found for {artist_slug}. Run create_vectorstore first."
        )

    embedding_fn = get_embedding_function()
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_fn,
        collection_name=f"{artist_slug}_lyrics",
    )
    return vectorstore


def query_similar(
    artist_slug: str,
    query_text: str,
    k: int = 5,
) -> list[dict]:
    """
    Query the vectorstore for similar lyrics chunks.

    Args:
        artist_slug: Artist identifier slug.
        query_text: The query text to search for.
        k: Number of results to return.

    Returns:
        List of dicts with 'text' and 'metadata' keys.
    """
    vectorstore = load_vectorstore(artist_slug)
    results = vectorstore.similarity_search_with_score(query_text, k=k)

    return [
        {
            "text": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score),
        }
        for doc, score in results
    ]


def get_collection_stats(artist_slug: str) -> dict:
    """Get stats about an artist's vectorstore collection."""
    vectorstore = load_vectorstore(artist_slug)
    collection = vectorstore._collection
    count = collection.count()

    return {
        "artist": artist_slug,
        "total_documents": count,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage lyrics vectorstore")
    parser.add_argument("--artist", required=True, help="Artist slug")
    parser.add_argument(
        "--action",
        choices=["create", "stats", "query"],
        default="create",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--query", type=str, default="love and rain")
    args = parser.parse_args()

    if args.action == "create":
        create_vectorstore(args.artist, force=args.force)
    elif args.action == "stats":
        stats = get_collection_stats(args.artist)
        print(stats)
    elif args.action == "query":
        results = query_similar(args.artist, args.query)
        for r in results:
            print(f"\n--- {r['metadata'].get('song_title', 'Unknown')} ---")
            print(r["text"][:200])
            print(f"Score: {r['score']:.4f}")
