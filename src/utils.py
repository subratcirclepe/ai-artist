"""Helper functions for the AI Artist Agent."""

import os
import re
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"


def get_api_key(key_name: str) -> str:
    """Get an API key from environment variables."""
    value = os.getenv(key_name)
    if not value or value.startswith("your_"):
        raise ValueError(
            f"{key_name} not set. Please add it to your .env file."
        )
    return value


def load_artist_config(artist_slug: str) -> dict:
    """Load artist configuration from artists.yaml."""
    config_path = CONFIG_DIR / "artists.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    artists = config.get("artists", {})
    if artist_slug not in artists:
        available = list(artists.keys())
        raise ValueError(
            f"Artist '{artist_slug}' not found. Available: {available}"
        )
    return artists[artist_slug]


def load_all_artists() -> dict:
    """Load all artist configurations."""
    config_path = CONFIG_DIR / "artists.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("artists", {})


def load_prompts() -> dict:
    """Load prompt templates from prompts.yaml."""
    prompts_path = CONFIG_DIR / "prompts.yaml"
    with open(prompts_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def slugify(name: str) -> str:
    """Convert an artist name to a URL-safe slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "_", slug)
    return slug


def ensure_dirs():
    """Ensure all data directories exist."""
    for d in [RAW_DIR, PROCESSED_DIR, VECTORSTORE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def vectorstore_exists(artist_slug: str) -> bool:
    """Check if a vectorstore already exists for an artist."""
    vs_path = VECTORSTORE_DIR / artist_slug
    return vs_path.exists() and any(vs_path.iterdir())


def raw_data_exists(artist_slug: str) -> bool:
    """Check if raw scraped data exists for an artist."""
    raw_path = RAW_DIR / f"{artist_slug}.json"
    return raw_path.exists()


def processed_data_exists(artist_slug: str) -> bool:
    """Check if processed data exists for an artist."""
    processed_path = PROCESSED_DIR / f"{artist_slug}_processed.json"
    return processed_path.exists()
