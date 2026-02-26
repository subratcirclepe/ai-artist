"""KuzuDB connection management for the Music Knowledge Graph.

Modeled after GitNexus's kuzu-adapter.ts — provides a singleton connection
to a KuzuDB database with schema initialization and query execution.
"""

import os
import threading
from pathlib import Path

import kuzu

from src.utils import DATA_DIR
from src.graph.schema import NODE_TABLE_QUERIES, REL_TABLE_QUERIES, INDEX_QUERIES

# Default graph storage location
GRAPH_DIR = DATA_DIR / "graphstore"

# Global lock for thread-safe DB operations (KuzuDB allows one write txn)
_db_lock = threading.Lock()

# Singleton instances
_db_instance: kuzu.Database | None = None
_conn_instance: kuzu.Connection | None = None


def get_graph_dir(artist_slug: str | None = None) -> Path:
    """Get the graph storage directory, optionally scoped to an artist."""
    if artist_slug:
        return GRAPH_DIR / artist_slug
    return GRAPH_DIR


def get_database(artist_slug: str | None = None) -> kuzu.Database:
    """Get or create the KuzuDB database instance.

    Args:
        artist_slug: If provided, uses a per-artist database directory.
    """
    global _db_instance
    db_path = get_graph_dir(artist_slug)
    db_path.mkdir(parents=True, exist_ok=True)

    with _db_lock:
        # Always create a new instance if the path differs or none exists
        _db_instance = kuzu.Database(str(db_path))
    return _db_instance


def get_connection(artist_slug: str | None = None) -> kuzu.Connection:
    """Get or create a KuzuDB connection.

    Args:
        artist_slug: If provided, connects to the per-artist database.
    """
    global _conn_instance
    db = get_database(artist_slug)
    with _db_lock:
        _conn_instance = kuzu.Connection(db)
    return _conn_instance


def close_connection():
    """Close the current connection and database."""
    global _conn_instance, _db_instance
    with _db_lock:
        _conn_instance = None
        _db_instance = None


def initialize_schema(conn: kuzu.Connection) -> None:
    """Create all node tables, relationship tables, and indexes.

    Idempotent — uses IF NOT EXISTS on all CREATE statements.
    """
    print("Initializing KuzuDB schema...")

    # Create node tables
    for query in NODE_TABLE_QUERIES:
        try:
            conn.execute(query)
        except Exception as e:
            # Ignore "already exists" errors
            if "already exists" not in str(e).lower():
                print(f"  Warning creating node table: {e}")

    # Create relationship tables
    for query in REL_TABLE_QUERIES:
        try:
            conn.execute(query)
        except Exception as e:
            if "already exists" not in str(e).lower():
                print(f"  Warning creating rel table: {e}")

    print("  Schema initialized successfully.")


def create_indexes(conn: kuzu.Connection) -> None:
    """Create vector and FTS indexes. Call after data is loaded."""
    print("Creating indexes...")
    for query in INDEX_QUERIES:
        try:
            conn.execute(query)
            print(f"  Index created.")
        except Exception as e:
            if "already exists" not in str(e).lower():
                print(f"  Warning creating index: {e}")
    print("  Indexes created successfully.")


def execute_query(conn: kuzu.Connection, query: str, params: dict | None = None) -> list[dict]:
    """Execute a Cypher query and return results as list of dicts.

    Args:
        conn: KuzuDB connection.
        query: Cypher query string.
        params: Optional query parameters.

    Returns:
        List of result dictionaries.
    """
    with _db_lock:
        if params:
            result = conn.execute(query, params)
        else:
            result = conn.execute(query)

        rows = []
        while result.has_next():
            row = result.get_next()
            col_names = result.get_column_names()
            rows.append(dict(zip(col_names, row)))
        return rows


def graph_exists(artist_slug: str) -> bool:
    """Check if a graph database exists for an artist."""
    db_path = get_graph_dir(artist_slug)
    return db_path.exists() and any(db_path.iterdir())
