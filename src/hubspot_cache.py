"""SQLite-backed cache for HubSpot CRM Documents."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document

from .config import HUBSPOT_CACHE_PATH, HUBSPOT_CACHE_TTL_HOURS


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(HUBSPOT_CACHE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    """Create the hubspot_cache table if it does not exist."""
    HUBSPOT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hubspot_cache (
                object_type TEXT NOT NULL,
                hs_object_id TEXT NOT NULL,
                page_content TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                synced_at TEXT NOT NULL,
                PRIMARY KEY (object_type, hs_object_id)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hubspot_cache_object_type "
            "ON hubspot_cache(object_type)"
        )


def save_hubspot_docs(docs: list[Document]) -> None:
    """Upsert Documents into the HubSpot cache."""
    if not docs:
        return
    _init_db()
    synced_at = datetime.utcnow().isoformat() + "Z"
    with _get_conn() as conn:
        for doc in docs:
            obj_type = doc.metadata.get("object_type", "unknown")
            hs_id = str(doc.metadata.get("hs_object_id", ""))
            metadata_json = json.dumps(doc.metadata)
            conn.execute(
                """
                INSERT INTO hubspot_cache
                    (object_type, hs_object_id, page_content, metadata_json, synced_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (object_type, hs_object_id) DO UPDATE SET
                    page_content = excluded.page_content,
                    metadata_json = excluded.metadata_json,
                    synced_at = excluded.synced_at
                """,
                (obj_type, hs_id, doc.page_content, metadata_json, synced_at),
            )


def load_hubspot_docs(
    object_types: list[str] | None = None,
) -> list[Document]:
    """
    Load Documents from the HubSpot cache.

    Args:
        object_types: If provided, only load docs with these object_type values.
                     e.g. ["contact", "company"]. If None, load all.

    Returns:
        List of Document objects.
    """
    _init_db()
    with _get_conn() as conn:
        if object_types:
            placeholders = ",".join("?" * len(object_types))
            rows = conn.execute(
                f"""
                SELECT object_type, hs_object_id, page_content, metadata_json
                FROM hubspot_cache
                WHERE object_type IN ({placeholders})
                ORDER BY object_type, hs_object_id
                """,
                object_types,
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT object_type, hs_object_id, page_content, metadata_json
                FROM hubspot_cache
                ORDER BY object_type, hs_object_id
                """
            ).fetchall()

    docs: list[Document] = []
    for row in rows:
        metadata = json.loads(row["metadata_json"])
        docs.append(
            Document(
                page_content=row["page_content"],
                metadata=metadata,
            )
        )
    return docs


def get_cache_timestamp() -> datetime | None:
    """Return when the cache was last updated, or None if empty."""
    if not HUBSPOT_CACHE_PATH.exists():
        return None
    _init_db()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT MAX(synced_at) AS latest FROM hubspot_cache"
        ).fetchone()
    if row is None or row["latest"] is None:
        return None
    try:
        s = row["latest"].replace("Z", "")
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def is_cache_stale() -> bool:
    """Return True if cache is empty or older than TTL."""
    if HUBSPOT_CACHE_TTL_HOURS is None or HUBSPOT_CACHE_TTL_HOURS <= 0:
        return False
    ts = get_cache_timestamp()
    if ts is None:
        return True
    now = datetime.utcnow()
    if ts.tzinfo:
        ts = ts.replace(tzinfo=None)
    delta = (now - ts).total_seconds()
    return delta > (HUBSPOT_CACHE_TTL_HOURS * 3600)


def get_cache_counts() -> dict[str, int]:
    """Return count of cached records per object_type."""
    if not HUBSPOT_CACHE_PATH.exists():
        return {}
    _init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT object_type, COUNT(*) AS cnt
            FROM hubspot_cache
            GROUP BY object_type
            """
        ).fetchall()
    return {row["object_type"]: row["cnt"] for row in rows}


def clear_cache() -> None:
    """Remove all cached HubSpot documents."""
    if not HUBSPOT_CACHE_PATH.exists():
        return
    _init_db()
    with _get_conn() as conn:
        conn.execute("DELETE FROM hubspot_cache")
