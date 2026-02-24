"""SQLite layer for Facebook PSID–HubSpot contact mapping and message storage."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

_DB_PATH: Path = Path(__file__).resolve().parent.parent / "fb_messenger.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they do not exist."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fb_contact_mapping (
                psid TEXT PRIMARY KEY,
                hubspot_contact_id TEXT NOT NULL,
                contact_name TEXT NOT NULL,
                linked_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fb_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                psid TEXT NOT NULL,
                direction TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fb_messages_psid ON fb_messages(psid)"
        )


def get_contact_for_psid(psid: str) -> dict | None:
    """Return mapping for PSID if linked, else None."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT psid, hubspot_contact_id, contact_name, linked_at FROM fb_contact_mapping WHERE psid = ?",
            (psid,),
        ).fetchone()
    if row is None:
        return None
    return dict(row)


def link_psid_to_contact(psid: str, hubspot_contact_id: str, contact_name: str) -> None:
    """Link a Facebook PSID to a HubSpot contact."""
    linked_at = datetime.utcnow().isoformat() + "Z"
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO fb_contact_mapping (psid, hubspot_contact_id, contact_name, linked_at)
            VALUES (?, ?, ?, ?)
            """,
            (psid, hubspot_contact_id, contact_name, linked_at),
        )


def save_message(psid: str, direction: str, message: str) -> None:
    """Store an inbound or outbound message."""
    timestamp = datetime.utcnow().isoformat() + "Z"
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO fb_messages (psid, direction, message, timestamp) VALUES (?, ?, ?, ?)",
            (psid, direction, message, timestamp),
        )


def get_unlinked_psids() -> list[dict]:
    """Return PSIDs that have sent messages but are not yet linked to a contact."""
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT m.psid,
                   MAX(m.timestamp) AS last_message_at,
                   COUNT(*) AS message_count,
                   (SELECT message FROM fb_messages m2
                    WHERE m2.psid = m.psid AND m2.direction = 'in'
                    ORDER BY m2.timestamp DESC LIMIT 1) AS message_preview
            FROM fb_messages m
            LEFT JOIN fb_contact_mapping c ON m.psid = c.psid
            WHERE c.psid IS NULL AND m.direction = 'in'
            GROUP BY m.psid
            ORDER BY last_message_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_messages_for_psid(psid: str, limit: int = 50) -> list[dict]:
    """Return recent messages for a PSID."""
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT psid, direction, message, timestamp
            FROM fb_messages
            WHERE psid = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (psid, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_linked() -> list[dict]:
    """Return all linked PSID–contact mappings with message counts."""
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT c.psid, c.hubspot_contact_id, c.contact_name, c.linked_at,
                   (SELECT COUNT(*) FROM fb_messages m WHERE m.psid = c.psid) AS message_count
            FROM fb_contact_mapping c
            ORDER BY c.linked_at DESC
        """).fetchall()
    return [dict(r) for r in rows]
