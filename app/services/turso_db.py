"""
app/services/turso_db.py
------------------------
Turso (LibSQL) database client and helpers.

Schema overview
~~~~~~~~~~~~~~~
- ``pub_resume``             : LLM-generated article summaries (cache).
- ``condition``              : Search counter per corrected condition name.
- ``condition_misspellings`` : Cache mapping raw user input -> corrected condition.
- ``usage_daily``            : Per-identity daily quota counters.
                               ``user_id`` is either a hashed client IP or the
                               reserved key ``_global`` for the service-wide ceiling.
"""
from __future__ import annotations
import datetime
import os
from typing import Optional, Sequence, Any
from libsql_client import create_client
from app.core.config import settings

TURSO_DATABASE_URL = getattr(settings, "TURSO_DATABASE_URL", None) or os.getenv("TURSO_DATABASE_URL", None)
TURSO_AUTH_TOKEN = getattr(settings, "TURSO_AUTH_TOKEN", None) or os.getenv("TURSO_AUTH_TOKEN", None)


def _check_cfg():
    """Raise RuntimeError if required Turso credentials are missing."""
    if not TURSO_DATABASE_URL:
        raise RuntimeError("TURSO_DATABASE_URL missing")
    if not TURSO_AUTH_TOKEN:
        raise RuntimeError("TURSO_AUTH_TOKEN missing")


# ---------------------------------------------------------------------
# Generic helpers (reusable across the app)
# ---------------------------------------------------------------------
async def db_execute(sql: str, params: Sequence[Any] = ()) -> None:
    """Execute a statement (INSERT/UPDATE/DDL)."""
    _check_cfg()
    async with create_client(TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN) as db:
        await db.execute(sql, tuple(params))


async def db_fetchone(sql: str, params: Sequence[Any] = ()) -> Optional[tuple]:
    """Execute a SELECT and return first row, or None."""
    _check_cfg()
    async with create_client(TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN) as db:
        rs = await db.execute(sql, tuple(params))
        rows = rs.rows or []
        return rows[0] if rows else None


# ---------------------------------------------------------------------
# Schema init (safe to call multiple times)
# ---------------------------------------------------------------------
async def init_db() -> None:
    """
    Create all required tables and indexes if they do not already exist.

    Safe to call multiple times (idempotent ``CREATE … IF NOT EXISTS``).
    Tables created: ``pub_resume``, ``condition``, ``condition_misspellings``,
    ``usage_daily``.
    """
    # --- pub_resume (cache summaries) ---
    await db_execute(
        """
        CREATE TABLE IF NOT EXISTS pub_resume (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pub_id TEXT NOT NULL,
            resume TEXT,
            insertion_date TEXT DEFAULT CURRENT_TIMESTAMP,
            searched INTEGER DEFAULT 0,
            verdict INTEGER DEFAULT NULL
        );
        """
    )

    await db_execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS pub_id_uq_idx
        ON pub_resume(pub_id);
        """
    )

    # Add verdict column to existing databases (ignored if already present)
    try:
        await db_execute("ALTER TABLE pub_resume ADD COLUMN verdict INTEGER DEFAULT NULL;")
    except Exception:
        pass

    # --- condition (search counter per corrected condition) ---
    await db_execute(
        """
        CREATE TABLE IF NOT EXISTS condition (
          condition TEXT PRIMARY KEY,
          searched INTEGER NOT NULL DEFAULT 0,
          inserted_date TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)
        );
        """
    )

    # --- condition_misspellings (cache misspelling -> corrected) ---
    await db_execute(
        """
        CREATE TABLE IF NOT EXISTS condition_misspellings (
          misspelling TEXT PRIMARY KEY,
          condition TEXT NOT NULL,
          misspelled_searched INTEGER NOT NULL DEFAULT 0,
          inserted_date TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)
        );
        """
    )

    await db_execute(
        """
        CREATE INDEX IF NOT EXISTS idx_condition_misspellings_condition
          ON condition_misspellings(condition);
        """
    )

    # --- usage_daily ---
    await db_execute("""
        CREATE TABLE IF NOT EXISTS usage_daily (
            user_id TEXT NOT NULL,
            day TEXT NOT NULL, -- 'YYYY-MM-DD' UTC
            hf_tokens INTEGER NOT NULL DEFAULT 0,
            turso_ops INTEGER NOT NULL DEFAULT 0,
            explore_calls INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, day)
        );
    """)
    await db_execute("""CREATE INDEX IF NOT EXISTS idx_usage_daily_user ON usage_daily(user_id);""")

    # --- user_sessions (one row per hashed IP) ---
    await db_execute("""
        CREATE TABLE IF NOT EXISTS user_sessions (
            user_id     TEXT PRIMARY KEY,
            first_seen  TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
            last_seen   TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
            visit_count INTEGER NOT NULL DEFAULT 1
        );
    """)

    # --- user_searches (one row per condition query) ---
    await db_execute("""
        CREATE TABLE IF NOT EXISTS user_searches (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT NOT NULL,
            condition   TEXT NOT NULL,
            searched_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)
        );
    """)
    await db_execute("""CREATE INDEX IF NOT EXISTS idx_user_searches_user ON user_searches(user_id);""")


# ---------------------------------------------------------------------
# pub_resume functions
# ---------------------------------------------------------------------
async def get_resume_by_pub_id(pub_id: str) -> Optional[tuple]:
    """
    Return ``(resume_text, verdict)`` for a cached article, or ``None`` if not cached.

    ``verdict`` is ``1`` (negative), ``0`` (positive/neutral), or ``None`` (unknown).
    """
    row = await db_fetchone(
        "SELECT resume, verdict FROM pub_resume WHERE pub_id = ? LIMIT 1;",
        (pub_id,),
    )
    return (row[0], row[1]) if row else None


async def increment_searched(pub_id: str) -> None:
    """Increment the ``searched`` hit counter for a cached article summary."""
    await db_execute(
        "UPDATE pub_resume SET searched = COALESCE(searched, 0) + 1 WHERE pub_id = ?;",
        (pub_id,),
    )


async def get_negative_pub_ids(pub_ids: list[str]) -> set[str]:
    """Return the subset of ``pub_ids`` that have ``verdict = 1`` in ``pub_resume``."""
    if not pub_ids:
        return set()
    placeholders = ",".join("?" * len(pub_ids))
    _check_cfg()
    async with create_client(TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN) as db:
        rs = await db.execute(
            f"SELECT pub_id FROM pub_resume WHERE verdict = 1 AND pub_id IN ({placeholders});",
            tuple(pub_ids),
        )
        return {row[0] for row in (rs.rows or [])}


async def insert_resume(pub_id: str, resume: str, verdict: Optional[int] = None) -> None:
    """Insert a new LLM-generated summary into the cache (``searched`` initialised to 1).

    ``verdict``: ``1`` = negative context, ``0`` = positive/neutral, ``None`` = unknown.
    """
    await db_execute(
        """
        INSERT INTO pub_resume (pub_id, resume, searched, verdict)
        VALUES (?, ?, 1, ?);
        """,
        (pub_id, resume, verdict),
    )




# ---------------------------
# Session tracking
# ---------------------------
async def upsert_user_session(user_id: str) -> None:
    """
    Record a visit for ``user_id`` (hashed IP).

    Creates a row on first visit; on subsequent visits increments
    ``visit_count`` and updates ``last_seen``.
    """
    await db_execute(
        """
        INSERT INTO user_sessions(user_id, first_seen, last_seen, visit_count)
        VALUES(?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1)
        ON CONFLICT(user_id) DO UPDATE SET
            last_seen   = CURRENT_TIMESTAMP,
            visit_count = visit_count + 1;
        """,
        (user_id,),
    )


async def insert_user_search(user_id: str, condition: str) -> None:
    """Record a condition search for ``user_id``."""
    await db_execute(
        "INSERT INTO user_searches(user_id, condition) VALUES(?, ?);",
        (user_id, condition),
    )


def _utc_day_str() -> str:
    """Return today's date in ``YYYY-MM-DD`` UTC format for quota bucketing."""
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


# ---------------------------
# Usage / quota helpers
# ---------------------------
async def usage_get(user_id: str) -> tuple[int, int, int]:
    """
    Return today's quota counters for ``user_id`` as ``(hf_tokens, turso_ops, explore_calls)``.

    Creates a zeroed row for today if none exists yet.
    ``user_id`` is either a hashed client IP or ``_global`` for the service-wide counter.
    """
    day = _utc_day_str()
    row = await db_fetchone(
        "SELECT hf_tokens, turso_ops, explore_calls FROM usage_daily WHERE user_id=? AND day=? LIMIT 1;",
        (user_id, day),
    )
    if not row:
        await db_execute(
            "INSERT INTO usage_daily(user_id, day, hf_tokens, turso_ops, explore_calls) VALUES(?, ?, 0, 0, 0);",
            (user_id, day),
        )
        return (0, 0, 0)
    return (int(row[0] or 0), int(row[1] or 0), int(row[2] or 0))


async def usage_add(user_id: str, add_hf_tokens: int = 0, add_turso_ops: int = 0, add_explore: int = 0) -> None:
    """Atomically increment today's quota counters for ``user_id`` using an UPSERT."""
    day = _utc_day_str()
    await db_execute(
        """
        INSERT INTO usage_daily(user_id, day, hf_tokens, turso_ops, explore_calls)
        VALUES(?, ?, ?, ?, ?)
        ON CONFLICT(user_id, day) DO UPDATE SET
          hf_tokens = hf_tokens + excluded.hf_tokens,
          turso_ops = turso_ops + excluded.turso_ops,
          explore_calls = explore_calls + excluded.explore_calls;
        """,
        (user_id, day, add_hf_tokens, add_turso_ops, add_explore),
    )