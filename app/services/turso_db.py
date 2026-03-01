# app/services/turso_db.py
from __future__ import annotations

import os
from typing import Optional, Sequence, Any
from libsql_client import create_client
from app.core.config import settings

TURSO_DATABASE_URL = getattr(settings, "TURSO_DATABASE_URL", None) or os.getenv("TURSO_DATABASE_URL", None)
TURSO_AUTH_TOKEN = getattr(settings, "TURSO_AUTH_TOKEN", None) or os.getenv("TURSO_AUTH_TOKEN", None)


def _check_cfg():
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
    Create all required tables/indexes if missing:
    - pub_resume + unique index
    - condition + condition_misspellings
    """
    # --- pub_resume (cache summaries) ---
    await db_execute(
        """
        CREATE TABLE IF NOT EXISTS pub_resume (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pub_id TEXT NOT NULL,
            resume TEXT,
            insertion_date TEXT DEFAULT CURRENT_TIMESTAMP,
            searched INTEGER DEFAULT 0
        );
        """
    )

    await db_execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS pub_id_uq_idx
        ON pub_resume(pub_id);
        """
    )

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


# ---------------------------------------------------------------------
# pub_resume functions
# ---------------------------------------------------------------------
async def get_resume_by_pub_id(pub_id: str) -> Optional[str]:
    row = await db_fetchone(
        "SELECT resume FROM pub_resume WHERE pub_id = ? LIMIT 1;",
        (pub_id,),
    )
    return row[0] if row else None


async def increment_searched(pub_id: str) -> None:
    await db_execute(
        "UPDATE pub_resume SET searched = COALESCE(searched, 0) + 1 WHERE pub_id = ?;",
        (pub_id,),
    )


async def insert_resume(pub_id: str, resume: str) -> None:
    await db_execute(
        """
        INSERT INTO pub_resume (pub_id, resume, searched)
        VALUES (?, ?, 1);
        """,
        (pub_id, resume),
    )


async def upsert_increment_or_insert(pub_id: str, resume_if_insert: str) -> None:
    """
    UPSERT:
    - if exists: increment searched
    - else: insert resume + searched=1
    Requires unique constraint on pub_id (pub_id_uq_idx).
    """
    await db_execute(
        """
        INSERT INTO pub_resume (pub_id, resume, searched)
        VALUES (?, ?, 1)
        ON CONFLICT(pub_id) DO UPDATE SET
          searched = COALESCE(pub_resume.searched, 0) + 1;
        """,
        (pub_id, resume_if_insert),
    )