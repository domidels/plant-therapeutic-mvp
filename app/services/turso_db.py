# app/services/turso_db.py
from __future__ import annotations
import uuid
import datetime
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

    # --- auth_users ---
    await db_execute("""
        CREATE TABLE IF NOT EXISTS auth_users (
            user_id TEXT PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
            last_login TEXT
        );
    """)

    # --- auth_otp (stocke hash du code) ---
    await db_execute("""
        CREATE TABLE IF NOT EXISTS auth_otp (
            email TEXT PRIMARY KEY,
            code_hash TEXT NOT NULL,
            expires_at INTEGER NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)
        );
    """)

    # --- auth_sessions ---
    await db_execute("""
        CREATE TABLE IF NOT EXISTS auth_sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            expires_at INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)
    );
    """)
    await db_execute("""CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_id ON auth_sessions(user_id);""")

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
    
def _utc_day_str() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


# ---------------------------
# Auth helpers
# ---------------------------
async def auth_get_user_by_email(email: str) -> Optional[tuple]:
    return await db_fetchone(
        "SELECT user_id, email FROM auth_users WHERE email=? LIMIT 1;",
        (email,),
    )


async def auth_create_user(email: str) -> str:
    user_id = str(uuid.uuid4())
    await db_execute("INSERT INTO auth_users(user_id, email) VALUES(?, ?);", (user_id, email))
    return user_id


async def auth_touch_last_login(user_id: str) -> None:
    await db_execute(
        "UPDATE auth_users SET last_login=CURRENT_TIMESTAMP WHERE user_id=?;",
        (user_id,),
    )


async def auth_upsert_otp(email: str, code_hash: str, expires_at: int) -> None:
    await db_execute(
        """
        INSERT INTO auth_otp(email, code_hash, expires_at, attempts)
        VALUES(?, ?, ?, 0)
        ON CONFLICT(email) DO UPDATE SET
          code_hash=excluded.code_hash,
          expires_at=excluded.expires_at,
          attempts=0;
        """,
        (email, code_hash, expires_at),
    )


async def auth_get_otp(email: str) -> Optional[tuple]:
    return await db_fetchone(
        "SELECT code_hash, expires_at, attempts FROM auth_otp WHERE email=? LIMIT 1;",
        (email,),
    )


async def auth_inc_otp_attempts(email: str) -> None:
    await db_execute(
        "UPDATE auth_otp SET attempts=attempts+1 WHERE email=?;",
        (email,),
    )


async def auth_delete_otp(email: str) -> None:
    await db_execute("DELETE FROM auth_otp WHERE email=?;", (email,))


async def auth_create_session(session_id: str, user_id: str, expires_at: int) -> None:
    await db_execute(
        "INSERT INTO auth_sessions(session_id, user_id, expires_at) VALUES(?, ?, ?);",
        (session_id, user_id, expires_at),
    )


async def auth_get_session(session_id: str) -> Optional[tuple]:
    return await db_fetchone(
        "SELECT user_id, expires_at FROM auth_sessions WHERE session_id=? LIMIT 1;",
        (session_id,),
    )


async def auth_delete_session(session_id: str) -> None:
    await db_execute("DELETE FROM auth_sessions WHERE session_id=?;", (session_id,))


# ---------------------------
# Usage / quota helpers
# ---------------------------
async def usage_get(user_id: str) -> tuple[int, int, int]:
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