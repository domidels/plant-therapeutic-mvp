# app/services/turso_db.py
from __future__ import annotations

import os
from typing import Optional
from app.services.turso_db import get_resume_by_pub_id, increment_searched, insert_resume

from libsql_client import create_client

TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL", "").strip()
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "").strip()

def _check_cfg():
    if not TURSO_DATABASE_URL:
        raise RuntimeError("TURSO_DATABASE_URL missing")
    if not TURSO_AUTH_TOKEN:
        raise RuntimeError("TURSO_AUTH_TOKEN missing")

async def get_resume_by_pub_id(pub_id: str) -> Optional[str]:
    _check_cfg()
    async with create_client(TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN) as db:
        rs = await db.execute(
            "SELECT resume FROM pub_resume WHERE pub_id = ? LIMIT 1;",
            (pub_id,),
        )
        rows = rs.rows or []
        if not rows:
            return None
        return rows[0][0]

async def increment_searched(pub_id: str) -> None:
    _check_cfg()
    async with create_client(TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN) as db:
        await db.execute(
            "UPDATE pub_resume SET searched = COALESCE(searched, 0) + 1 WHERE pub_id = ?;",
            (pub_id,),
        )

async def insert_resume(pub_id: str, resume: str) -> None:
    _check_cfg()
    async with create_client(TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN) as db:
        # insertion_date est gérée par DEFAULT CURRENT_TIMESTAMP
        await db.execute(
            """
            INSERT INTO pub_resume (pub_id, resume, searched)
            VALUES (?, ?, 1);
            """,
            (pub_id, resume),
        )

async def upsert_increment_or_insert(pub_id: str, resume_if_insert: str) -> None:
    """
    Si pub_id est UNIQUE, on peut faire un UPSERT simple :
    - si existe: increment searched
    - sinon: insert resume + searched=1
    """
    _check_cfg()
    async with create_client(TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN) as db:
        await db.execute(
            """
            INSERT INTO pub_resume (pub_id, resume, searched)
            VALUES (?, ?, 1)
            ON CONFLICT(pub_id) DO UPDATE SET
              searched = COALESCE(pub_resume.searched, 0) + 1;
            """,
            (pub_id, resume_if_insert),
        )