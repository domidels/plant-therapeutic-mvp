# app/api/routes.py
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
import traceback
import uuid
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.core.config import settings
from app.services.medline import get_medlineplus_fullsummary
from app.services.plants_v2 import find_plants_in_text, load_plants
from app.services.pubmed import efetch, search_and_fetch
from app.services.ranking import score_article, summarize_for_patients
from app.services.turso_db import (
    # existing cache
    get_resume_by_pub_id,
    increment_searched,
    insert_resume,
    # generic db helpers
    db_fetchone,
    db_execute,
    # auth + usage
    auth_get_user_by_email,
    auth_create_user,
    auth_touch_last_login,
    auth_upsert_otp,
    auth_get_otp,
    auth_inc_otp_attempts,
    auth_delete_otp,
    auth_create_session,
    auth_get_session,
    auth_delete_session,
    usage_get,
    usage_add,
)

router = APIRouter()

# ---------------------------------------------------------------------
# Turnstile (human gate)
# ---------------------------------------------------------------------
TURNSTILE_SECRET = (
    getattr(settings, "TURNSTILE_SECRET", None)
    or os.getenv("TURNSTILE_SECRET", "")
).strip()

_session_secret_raw = getattr(settings, "SESSION_SECRET", None) or os.getenv("SESSION_SECRET", "")
SESSION_SECRET: bytes = (_session_secret_raw or "change-me").encode("utf-8")

HUMAN_COOKIE_NAME = "human_ok"
HUMAN_COOKIE_TTL_SECONDS = 6 * 3600  # 6h


class VerifyReq(BaseModel):
    token: str


def _sign_payload(payload: str) -> str:
    sig = hmac.new(SESSION_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def _verify_signed(value: str) -> Optional[str]:
    try:
        payload, sig = value.rsplit(".", 1)
        expected = hmac.new(SESSION_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()
        if hmac.compare_digest(sig, expected):
            return payload
    except Exception:
        return None
    return None


def _is_human_cookie_valid(cookie_val: str) -> bool:
    payload = _verify_signed(cookie_val)
    if not payload:
        return False
    try:
        exp = int(payload)
    except ValueError:
        return False
    return time.time() < exp


async def require_human(request: Request) -> None:
    cookie_val = request.cookies.get(HUMAN_COOKIE_NAME)
    if not cookie_val or not _is_human_cookie_valid(cookie_val):
        raise HTTPException(status_code=401, detail="Human verification required")


@router.post("/verify_human")
async def verify_human(req: VerifyReq, request: Request):
    if not TURNSTILE_SECRET:
        raise HTTPException(status_code=500, detail="TURNSTILE_SECRET missing")

    token = (req.token or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="Missing token")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                "https://challenges.cloudflare.com/turnstile/v0/siteverify",
                data={
                    "secret": TURNSTILE_SECRET,
                    "response": token,
                    "remoteip": request.client.host if request.client else None,
                },
            )
            r.raise_for_status()
            js = r.json()
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="Turnstile upstream error")
    except Exception:
        raise HTTPException(status_code=502, detail="Turnstile verify error")

    if not js.get("success"):
        raise HTTPException(status_code=403, detail="Turnstile failed")

    exp = int(time.time()) + HUMAN_COOKIE_TTL_SECONDS
    cookie_val = _sign_payload(str(exp))

    resp = JSONResponse({"ok": True})

    is_https = (request.url.scheme or "").lower() == "https"
    secure_flag = bool(getattr(settings, "COOKIE_SECURE", False)) or is_https

    resp.set_cookie(
        HUMAN_COOKIE_NAME,
        cookie_val,
        max_age=HUMAN_COOKIE_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=secure_flag,
        path="/",
    )
    return resp


# ---------------------------------------------------------------------
# Auth (email OTP + session cookie)
# ---------------------------------------------------------------------
AUTH_SESSION_COOKIE = "pm_sess"
AUTH_SESSION_TTL_SECONDS = int(os.getenv("AUTH_SESSION_TTL_SECONDS", "1209600"))  # 14d
AUTH_OTP_TTL_SECONDS = int(os.getenv("AUTH_OTP_TTL_SECONDS", "600"))  # 10 min

RESEND_API_KEY = (os.getenv("RESEND_API_KEY", "")).strip()
EMAIL_FROM = (os.getenv("EMAIL_FROM", "Plant-Med <no-reply@plant-med.org>")).strip()

# Quotas per user per UTC day
DAILY_HF_TOKEN_LIMIT = int(os.getenv("DAILY_HF_TOKEN_LIMIT", "4000"))
DAILY_TURSO_OP_LIMIT = int(os.getenv("DAILY_TURSO_OP_LIMIT", "5000"))
DAILY_EXPLORE_LIMIT = int(os.getenv("DAILY_EXPLORE_LIMIT", "25"))


def _valid_email(email: str) -> bool:
    email = (email or "").strip()
    if len(email) < 5 or len(email) > 200:
        return False
    # very simple check
    return ("@" in email) and ("." in email.split("@")[-1])


def _hash_otp(email: str, code: str) -> str:
    msg = f"{email.lower().strip()}|{code.strip()}".encode("utf-8")
    return hmac.new(SESSION_SECRET, msg, hashlib.sha256).hexdigest()


async def _send_email_otp(email: str, code: str) -> None:
    """
    Uses Resend if configured; otherwise prints to logs (dev).
    """
    if not RESEND_API_KEY:
        print(f"[DEV OTP] email={email} code={code}")
        return

    payload = {
        "from": EMAIL_FROM,
        "to": [email],
        "subject": "Your Plant-Med sign-in code",
        "text": f"Your sign-in code is: {code}\n\nThis code expires in {AUTH_OTP_TTL_SECONDS // 60} minutes.",
    }
    headers = {"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post("https://api.resend.com/emails", headers=headers, json=payload)
        r.raise_for_status()


async def require_user(request: Request) -> str:
    sid = (request.cookies.get(AUTH_SESSION_COOKIE) or "").strip()
    if not sid:
        raise HTTPException(status_code=401, detail="Sign-in required")

    row = await auth_get_session(sid)
    if not row:
        raise HTTPException(status_code=401, detail="Invalid session")

    user_id, expires_at = str(row[0]), int(row[1])
    if time.time() >= expires_at:
        raise HTTPException(status_code=401, detail="Session expired")

    return user_id


async def enforce_quota(
    user_id: str,
    add_hf_tokens: int = 0,
    add_turso_ops: int = 1,
    add_explore: int = 0,
) -> None:
    hf_used, turso_used, explore_used = await usage_get(user_id)

    if hf_used + add_hf_tokens > DAILY_HF_TOKEN_LIMIT:
        raise HTTPException(status_code=429, detail="Daily LLM token quota exceeded")
    if turso_used + add_turso_ops > DAILY_TURSO_OP_LIMIT:
        raise HTTPException(status_code=429, detail="Daily database quota exceeded")
    if explore_used + add_explore > DAILY_EXPLORE_LIMIT:
        raise HTTPException(status_code=429, detail="Daily explore quota exceeded")

    await usage_add(user_id, add_hf_tokens=add_hf_tokens, add_turso_ops=add_turso_ops, add_explore=add_explore)


def _estimate_tokens(text: str) -> int:
    # rough heuristic: ~4 chars/token
    n = len(text or "")
    return max(1, (n + 3) // 4)


class AuthRequestCode(BaseModel):
    email: str


class AuthVerifyCode(BaseModel):
    email: str
    code: str


@router.post("/auth/request_code")
async def auth_request_code(payload: AuthRequestCode, _: None = Depends(require_human)):
    email = (payload.email or "").strip().lower()
    if not _valid_email(email):
        raise HTTPException(status_code=400, detail="Invalid email")

    code = f"{secrets.randbelow(1000000):06d}"
    code_hash = _hash_otp(email, code)
    expires_at = int(time.time()) + AUTH_OTP_TTL_SECONDS

    await auth_upsert_otp(email, code_hash, expires_at)
    await _send_email_otp(email, code)

    return {"ok": True}


@router.post("/auth/verify_code")
async def auth_verify_code(payload: AuthVerifyCode, request: Request, _: None = Depends(require_human)):
    email = (payload.email or "").strip().lower()
    code = (payload.code or "").strip()

    if not _valid_email(email):
        raise HTTPException(status_code=400, detail="Invalid email")
    if (not code.isdigit()) or len(code) != 6:
        raise HTTPException(status_code=400, detail="Invalid code")

    row = await auth_get_otp(email)
    if not row:
        raise HTTPException(status_code=401, detail="Code not found")

    code_hash_db, expires_at, attempts = str(row[0]), int(row[1]), int(row[2] or 0)

    if time.time() >= expires_at:
        await auth_delete_otp(email)
        raise HTTPException(status_code=401, detail="Code expired")

    if attempts >= 8:
        await auth_delete_otp(email)
        raise HTTPException(status_code=429, detail="Too many attempts")

    if not hmac.compare_digest(code_hash_db, _hash_otp(email, code)):
        await auth_inc_otp_attempts(email)
        raise HTTPException(status_code=401, detail="Incorrect code")

    # success
    await auth_delete_otp(email)

    user = await auth_get_user_by_email(email)
    if user:
        user_id = str(user[0])
    else:
        user_id = await auth_create_user(email)

    await auth_touch_last_login(user_id)

    session_id = str(uuid.uuid4())
    sess_exp = int(time.time()) + AUTH_SESSION_TTL_SECONDS
    await auth_create_session(session_id, user_id, sess_exp)

    resp = JSONResponse({"ok": True})

    is_https = (request.url.scheme or "").lower() == "https"
    secure_flag = bool(getattr(settings, "COOKIE_SECURE", False)) or is_https

    resp.set_cookie(
        AUTH_SESSION_COOKIE,
        session_id,
        max_age=AUTH_SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=secure_flag,
        path="/",
    )
    return resp


@router.get("/auth/me")
async def auth_me(user_id: str = Depends(require_user), _: None = Depends(require_human)):
    # tiny quota tick (DB)
    await enforce_quota(user_id, add_turso_ops=1)
    hf_used, turso_used, explore_used = await usage_get(user_id)
    return {
        "ok": True,
        "user_id": user_id,
        "usage": {
            "hf_tokens_used": hf_used,
            "hf_tokens_limit": DAILY_HF_TOKEN_LIMIT,
            "turso_ops_used": turso_used,
            "turso_ops_limit": DAILY_TURSO_OP_LIMIT,
            "explore_used": explore_used,
            "explore_limit": DAILY_EXPLORE_LIMIT,
        },
    }


@router.post("/auth/logout")
async def auth_logout(request: Request, _: None = Depends(require_human)):
    sid = (request.cookies.get(AUTH_SESSION_COOKIE) or "").strip()
    if sid:
        try:
            await auth_delete_session(sid)
        except Exception:
            pass
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(AUTH_SESSION_COOKIE, path="/")
    return resp


# ---------------------------------------------------------------------
# Hugging Face Inference API config
# ---------------------------------------------------------------------
HF_TOKEN = (getattr(settings, "HF_TOKEN", None) or os.getenv("HF_TOKEN", "")).strip()
HF_BASE_URL = "https://router.huggingface.co/v1"
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct").strip()

_HF_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=60.0, pool=10.0)
_HF_STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=60.0, pool=10.0)
_LIMITS = httpx.Limits(max_connections=10, max_keepalive_connections=5)
_TRANSPORT = httpx.AsyncHTTPTransport(http2=False)


def _clean_text(s: str) -> str:
    return "".join(ch for ch in s if ch in ("\n", "\t") or ord(ch) >= 32)


async def _llm_chat(
    messages: List[Dict[str, str]],
    max_tokens: int = 300,
    temperature: float = 0.2,
) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN missing")

    payload = {
        "model": HF_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=_HF_TIMEOUT, limits=_LIMITS, transport=_TRANSPORT) as client:
        r = await client.post(f"{HF_BASE_URL}/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        js = r.json()

    choices = js.get("choices") or []
    if choices:
        msg = (choices[0].get("message") or {}).get("content")
        if msg:
            return _clean_text(msg).strip()
    return json.dumps(js, ensure_ascii=False)


async def _llm_chat_stream(
    messages: List[Dict[str, str]],
    max_tokens: int = 300,
    temperature: float = 0.2,
) -> AsyncIterator[str]:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN missing")

    payload = {
        "model": HF_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=_HF_STREAM_TIMEOUT, limits=_LIMITS, transport=_TRANSPORT) as client:
        async with client.stream(
            "POST",
            f"{HF_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        ) as r:
            if r.status_code >= 400:
                raw = await r.aread()
                print("[HF 400 BODY]", raw.decode("utf-8", errors="replace"))
            r.raise_for_status()

            async for line in r.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: ") :].strip()
                if data == "[DONE]":
                    break
                try:
                    js = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = js.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {}).get("content")
                if not delta:
                    continue
                chunk = _clean_text(delta)
                if chunk:
                    yield chunk


# ---------------------------------------------------------------------
# Helpers: chunking abstracts
# ---------------------------------------------------------------------
def _sentence_split(text: str) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [s for s in sents if s]


def _chunk_text_sentence_safe(txt: str, max_len: int = 800) -> List[str]:
    sentences = _sentence_split(txt)
    chunks: List[str] = []
    current = ""
    for s in sentences:
        if len(current) + len(s) + 1 > max_len:
            if current:
                chunks.append(current.strip())
            current = s
        else:
            current += (" " + s) if current else s
    if current:
        chunks.append(current.strip())
    return chunks


def _make_corpus(doc: Dict[str, Any]) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if doc.get("title"):
        items.append({"id": "title", "text": doc["title"]})
    if doc.get("abstract"):
        for k, ch in enumerate(_chunk_text_sentence_safe(doc["abstract"])):
            items.append({"id": f"abs_{k}", "text": ch})
    return items


# ---------------------------------------------------------------------
# Plants DB
# ---------------------------------------------------------------------
PLANTS_DB = load_plants(Path(__file__).resolve().parent.parent / "data" / "seed_plants.csv")


# ---------------------------------------------------------------------
# Condition correction via Turso cache (your existing logic)
# ---------------------------------------------------------------------
def normalize_condition(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


SQL_LOOKUP_MISSPELLING = """
SELECT condition
FROM condition_misspellings
WHERE misspelling = ?
LIMIT 1;
"""

SQL_UPSERT_CONDITION = """
INSERT INTO condition(condition, searched)
VALUES (?, 1)
ON CONFLICT(condition) DO UPDATE
SET searched = searched + 1;
"""

SQL_UPSERT_MISSPELLING = """
INSERT INTO condition_misspellings(misspelling, condition, misspelled_searched)
VALUES (?, ?, 1)
ON CONFLICT(misspelling) DO UPDATE
SET condition = excluded.condition,
    misspelled_searched = misspelled_searched + 1;
"""


async def _expand_condition_with_llm(cond: str) -> str:
    cond = (cond or "").strip()
    if not cond:
        return ""

    fallback_system = (
        "You are a medical terminology assistant.\n"
        "Your ONLY job is to correct spelling mistakes in a disease or condition name or to replace abbreviations with their full forms\n"
        "\n"
        "You MUST ALWAYS answer with EXACTLY ONE LINE in this format:\n"
        "CORRECTED: <best standard condition name in English>\n"
        "\n"
        "RULES:\n"
        "- The line MUST start with 'CORRECTED: '.\n"
        "- Do NOT add any other text.\n"
        "- Do NOT add explanations.\n"
        "- Do NOT add other lines.\n"
    )

    fallback_user = (
        f"User condition: {cond}\n\n"
        "Correct spelling mistakes in the User condition and RETURN EXACTLY ONE LINE:\n"
        "CORRECTED: <best condition name>"
    )

    messages_txt = [
        {"role": "system", "content": fallback_system},
        {"role": "user", "content": fallback_user},
    ]

    corrected = cond
    raw = await _llm_chat(messages_txt, max_tokens=64, temperature=0.0)
    raw = _clean_text(raw).strip()

    for line in raw.splitlines():
        ls = line.strip()
        if ls.upper().startswith("CORRECTED:"):
            value = ls[len("CORRECTED:") :].strip()
            if value:
                corrected = value
                break

    return corrected


async def resolve_condition_db_first(raw_user_input: str) -> Tuple[str, str, bool]:
    raw_norm = normalize_condition(raw_user_input)
    if not raw_norm:
        return "", "", False

    used_llm = False
    corrected_norm = ""

    # 1) DB cache lookup
    try:
        row = await db_fetchone(SQL_LOOKUP_MISSPELLING, (raw_norm,))
    except Exception as e:
        print(f"[cond-db] lookup failed: {type(e).__name__}: {e}")
        row = None

    if row and row[0]:
        corrected_norm = normalize_condition(row[0])
    else:
        # 2) LLM fallback
        try:
            corrected = await _expand_condition_with_llm(raw_norm)
            corrected_norm = normalize_condition(corrected) or raw_norm
            used_llm = True
        except Exception as e:
            print(f"[cond-llm] failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            corrected_norm = raw_norm
            used_llm = False

    # 3) increment corrected condition counter
    try:
        await db_execute(SQL_UPSERT_CONDITION, (corrected_norm,))
    except Exception as e:
        print(f"[cond-db] upsert condition failed: {type(e).__name__}: {e}")

    # 4) cache mapping
    try:
        await db_execute(SQL_UPSERT_MISSPELLING, (raw_norm, corrected_norm))
    except Exception as e:
        print(f"[cond-db] upsert misspelling failed: {type(e).__name__}: {e}")

    return corrected_norm, corrected_norm, used_llm


# ---------------------------------------------------------------------
# Routes: now require both human + user session
# ---------------------------------------------------------------------
@router.get("/condition_query")
async def condition_query(
    condition: str = Query(..., min_length=2),
    _: None = Depends(require_human),
    user_id: str = Depends(require_user),
):
    # baseline turso quota tick
    await enforce_quota(user_id, add_turso_ops=2)

    condition = (condition or "").strip()
    if not condition:
        raise HTTPException(status_code=400, detail="Empty condition")

    corrected = condition
    search_query = condition
    used_llm = False

    try:
        corrected, search_query, used_llm = await resolve_condition_db_first(condition)
        if not corrected:
            corrected = condition
            search_query = condition
    except Exception as e:
        print(f"[cond_api] resolve_condition_db_first error: {type(e).__name__}: {e}")
        traceback.print_exc()

    # If LLM was used, charge HF tokens (approx)
    if used_llm:
        est = 64 + _estimate_tokens(condition)
        await enforce_quota(user_id, add_hf_tokens=est, add_turso_ops=1)

    medline_html = ""
    try:
        medline_html = await get_medlineplus_fullsummary(corrected)
    except TypeError:
        try:
            medline_html = get_medlineplus_fullsummary(corrected)
        except Exception:
            medline_html = ""
    except Exception:
        medline_html = ""

    return {
        "condition": condition,
        "corrected": corrected,
        "search_query": search_query,
        "medline_html": medline_html,
    }


@router.get("/recommendations")
async def recommendations(
    condition: str = Query(..., min_length=2),
    from_year: Optional[int] = Query(None, ge=1800, le=3000),
    to_year: Optional[int] = Query(None, ge=1800, le=3000),
    llm_query: Optional[str] = Query(None),
    _: None = Depends(require_human),
    user_id: str = Depends(require_user),
):
    # minimal quota tick (this endpoint does not call HF in your design)
    await enforce_quota(user_id, add_turso_ops=1)

    condition = (condition or "").strip()
    if not condition:
        raise HTTPException(status_code=400, detail="Empty condition")

    now_year = datetime.utcnow().year
    to_year = to_year or now_year
    from_year = from_year or 2024

    if from_year > to_year:
        from_year, to_year = to_year, from_year

    search_query = (llm_query or condition).strip() or condition

    try:
        articles = await search_and_fetch(search_query, str(from_year), str(to_year))
    except Exception as e:
        print(f"[reco] search_and_fetch error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"condition": condition, "search_query": search_query, "results": [], "error": "PubMed upstream error"}

    plant_groups: Dict[str, Dict[str, Any]] = {}

    for a in articles:
        text = f"{a.get('title','')} {a.get('abstract','')}"
        plants_found = list(find_plants_in_text(text, PLANTS_DB))
        plants_unique = sorted(set(plants_found))
        if not plants_unique:
            continue

        group_label = plants_unique[0] if len(plants_unique) == 1 else ", ".join(plants_unique)

        grp = plant_groups.get(group_label)
        if not grp:
            grp = {"plants": plants_unique, "articles": []}
            plant_groups[group_label] = grp
        grp["articles"].append(a)

    results: List[Dict[str, Any]] = []
    for label, grp in plant_groups.items():
        arts = grp["articles"]
        scored = sorted(arts, key=score_article, reverse=True)
        plant_score = sum(score_article(a) for a in scored[:5])
        summary = summarize_for_patients(label, scored)

        items = [
            {"pmid": a.get("pmid", ""), "title": a.get("title", ""), "year": a.get("year", ""), "journal": a.get("journal", "")}
            for a in scored[:5]
        ]

        results.append({"plant": label, "score": round(plant_score, 2), "summary": summary, "top_studies": items})

    results.sort(key=lambda x: x["score"], reverse=True)
    return {"condition": condition, "search_query": search_query, "results": results}


@router.get("/explore_stream")
async def explore_stream(
    pmid: str = Query(..., min_length=1),
    plant: Optional[str] = Query(None, min_length=1),
    _: None = Depends(require_human),
    user_id: str = Depends(require_user),
):
    # baseline DB tick
    await enforce_quota(user_id, add_turso_ops=1)

    pub_id = f"pub_{pmid}"

    # 1) Turso cache
    try:
        cached = await get_resume_by_pub_id(pub_id)
    except Exception as e:
        print(f"[turso] cache read failed: {type(e).__name__}: {e}")
        cached = None

    if cached:
        try:
            await increment_searched(pub_id)
        except Exception:
            pass

        async def gen_cached():
            yield cached
            yield f"\n\nReferences:\nPubMed: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        return StreamingResponse(gen_cached(), media_type="text/plain")

    # cache miss => will call HF => enforce explore + token budget now
    # (conservative estimate)
    await enforce_quota(user_id, add_explore=1, add_turso_ops=1)

    # 2) efetch + LLM stream
    async with httpx.AsyncClient(
        timeout=_HF_STREAM_TIMEOUT,
        limits=_LIMITS,
        transport=_TRANSPORT,
        trust_env=False,
    ) as http:
        arts = await efetch(http, [pmid])

    if not arts:
        async def gen_empty():
            yield "No abstract found.\n"
            yield f"References:\nPubMed: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        return StreamingResponse(gen_empty(), media_type="text/plain")

    a = arts[0]
    doc = {
        "pmid": pmid,
        "title": a.get("title", ""),
        "abstract": a.get("abstract", ""),
        "journal": a.get("journal", ""),
        "year": a.get("year", ""),
    }

    chunks = _make_corpus(doc)
    if not chunks:
        async def gen_no_abs():
            yield "No abstract text available.\n"
            yield f"References:\nPubMed: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        return StreamingResponse(gen_no_abs(), media_type="text/plain")

    # Build context and token estimate
    max_chars = 6000
    parts: List[str] = []
    if doc["title"]:
        parts.append(f"TITLE: {doc['title']}")
    abs_chunks = [c["text"] for c in chunks if c["id"].startswith("abs_")]
    for ch in abs_chunks[:6]:
        parts.append(ch)
    context = "\n".join(parts)[:max_chars]

    # charge HF tokens estimate once before streaming
    est_hf = _estimate_tokens(context) + 300  # output cap
    await enforce_quota(user_id, add_hf_tokens=est_hf, add_turso_ops=1)

    raw_plant = (plant or "").strip()
    if raw_plant:
        parts_pl = [p.strip() for p in raw_plant.split(",") if p.strip()]
        if len(parts_pl) == 1:
            plant_for_prompt = parts_pl[0]
        elif len(parts_pl) == 2:
            plant_for_prompt = " and ".join(parts_pl)
        else:
            plant_for_prompt = ", ".join(parts_pl[:-1]) + " and " + parts_pl[-1]
    else:
        plant_for_prompt = "all plants mentioned in the CONTEXT"

    _SYSTEM = (
        "You are a health science communicator for the general as related public.\n"
        "Your goal is to provide information from a scientific article by:\n"
        "- following the user instructions exactly.\n"
        "- relating KEY FINDINGS as described in the user CONTEXT section.\n"
        "\n"
        "PRIORITY RULES:\n"
        "1. Do NOT copy codes/identifiers.\n"
        "2. Use simple everyday words only.\n"
    )

    _USER_TMPL = (
        "Study: {title} — {year} / {journal}\n\n"
        "CONTEXT:\n{context}\n\n"
        "Write ONE paragraph of 4–6 sentences.\n"
        "Focus on {plant}.\n"
        "Return ONLY the paragraph."
    )

    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": _USER_TMPL.format(
                title=doc["title"],
                year=doc["year"],
                journal=doc["journal"],
                context=context,
                plant=plant_for_prompt,
            ),
        },
    ]

    async def event_generator():
        buf_parts: List[str] = []
        try:
            async for chunk in _llm_chat_stream(messages, max_tokens=300, temperature=0.0):
                buf_parts.append(chunk)
                yield chunk
        except Exception as e:
            err = f"\n[ERROR] LLM call failed: {type(e).__name__}: {e}\n"
            print(err)
            traceback.print_exc()
            yield err
            return

        refs = f"\n\nReferences:\nPubMed: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        yield refs

        full_resume = "".join(buf_parts).strip()
        if full_resume:
            try:
                await insert_resume(pub_id, full_resume)
            except Exception as e:
                print(f"[turso] insert failed: {type(e).__name__}: {e}")

    return StreamingResponse(event_generator(), media_type="text/plain")