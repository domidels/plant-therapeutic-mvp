"""
app/api/routes.py
-----------------
All HTTP endpoints for the Plant Therapeutic MVP.

Rate limiting strategy
~~~~~~~~~~~~~~~~~~~~~~
- Cloudflare Turnstile verifies each visitor is human (cookie valid 6 h).
- Requests are then rate-limited by hashed client IP — no account required.
- Two quota levels are enforced per UTC day:
    * Per-IP   : limits individual abuse.
    * Global   : hard ceiling on total LLM cost regardless of IP count.
- Quotas are stored in the ``usage_daily`` Turso table keyed by ``user_id``
  (either a hashed IP or the reserved key ``_global``).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
import traceback
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
    get_negative_pub_ids,
    # generic db helpers
    db_fetchone,
    db_execute,
    # usage
    usage_get,
    usage_add,
    # tracking
    upsert_user_session,
    insert_user_search,
    increment_invalid_query,
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
    """Return ``payload.HMAC`` so the cookie value cannot be forged."""
    sig = hmac.new(SESSION_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def _verify_signed(value: str) -> Optional[str]:
    """Verify a signed cookie value and return the payload, or None if invalid."""
    try:
        payload, sig = value.rsplit(".", 1)
        expected = hmac.new(SESSION_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()
        if hmac.compare_digest(sig, expected):
            return payload
    except Exception:
        return None
    return None


def _is_human_cookie_valid(cookie_val: str) -> bool:
    """Return True when the Turnstile cookie is present, correctly signed, and not expired."""
    payload = _verify_signed(cookie_val)
    if not payload:
        return False
    try:
        exp = int(payload)
    except ValueError:
        return False
    return time.time() < exp


async def require_human(request: Request) -> None:
    """FastAPI dependency — reject the request with 401 if the Turnstile cookie is missing or expired."""
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

    # Track the visit (fire-and-forget — never block the response on failure)
    try:
        user_id = _ip_to_id(_get_client_ip(request))
        await upsert_user_session(user_id)
    except Exception:
        pass

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
# IP-based rate limiting (replaces email OTP auth)
# ---------------------------------------------------------------------

# Quotas per IP per UTC day
DAILY_HF_TOKEN_LIMIT = int(os.getenv("DAILY_HF_TOKEN_LIMIT", "4000"))
DAILY_TURSO_OP_LIMIT = int(os.getenv("DAILY_TURSO_OP_LIMIT", "5000"))
DAILY_EXPLORE_LIMIT = int(os.getenv("DAILY_EXPLORE_LIMIT", "25"))

# Global quotas (all IPs combined) — hard ceiling to cap total daily cost
DAILY_GLOBAL_HF_TOKEN_LIMIT = int(os.getenv("DAILY_GLOBAL_HF_TOKEN_LIMIT", "50000"))
DAILY_GLOBAL_EXPLORE_LIMIT = int(os.getenv("DAILY_GLOBAL_EXPLORE_LIMIT", "150"))
_GLOBAL_USER_ID = "_global"


def _get_client_ip(request: Request) -> str:
    """Return the real client IP, handling proxies (Vercel, etc.)."""
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _ip_to_id(ip: str) -> str:
    """Hash the IP with HMAC-SHA256 so raw IPs are never stored."""
    return hmac.new(SESSION_SECRET, ip.encode("utf-8"), hashlib.sha256).hexdigest()


async def get_ip_user_id(request: Request) -> str:
    """Return the HMAC-hashed IP to use as ``user_id`` in quota tables."""
    return _ip_to_id(_get_client_ip(request))


async def enforce_global_quota(add_hf_tokens: int = 0, add_explore: int = 0) -> None:
    """
    Check and increment the service-wide daily quota (key ``_global``).

    Raises HTTP 429 when the combined usage of all IPs would exceed
    ``DAILY_GLOBAL_HF_TOKEN_LIMIT`` or ``DAILY_GLOBAL_EXPLORE_LIMIT``.
    This is the last line of defence against proxy-based quota exhaustion.
    """
    hf_used, _, explore_used = await usage_get(_GLOBAL_USER_ID)

    if hf_used + add_hf_tokens > DAILY_GLOBAL_HF_TOKEN_LIMIT:
        raise HTTPException(status_code=429, detail="The service has reached its daily limit. Come back tomorrow — quotas reset at midnight UTC.")
    if explore_used + add_explore > DAILY_GLOBAL_EXPLORE_LIMIT:
        raise HTTPException(status_code=429, detail="The service has reached its daily exploration limit. Come back tomorrow — quotas reset at midnight UTC.")

    await usage_add(_GLOBAL_USER_ID, add_hf_tokens=add_hf_tokens, add_explore=add_explore)


async def enforce_quota(
    user_id: str,
    add_hf_tokens: int = 0,
    add_turso_ops: int = 1,
    add_explore: int = 0,
) -> None:
    """
    Check and increment the per-IP daily quota.

    All three counters (HF tokens, Turso operations, explore calls) are
    checked atomically before being incremented, so a single over-limit
    call never silently consumes quota.
    """
    hf_used, turso_used, explore_used = await usage_get(user_id)

    if hf_used + add_hf_tokens > DAILY_HF_TOKEN_LIMIT:
        raise HTTPException(status_code=429, detail="Your quota for today is reached. Come back tomorrow — quotas reset at midnight UTC.")
    if turso_used + add_turso_ops > DAILY_TURSO_OP_LIMIT:
        raise HTTPException(status_code=429, detail="Your quota for today is reached. Come back tomorrow — quotas reset at midnight UTC.")
    if explore_used + add_explore > DAILY_EXPLORE_LIMIT:
        raise HTTPException(status_code=429, detail="Your daily exploration quota is reached. Come back tomorrow — quotas reset at midnight UTC.")

    await usage_add(user_id, add_hf_tokens=add_hf_tokens, add_turso_ops=add_turso_ops, add_explore=add_explore)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token (GPT-style heuristic)."""
    n = len(text or "")
    return max(1, (n + 3) // 4)


@router.get("/auth/me")
async def auth_me(request: Request, _: None = Depends(require_human)):
    user_id = await get_ip_user_id(request)
    hf_used, turso_used, explore_used = await usage_get(user_id)
    return {
        "ok": True,
        "usage": {
            "hf_tokens_used": hf_used,
            "hf_tokens_limit": DAILY_HF_TOKEN_LIMIT,
            "turso_ops_used": turso_used,
            "turso_ops_limit": DAILY_TURSO_OP_LIMIT,
            "explore_used": explore_used,
            "explore_limit": DAILY_EXPLORE_LIMIT,
        },
    }


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
    """Strip non-printable characters from LLM output, keeping newlines and tabs."""
    return "".join(ch for ch in s if ch in ("\n", "\t") or ord(ch) >= 32)


async def _llm_chat(
    messages: List[Dict[str, str]],
    max_tokens: int = 300,
    temperature: float = 0.2,
) -> str:
    """Send a blocking chat-completion request to Hugging Face and return the text response."""
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
    """Stream chat-completion tokens from Hugging Face, yielding each text chunk as it arrives."""
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
    """Split text into sentences on sentence-ending punctuation."""
    sents = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [s for s in sents if s]


def _chunk_text_sentence_safe(txt: str, max_len: int = 800) -> List[str]:
    """Split text into chunks of at most ``max_len`` characters without breaking sentences."""
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
    """Build a list of labelled text chunks (title + abstract) from a PubMed article dict."""
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
# Condition correction via Turso cache then LLM fallback
# ---------------------------------------------------------------------
def normalize_condition(s: str) -> str:
    """Lowercase and collapse whitespace in a condition string for consistent cache lookups."""
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


async def _expand_condition_with_llm(cond: str) -> Tuple[str, List[str], bool]:
    """
    Use the LLM to correct spelling, return synonyms, and validate that the
    input is a recognisable medical condition.

    Returns ``(corrected, synonyms, is_condition)`` where ``is_condition`` is
    False when the input cannot be interpreted as a disease, symptom, or syndrome.
    Falls back to ``(original, [], True)`` on LLM failure (fail open).
    """
    cond = (cond or "").strip()
    if not cond:
        return "", [], False

    fallback_system = (
        "You are a medical terminology assistant.\n"
        "Your job is to interpret a user-entered condition name and return THREE lines.\n"
        "\n"
        "Line 1 — IS_CONDITION: YES or NO\n"
        "  Write YES if the input is a recognisable disease, symptom, syndrome, or medical condition.\n"
        "  Write NO if the input is a random word, a plant name, a food, a number, a question, or anything else.\n"
        "\n"
        "Line 2 — CORRECTED: <standard medical term in English>\n"
        "  Fix spelling and expand abbreviations. If IS_CONDITION is NO, write CORRECTED: N/A\n"
        "\n"
        "Line 3 — SYNONYMS: <synonym1>, <synonym2>, <synonym3>\n"
        "  Up to 3 EXACT synonyms used in clinical literature for the SAME condition.\n"
        "  Example: 'urticaria' → hives, urticaria chronica, nettle rash\n"
        "  Example: 'HTN' → high blood pressure, arterial hypertension\n"
        "  Do NOT add broader categories or parent diseases.\n"
        "  If no useful synonyms exist, or if IS_CONDITION is NO, write SYNONYMS: (empty)\n"
        "\n"
        "STRICT RULES:\n"
        "- Do NOT add any other text or explanation.\n"
        "- Answer with EXACTLY THREE LINES.\n"
    )

    fallback_user = (
        f"User input: {cond}\n\n"
        "IS_CONDITION: YES or NO\n"
        "CORRECTED: <standard medical term>\n"
        "SYNONYMS: <synonym1>, <synonym2>, ..."
    )

    messages_txt = [
        {"role": "system", "content": fallback_system},
        {"role": "user", "content": fallback_user},
    ]

    corrected = cond
    synonyms: List[str] = []
    is_condition = True  # fail open
    raw = await _llm_chat(messages_txt, max_tokens=150, temperature=0.0)
    raw = _clean_text(raw).strip()

    for line in raw.splitlines():
        ls = line.strip()
        if ls.upper().startswith("IS_CONDITION:"):
            value = ls[len("IS_CONDITION:"):].strip().upper()
            is_condition = value != "NO"
        elif ls.upper().startswith("CORRECTED:"):
            value = ls[len("CORRECTED:"):].strip()
            if value and value.upper() not in ("N/A", "NA"):
                corrected = value
        elif ls.upper().startswith("SYNONYMS:"):
            value = ls[len("SYNONYMS:"):].strip()
            if value and value.lower() != "(empty)":
                synonyms = [s.strip() for s in value.split(",") if s.strip()]

    return corrected, synonyms, is_condition


async def resolve_condition_db_first(raw_user_input: str) -> Tuple[str, str, bool]:
    """
    Resolve a raw condition string to its corrected canonical form.

    Strategy (cache-first to minimise LLM calls):
      1. Look up the normalised input in ``condition_misspellings``.
      2. On cache miss, call the LLM and persist the mapping.
      3. Increment the search counter for the corrected condition.

    Returns:
        (corrected, search_query, used_llm)
    """
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

    synonyms: List[str] = []
    is_condition = True

    if row and row[0]:
        # Condition already known — use cached correction, still fetch synonyms via LLM
        corrected_norm = normalize_condition(row[0])
        try:
            _, synonyms, is_condition = await _expand_condition_with_llm(corrected_norm)
            used_llm = True
        except Exception:
            synonyms = []
    else:
        # 2) LLM fallback for both correction and synonyms
        try:
            corrected, synonyms, is_condition = await _expand_condition_with_llm(raw_norm)
            corrected_norm = normalize_condition(corrected) or raw_norm
            used_llm = True
        except Exception as e:
            print(f"[cond-llm] failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            corrected_norm = raw_norm
            used_llm = False

    # Skip DB writes if input is not a medical condition
    if not is_condition:
        return corrected_norm, corrected_norm, used_llm, synonyms, is_condition

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

    return corrected_norm, corrected_norm, used_llm, synonyms, is_condition


# ---------------------------------------------------------------------
# Routes: now require both human + user session
# ---------------------------------------------------------------------
@router.get("/condition_query")
async def condition_query(
    condition: str = Query(..., min_length=2, max_length=100),
    request: Request = None,
    _: None = Depends(require_human),
):
    user_id = await get_ip_user_id(request)
    # baseline turso quota tick
    await enforce_quota(user_id, add_turso_ops=2)

    condition = (condition or "").strip()
    if not condition:
        raise HTTPException(status_code=400, detail="Empty condition")

    corrected = condition
    search_query = condition
    used_llm = False
    synonyms: List[str] = []
    is_condition = True

    try:
        corrected, search_query, used_llm, synonyms, is_condition = await resolve_condition_db_first(condition)
        if not corrected:
            corrected = condition
            search_query = condition
    except Exception as e:
        print(f"[cond_api] resolve_condition_db_first error: {type(e).__name__}: {e}")
        traceback.print_exc()

    if not is_condition:
        try:
            invalid_count = await increment_invalid_query(user_id)
        except Exception:
            invalid_count = 0
        if invalid_count >= 10:
            raise HTTPException(
                status_code=429,
                detail="Too many invalid requests today. Access is blocked until midnight UTC.",
            )
        return JSONResponse(
            content={
                "error": "no_condition",
                "message": (
                    "No medical condition could be deduced from your input. "
                    "Please enter a disease, symptom, or medical syndrome "
                    "(e.g. eczema, type 2 diabetes, anxiety)."
                ),
            }
        )

    # Record the search (fire-and-forget — never block the response on failure)
    try:
        await insert_user_search(user_id, corrected or condition)
    except Exception:
        pass

    # If LLM was used, charge HF tokens (approx)
    if used_llm:
        est = 128 + _estimate_tokens(condition)
        await enforce_global_quota(add_hf_tokens=est)
        await enforce_quota(user_id, add_hf_tokens=est, add_turso_ops=1)

    # Build expanded search query including synonyms (max 3 extra terms)
    all_terms = [corrected] + [s for s in synonyms[:3] if s.lower() != corrected.lower()]
    search_query = " OR ".join(all_terms)

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
        "synonyms": synonyms,
        "search_query": search_query,
        "medline_html": medline_html,
    }


@router.get("/recommendations")
async def recommendations(
    condition: str = Query(..., min_length=2),
    from_year: Optional[int] = Query(None, ge=1800, le=3000),
    to_year: Optional[int] = Query(None, ge=1800, le=3000),
    llm_query: Optional[str] = Query(None),
    request: Request = None,
    _: None = Depends(require_human),
):
    user_id = await get_ip_user_id(request)
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
        # Skip articles whose parsed year falls outside the requested range.
        # PubMed sometimes returns articles with cross-year dates (e.g. accepted 2025,
        # published online 2026), causing duplicates across year sections.
        art_year = a.get("year", "")
        if art_year:
            try:
                y = int(art_year)
                if y < from_year or y > to_year:
                    continue
            except ValueError:
                pass

        text = f"{a.get('title','')} {a.get('abstract','')}"
        plants_found, negative_found = find_plants_in_text(text, PLANTS_DB)
        plants_unique = sorted(set(plants_found))
        if not plants_unique:
            continue

        group_label = plants_unique[0] if len(plants_unique) == 1 else ", ".join(plants_unique)

        grp = plant_groups.get(group_label)
        if not grp:
            grp = {"plants": plants_unique, "articles": [], "keyword_negative": set()}
            plant_groups[group_label] = grp
        grp["articles"].append(a)
        grp["keyword_negative"].update(negative_found)

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

        keyword_neg = bool(grp["keyword_negative"])
        results.append({"plant": label, "score": round(plant_score, 2), "summary": summary, "top_studies": items, "keyword_negative": keyword_neg})

    results.sort(key=lambda x: x["score"], reverse=True)

    # Enrich results with cached verdict — single batch query
    all_pub_ids = [f"pub_{s['pmid']}" for r in results for s in r["top_studies"] if s.get("pmid")]
    try:
        negative_ids = await get_negative_pub_ids(all_pub_ids)
    except Exception:
        negative_ids = set()
    for r in results:
        llm_negative = any(f"pub_{s['pmid']}" in negative_ids for s in r["top_studies"])
        r["has_negative"] = llm_negative or r.pop("keyword_negative", False)

    return {"condition": condition, "search_query": search_query, "results": results}


@router.get("/explore_stream")
async def explore_stream(
    pmid: str = Query(..., min_length=1),
    plant: Optional[str] = Query(None, min_length=1),
    request: Request = None,
    _: None = Depends(require_human),
):
    user_id = await get_ip_user_id(request)
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
        cached_text, cached_verdict = cached
        try:
            await increment_searched(pub_id)
        except Exception:
            pass

        async def gen_cached():
            yield cached_text
            yield f"\n\nReferences:\nPubMed: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            verdict_label = "NEGATIVE" if cached_verdict == 1 else "POSITIVE"
            yield f"\n__VERDICT:{verdict_label}__"

        return StreamingResponse(gen_cached(), media_type="text/plain")

    # cache miss => will call HF => enforce explore + token budget now
    # (conservative estimate)
    await enforce_global_quota(add_explore=1)
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
    await enforce_global_quota(add_hf_tokens=est_hf)
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
        "You are a health science communicator writing for the general public.\n"
        "Your goal is to summarise key findings from a scientific article in plain, accessible language.\n"
        "\n"
        "RULES:\n"
        "1. Use simple, everyday words — avoid medical jargon or explain it when unavoidable.\n"
        "2. Do NOT copy article identifiers, DOIs, or numeric codes.\n"
        "3. Be factual and neutral — do not overstate preliminary or observational findings.\n"
        "4. This is information only, not medical advice.\n"
    )

    _USER_TMPL = (
        "Study: {title} — {year} / {journal}\n\n"
        "CONTEXT:\n{context}\n\n"
        "Write ONE paragraph of 4–6 sentences that:\n"
        "- States what the study examined and how (mention the study type if stated: RCT, meta-analysis, etc.).\n"
        "- Explains specifically how {plant} acts on or affects the condition (mechanism, effect, outcome).\n"
        "- Quantifies the effect if the CONTEXT provides numbers (e.g. dosage, percentage improvement).\n"
        "- Mentions any limitations or caveats if stated in the CONTEXT.\n\n"
        "Return ONLY the paragraph, with no title, no bullet points, and no disclaimer.\n\n"
        "Then, on the very last line, write exactly one of:\n"
        "VERDICT:NEGATIVE  (if {plant} has an adverse, harmful or contraindicated effect in this context, "
        "or if the condition appears only as a side effect / adverse event of the treatment)\n"
        "VERDICT:POSITIVE  (otherwise)"
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
            async for chunk in _llm_chat_stream(messages, max_tokens=320, temperature=0.0):
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

        # Parse and strip the VERDICT line the LLM appended
        import re as _re
        verdict = "POSITIVE"
        clean_resume = full_resume
        verdict_match = _re.search(r"\nVERDICT:(NEGATIVE|POSITIVE)", full_resume)
        if verdict_match:
            verdict = verdict_match.group(1)
            clean_resume = full_resume[: verdict_match.start()].strip()

        verdict_int = 1 if verdict == "NEGATIVE" else 0
        yield f"\n__VERDICT:{verdict}__"

        if clean_resume:
            try:
                await insert_resume(pub_id, clean_resume, verdict_int)
            except Exception as e:
                print(f"[turso] insert failed: {type(e).__name__}: {e}")

    return StreamingResponse(event_generator(), media_type="text/plain")