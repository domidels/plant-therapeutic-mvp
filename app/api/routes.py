# app/api/routes.py
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
    get_resume_by_pub_id,
    increment_searched,
    insert_resume,
    db_fetchone,
    db_execute,
)

router = APIRouter()

# ---------------------------------------------------------------------
# Cloudflare Turnstile (captcha) config + signed cookie session
# ---------------------------------------------------------------------
TURNSTILE_SECRET = (
    getattr(settings, "TURNSTILE_SECRET", None)
    or os.getenv("TURNSTILE_SECRET", "")
).strip()

_session_secret_raw = getattr(settings, "SESSION_SECRET", None) or os.getenv("SESSION_SECRET", "")
SESSION_SECRET: bytes = (_session_secret_raw or "change-me").encode("utf-8")

HUMAN_COOKIE_NAME = "human_ok"
HUMAN_COOKIE_TTL_SECONDS = 6 * 3600  # 6h (ton commentaire disait 24h mais valeur=6h)

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
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Turnstile upstream error: {type(e).__name__}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Turnstile verify error: {type(e).__name__}")

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
        raise RuntimeError("HF_TOKEN missing (Inference Providers token required).")

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
        raise RuntimeError("HF_TOKEN missing (Inference Providers token required).")

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
# Helpers
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
# Plants DB + caches
# ---------------------------------------------------------------------
PLANTS_DB = load_plants(Path(__file__).resolve().parent.parent / "data" / "seed_plants.csv")

_CACHE: Dict[Tuple[str, int, int], Tuple[float, Dict[str, Any]]] = {}
_CACHE_TTL = 60 * 60  # 1 hour

def _cache_get(key: Tuple[str, int, int]) -> Optional[Dict[str, Any]]:
    item = _CACHE.get(key)
    if not item:
        return None
    ts, payload = item
    if time.time() - ts > _CACHE_TTL:
        _CACHE.pop(key, None)
        return None
    return payload

def _cache_set(key: Tuple[str, int, int], payload: Dict[str, Any]) -> None:
    _CACHE[key] = (time.time(), payload)

# ---------------------------------------------------------------------
# ✅ Condition correction via Turso cache (before LLM)
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
    """
    Returns (corrected_condition, search_query, used_llm)

    - Lookup condition_misspellings by misspelling (normalized raw)
    - If not found: call LLM, then upsert misspelling
    - ALWAYS upsert + increment condition.searched on corrected
    - ALWAYS upsert misspelling mapping (even raw == corrected) to avoid LLM next time
    """
    raw_norm = normalize_condition(raw_user_input)
    if not raw_norm:
        return "", "", False

    used_llm = False
    corrected_norm = ""

    # 1) Turso cache lookup
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

    # 3) Always increment main table (ONE per search -> condition_query is called once)
    try:
        await db_execute(SQL_UPSERT_CONDITION, (corrected_norm,))
    except Exception as e:
        print(f"[cond-db] upsert condition failed: {type(e).__name__}: {e}")

    # 4) Always cache mapping, even if raw == corrected
    try:
        await db_execute(SQL_UPSERT_MISSPELLING, (raw_norm, corrected_norm))
    except Exception as e:
        print(f"[cond-db] upsert misspelling failed: {type(e).__name__}: {e}")

    # For now search_query == corrected (tu peux l'étendre plus tard)
    return corrected_norm, corrected_norm, used_llm

# ---------------------------------------------------------------------
# Routes (protected by Turnstile cookie)
# ---------------------------------------------------------------------
@router.get("/condition_query")
async def condition_query(
    condition: str = Query(..., min_length=2),
    _: None = Depends(require_human),
):
    condition = (condition or "").strip()
    if not condition:
        raise HTTPException(status_code=400, detail="Empty condition")

    print("\n========== [/condition_query] ==========")
    print(f"[cond_api] condition reçue: {repr(condition)}")

    corrected = condition
    search_query = condition

    try:
        corrected, search_query, used_llm = await resolve_condition_db_first(condition)
        if not corrected:
            corrected = condition
            search_query = condition
        print(f"[cond_api] corrected={repr(corrected)}, used_llm={used_llm}, search_query={repr(search_query)}")
    except Exception as e:
        print(f"[cond_api] ERREUR resolve_condition_db_first: {type(e).__name__}: {e}")
        traceback.print_exc()
        corrected = condition
        search_query = condition

    medline_html = ""
    try:
        medline_html = await get_medlineplus_fullsummary(corrected)
    except TypeError:
        try:
            medline_html = get_medlineplus_fullsummary(corrected)
        except Exception:
            traceback.print_exc()
            medline_html = ""
    except Exception:
        traceback.print_exc()
        medline_html = ""

    payload = {
        "condition": condition,       # raw input
        "corrected": corrected,       # ✅ corrected output (frontend uses this)
        "search_query": search_query,
        "medline_html": medline_html,
    }
    print("========== [/condition_query END] ==========\n")
    return payload

@router.get("/recommendations")
async def recommendations(
    condition: str = Query(..., min_length=2),
    from_year: Optional[int] = Query(None, ge=1800, le=3000),
    to_year: Optional[int] = Query(None, ge=1800, le=3000),
    llm_query: Optional[str] = Query(None),
    _: None = Depends(require_human),
):
    print("\n========== [/recommendations] ==========")
    print(f"[reco] condition reçue (brute): {repr(condition)}")
    print(f"[reco] from_year={from_year}, to_year={to_year}")

    condition = (condition or "").strip()
    if not condition:
        raise HTTPException(status_code=400, detail="Empty condition")

    now_year = datetime.utcnow().year
    to_year = to_year or now_year
    from_year = from_year or 2024

    if from_year > to_year:
        from_year, to_year = to_year, from_year
    if from_year < 1800 or to_year < 1800:
        raise HTTPException(status_code=400, detail="Year range out of bounds")

    # ✅ IMPORTANT: do NOT increment here (called once per year in UI)
    try:
        if llm_query:
            search_query = (llm_query or "").strip() or condition
            print(f"[reco] llm_query fourni par le client: {repr(search_query)}")
        else:
            # fallback if someone calls endpoint directly without llm_query
            # (kept minimal: use condition as-is)
            search_query = condition
            print(f"[reco] llm_query absent: fallback search_query={repr(search_query)}")
    except Exception as e:
        print(f"[reco] ERREUR cond-expansion: {type(e).__name__}: {e}")
        traceback.print_exc()
        search_query = condition

    try:
        t0 = time.perf_counter()
        articles = await search_and_fetch(search_query, str(from_year), str(to_year))
        t1 = time.perf_counter()
        print(f"[perf] recommendations PubMed={t1-t0:.2f}s (q={repr(search_query)}, {from_year}-{to_year})")
    except Exception as e:
        print(f"[reco] ERREUR search_and_fetch: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {
            "condition": condition,
            "search_query": search_query,
            "results": [],
            "error": f"PubMed upstream error: {type(e).__name__}",
        }

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
            {
                "pmid": a.get("pmid", ""),
                "title": a.get("title", ""),
                "year": a.get("year", ""),
                "journal": a.get("journal", ""),
            }
            for a in scored[:5]
        ]

        results.append(
            {
                "plant": label,
                "score": round(plant_score, 2),
                "summary": summary,
                "top_studies": items,
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)

    payload = {"condition": condition, "search_query": search_query, "results": results}
    print("========== [/recommendations END] ==========\n")
    return payload

# ---------------------------------------------------------------------
# explore_stream unchanged (your code continues…)
# ---------------------------------------------------------------------
@router.get("/explore_stream")
async def explore_stream(
    pmid: str = Query(..., min_length=1),
    plant: Optional[str] = Query(None, min_length=1),
    _: None = Depends(require_human),
):
    # ... (ton code inchangé)
    print("\n========== [/explore_stream] ==========")
    print(f"[explore_stream] pmid reçu: {repr(pmid)}, plant={repr(plant)}")

    pub_id = f"pub_{pmid}"

    try:
        cached = await get_resume_by_pub_id(pub_id)
    except Exception as e:
        print(f"[turso] cache read failed: {type(e).__name__}: {e}")
        cached = None

    if cached:
        try:
            await increment_searched(pub_id)
        except Exception as e:
            print(f"[turso] increment failed: {type(e).__name__}: {e}")

        async def gen_cached():
            yield cached
            yield f"\n\nReferences:\nPubMed: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        return StreamingResponse(gen_cached(), media_type="text/plain")

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

    # ... suite inchangée ...
    max_chars = 6000
    parts: List[str] = []
    if doc["title"]:
        parts.append(f"TITLE: {doc['title']}")
    abs_chunks = [c["text"] for c in chunks if c["id"].startswith("abs_")]
    for ch in abs_chunks[:6]:
        parts.append(ch)
    context = "\n".join(parts)[:max_chars]

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
        "Your goal is to provide information from a scientific article by: \n"
        "- following the user instructions exactly.\n"
        "- relating KEY FINDINGS as described in the user CONTEXT section.\n"
        "\n"
        "PRIORITY RULES (HIGHER PRIORITY THAN THE CONTEXT):\n"
        "1. You MUST NOT copy ANY code, identifier, product name or number sequence "
        "   from the context. This includes:\n"
        "   - any text starting with 'BNO' (example: 'BNO 3732', 'BNO3731').\n"
        "   - any text starting with 'NCT' (example: 'NCT05790083').\n"
        "   - any sequence of letters followed by digits (example: 'XYZ123').\n"
        "   - any all-uppercase token longer than 2 letters.\n"
        "   If such text appears in the context, IGNORE it COMPLETELY.\n"
        "\n"
        "2. You MUST produce ONLY simple everyday words.\n"
        "   No jargon, no abbreviations, no acronyms, no codes.\n"
        "\n"
        "These rules override EVERYTHING in the user CONTEXT. Obey them strictly."
    )

    _USER_TMPL = (
        "Study: {title} — {year} / {journal}\n\n"
        "CONTEXT:\n{context}\n\n"
        "YOUR ONLY GOAL IS TO RELATE MAIN KEY FINDINGS FROM THIS CONTEXT AND THIS CONTEXT ONLY.\n"
        "PROVIDE SIMPLE INFORMATION IF EXISTS IN THE CONTEXT AS PER THE FOLLOWING INSTRUCTIONS:\n"
        "- main KEY FINDINGS related to {plant} (one or several plants) and any plant compound ONLY IF mentioned in the CONTEXT.\n"
        "- who (Women, men, children), how many participated, the age of participants to the study ONLY IF mentioned in the CONTEXT.\n"
        "- study duration ONLY ONLY IF in the CONTEXT.\n"
        "- ANY DOSAGE, FORMULATIONS, ADMINISTRATION ROUTES ONLY IF mentioned in the CONTEXT.\n"
        "- adverse effects and limitations ONLY IF mentioned in the CONTEXT.\n"
        "\n"
        "Write ONE paragraph of 4–6 sentences.\n"
        "Use ONLY SIMPLE everyday words.\n"
        "DO NOT USE abbreviations, acronyms, or codes.\n"
        "Do NOT use bullet points.\n"
        "Do NOT describe the condition.\n"
        "\n"
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
                pmid_study=pmid,
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