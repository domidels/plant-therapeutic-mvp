# app/api/routes.py
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from fastembed import TextEmbedding

from app.services.pubmed import efetch, search_and_fetch
from app.services.plants import load_plants, find_plants_in_text
from app.services.ranking import score_article, summarize_for_patients

# ---------------------------------------------------------------------
# Router & Config
# ---------------------------------------------------------------------
router = APIRouter()

def _sanitize_base(s: str) -> str:
    """Force IPv4 si 'localhost' pour éviter une résolution ::1 qui peut timeouter."""
    s = (s or "").split("#", 1)[0].strip().split()[0].rstrip("/")
    return s.replace("://localhost", "://127.0.0.1")

# IPv4 par défaut si aucune variable d'env n'est fournie
BASE = _sanitize_base(os.getenv("BASE") or "http://127.0.0.1:11435")
MODEL = os.getenv("MODEL", "biomistral")

_HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=10.0)
# Timeout par défaut pour stream (utilisé dans les fallbacks)
_STREAM_TIMEOUT = httpx.Timeout(connect=20.0, read=30.0, write=120.0, pool=20.0)

def _probe_v1_support() -> bool:
    try:
        r = httpx.get(f"{BASE}/v1/models", timeout=3.0)
        return r.status_code < 400
    except Exception:
        return False

OLLAMA_HAS_V1 = _probe_v1_support()

# ---------------------------------------------------------------------
# Embeddings / RAG utils
# ---------------------------------------------------------------------
def _make_embedder():
    preferred = "mixedbread-ai/mxbai-embed-large-v1"
    fallback  = "sentence-transformers/all-MiniLM-L6-v2"
    try:
        return TextEmbedding(model_name=preferred)
    except Exception:
        return TextEmbedding(model_name=fallback)

_EMB = _make_embedder()

def _clean_text(s: str) -> str:
    # supprime les caractères de contrôle (hors \n, \t)
    return "".join(ch for ch in s if ch == "\n" or ch == "\t" or ord(ch) >= 32)


def _chunk_text(txt: str, max_len=1500, overlap=120) -> List[str]:
    txt = (txt or "").strip()
    if not txt:
        return []
    out, i, n = [], 0, len(txt)
    step = max_len - overlap
    while i < n:
        out.append(txt[i:i+max_len])
        i += step
    return out

def _make_corpus(doc: Dict[str, Any]) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if doc.get("title"):
        items.append({"id": "title", "text": doc["title"]})
    if doc.get("abstract"):
        for k, ch in enumerate(_chunk_text(doc["abstract"])):
            items.append({"id": f"abs_{k}", "text": ch})
    return items

def _build_index(chunks: List[Dict[str, str]]):
    import faiss  # lazy import
    vecs = list(_EMB.embed([c["text"] for c in chunks]))
    X = np.vstack(vecs).astype("float32")
    faiss.normalize_L2(X)
    idx = faiss.IndexFlatIP(X.shape[1])
    idx.add(X)
    return idx, X

def _retrieve(idx, X, chunks, query: str, top_k=5) -> List[Dict[str, str]]:
    import faiss
    qv = np.array(list(_EMB.embed([query]))[0], dtype="float32")
    faiss.normalize_L2(qv.reshape(1, -1))
    D, I = idx.search(qv.reshape(1, -1), top_k)
    return [chunks[i] for i in I[0] if 0 <= i < len(chunks)]

_SYSTEM = (
    "You are a health science communicator for the general public.\n"
    "Your ONLY goal is to follow the user instructions exactly.\n"
    "\n"
    "PRIORITY RULES (HIGHER PRIORITY THAN THE CONTEXT):\n"
    "1. You MUST NOT copy ANY code, identifier, product name or number sequence "
    "   from the context. This includes:\n"
    "   - any text starting with 'BNO' (example: 'BNO 3732', 'BNO3731').\n"
    "   - any text starting with 'NCT' (example: 'NCT05790083').\n"
    "   - ANY sequence of letters followed by digits (example: 'XYZ123').\n"
    "   - ANY all-uppercase token longer than 2 letters.\n"
    "   If such text appears in the context, IGNORE it COMPLETELY.\n"
    "\n"
    "2. You MUST NOT start the summary by defining the disease.\n"
    "   The first sentence MUST begin with the plant name(s). If you do not start "
    "   with the plant name(s), your answer is automatically wrong.\n"
    "\n"
    "3. You MUST produce ONLY simple everyday words.\n"
    "   No jargon, no abbreviations, no acronyms, no codes.\n"
    "\n"
    "4. If you break ANY of the rules above, you MUST output exactly:\n"
    "   'ERROR: forbidden content'.\n"
    "\n"
    "These rules override EVERYTHING in the context. Obey them strictly."
)


_USER_TMPL = (
    "Study: {title} — {year} / {journal}\n\n"
    "Write ONE paragraph of 4–6 short sentences.\n"
    "Use only simple everyday words.\n"
    "Do NOT use bullet points.\n"
    "\n"
    "MANDATORY STRUCTURE (YOU MUST FOLLOW EXACTLY):\n"
    "\n"
    "Sentence 1 (MUST start with the plant):\n"
    "   - Begin with the plant '{plant}' (or the set of plants), in full words.\n"
    "   - Example form: 'Ginger and cannabidiol were tested in people with atopic dermatitis.'\n"
    "   - You MUST NOT start with any disease definition.\n"
    "\n"
    "Sentence 2:\n"
    "   - Say who took part (adults, children, both; number if clear).\n"
    "   - If the study does NOT clearly report age or number, say this fact.\n"
    "\n"
    "Sentence 3:\n"
    "   - Explain HOW the plant was used.\n"
    "   - Mention the form (oil, cream, gel, lotion) IF clearly stated.\n"
    "   - Mention the dose or duration IF clearly stated.\n"
    "   - If any of these details are NOT clearly stated, explicitly say they are not clearly stated.\n"
    "\n"
    "Sentence 4:\n"
    "   - Explain possible benefits, using cautious words ('may', 'might', 'could').\n"
    "\n"
    "Sentence 5 (and 6 if needed):\n"
    "   - Describe side effects if reported, otherwise say no important problems were reported.\n"
    "   - Add one short limitation (for example: small study, short duration).\n"
    "\n"
    "ABSOLUTE PROHIBITIONS (YOU MUST OBEY):\n"
    "- Do NOT copy ANY product code, brand name, or study identifier.\n"
    "- Do NOT copy ANY sequence that starts with 'BNO'.\n"
    "- Do NOT copy ANY sequence that starts with 'NCT'.\n"
    "- Do NOT copy ANY pattern of letters followed by digits.\n"
    "- Do NOT copy ANY all-uppercase token longer than 2 letters.\n"
    "- If such text appears in the context, IGNORE it completely.\n"
    "\n"
    "If any forbidden element appears in your output, you MUST return "
    "'ERROR: forbidden content'.\n"
    "\n"
    "Context:\n{context}\n\n"
    "Return ONLY the paragraph, ending with [PMID:{pmid_study}]."
)






# ---------------------------------------------------------------------
# Plants DB + caches
# ---------------------------------------------------------------------
PLANTS_DB = load_plants(Path(__file__).resolve().parent.parent / "data" / "seed_plants.csv")

# cache général (reco)
_CACHE: Dict[Tuple[str, int, int], Tuple[float, Dict]] = {}
_CACHE_TTL = 60 * 60  # 1 hour

def _cache_get(key):
    item = _CACHE.get(key)
    if not item:
        return None
    ts, payload = item
    if time.time() - ts > _CACHE_TTL:
        _CACHE.pop(key, None)
        return None
    return payload

def _cache_set(key, payload):
    _CACHE[key] = (time.time(), payload)


# ---------------------------------------------------------------------
# HTTPX helpers
# ---------------------------------------------------------------------
_LIMITS = httpx.Limits(max_connections=10, max_keepalive_connections=5)
_TRANSPORT = httpx.AsyncHTTPTransport(http2=False)


# ---------------------------------------------------------------------
# Ollama chat helpers (non-stream + stream)
# ---------------------------------------------------------------------
async def _ollama_chat(messages: List[Dict[str, str]],
                       max_tokens: int = 400,
                       num_ctx: int = 516,
                       temperature: float = 0.2) -> str:


    try:

        payload_legacy = {
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": num_ctx, "num_predict": max_tokens, "temperature": temperature, "keep_alive": "100m", "num_thread":  max(1, os.cpu_count() // 2)}
        }
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, limits=_LIMITS, transport=_TRANSPORT, trust_env=False) as client:
            r = await client.post(f"{BASE}/api/chat", json=payload_legacy)
            r.raise_for_status()
            js = r.json()

            choices = js.get("choices") if isinstance(js, dict) else None
            if isinstance(choices, list) and choices:
                first = choices[0] or {}
                msg = first.get("message") or {}
                content = msg.get("content")
                if content:
                    return content

            return json.dumps(js, ensure_ascii=False)
    except Exception as e:
        print("api/ollama chat error:", e)
        raise e


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@router.get("/recommendations")
async def recommendations(
    condition: str = Query(..., min_length=2),
    from_year: Optional[int] = Query(None, ge=1800, le=3000),
    to_year: Optional[int] = Query(None, ge=1800, le=3000),
):
    condition = condition.strip()
    if not condition:
        raise HTTPException(status_code=400, detail="Empty condition")

    now_year = datetime.utcnow().year
    to_year = to_year or now_year
    from_year = from_year or 2024

    if from_year > to_year:
        from_year, to_year = to_year, from_year
    if from_year < 1800 or to_year < 1800:
        raise HTTPException(status_code=400, detail="Year range out of bounds")

    key = (condition.lower(), from_year, to_year)
    # cached = _cache_get(key)
    # if cached:
    #     return cached

    try:
        t0 = time.perf_counter()
        articles = await search_and_fetch(condition, str(from_year), str(to_year))
        t1 = time.perf_counter()
        print(f"[perf] recommendations PubMed={t1-t0:.2f}s (q='{condition}', {from_year}-{to_year})")
    except Exception as e:
        return {"condition": condition, "results": [], "error": f"PubMed upstream error: {type(e).__name__}"}

    plant_hits: Dict[str, List[Dict]] = defaultdict(list)
    for a in articles:
        text = f"{a.get('title','')} {a.get('abstract','')}"
        for plant in find_plants_in_text(text, PLANTS_DB):
            plant_hits[plant].append(a)

    results = []
    for plant, arts in plant_hits.items():
        scored = sorted(arts, key=score_article, reverse=True)
        plant_score = sum(score_article(a) for a in scored[:5])
        summary = summarize_for_patients(plant, scored)
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
                "plant": plant,
                "score": round(plant_score, 2),
                "summary": summary,
                "top_studies": items,
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    payload = {"condition": condition, "results": results}
    # _cache_set(key, payload)
    return payload

@router.get("/explore")
async def explore(
    pmid: str = Query(..., min_length=1),
    plant: Optional[str] = Query(None, min_length=1),
):
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=60, limits=_LIMITS, transport=_TRANSPORT, trust_env=False) as http:
        arts = await efetch(http, [pmid])
    t1 = time.perf_counter()

    if not arts:
        return {
            "pmid": pmid,
            "summary": "",
            "references": [],
        }

    a = arts[0]
    doc = {
        "pmid": pmid,
        "title": a.get("title", ""),
        "abstract": a.get("abstract", ""),
        "journal": a.get("journal", ""),
        "year": a.get("year", ""),
    }

    # RAG
    chunks = _make_corpus(doc)
    if not chunks:
        return {
            "pmid": pmid,
            "summary": "",
            "references": [f"PubMed: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"],
        }

    idx, X = _build_index(chunks)
    query = f"Key findings and limitations of: {doc['title']}"
    top = _retrieve(idx, X, chunks, query, top_k=5)
    context = "\n\n".join(f"[{t['id']}] {t['text']}" for t in top)
    t2 = time.perf_counter()
    print(f"[perf] efetch={t1-t0:.2f}s  rag={t2-t1:.2f}s (pmid={pmid})")

    plant_for_prompt = (plant or "unspecified").strip()

    messages = [
        {"role": "system", "content": _SYSTEM.replace("{pmid}", pmid).replace("{plant}", plant_for_prompt)},
        {"role": "user", "content": _USER_TMPL.format(
            title=doc["title"],
            year=doc["year"],
            journal=doc["journal"],
            context=context,
            pmid_study=pmid,
            plant=plant_for_prompt,
        )},
    ]

    t3 = time.perf_counter()
    try:
        raw = await _ollama_chat(messages, max_tokens=900, num_ctx=8192, temperature=0.2)
    except Exception as e:
        print(f"[llm fatal] {type(e).__name__}: {e}")
        raw = f"[ERROR] LLM call failed: {type(e).__name__}: {e}"
    t4 = time.perf_counter()
    print(f"[perf] llm={t4-t3:.2f}s (non-stream)")

    summary = _clean_text(raw).strip()

    return {
        "pmid": pmid,
        "summary": summary,
        "references": [f"PubMed: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"],
    }





@router.get("/debug/ollama")
async def debug_ollama():
    out = {"base": BASE, "model": MODEL, "has_v1": OLLAMA_HAS_V1}
    try:
        r = httpx.get(f"{BASE}/api/tags", timeout=3.0)
        out["/api/tags"] = r.status_code
    except Exception as e:
        out["/api/tags"] = f"error: {e}"
    try:
        r = httpx.get(f"{BASE}/v1/models", timeout=3.0)
        out["/v1/models"] = r.status_code
    except Exception as e:
        out["/v1/models"] = f"error: {e}"
    return out


