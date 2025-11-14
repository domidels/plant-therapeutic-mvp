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
OLLAMA_BASE = _sanitize_base(os.getenv("OLLAMA_BASE") or "http://127.0.0.1:11435")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "biomistral")

_HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=10.0)
# Timeout par défaut pour stream (utilisé dans les fallbacks)
_STREAM_TIMEOUT = httpx.Timeout(connect=20.0, read=30.0, write=120.0, pool=20.0)

def _probe_v1_support() -> bool:
    try:
        r = httpx.get(f"{OLLAMA_BASE}/v1/models", timeout=3.0)
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
    "Write simple summaries in clear, everyday language.\n"
    "\n"
    "ABSOLUTE RULES (the model MUST follow them):\n"
    "- Do NOT repeat any technical acronym or abbreviation, even if it appears in the text "
    " (e.g., vIGA-ADTM, SCORAD, EASI, PROMS, or any other).\n"
    "- Do NOT repeat any product name or code (e.g., BNO 3731, BNO 3732, or similar).\n"
    "- Do NOT repeat any clinical trial identifier (e.g., NCT numbers).\n"
    "- Do NOT repeat names of scores, scales, or instruments used in the study.\n"
    "- If the context contains these terms, IGNORE them entirely.\n"
    "- Never describe laboratory or instrumental measurements.\n"
    "- NEVER include technical vocabulary.\n"
    "- Use only everyday words.\n"
    "- Never give medical advice.\n"
    "- Use cautious language ('may', 'might help').\n"
    "\n"
    "ABOUT THE PLANT:\n"
    "- Mention the plant ('{plant}') clearly in the FIRST sentence.\n"
    "- Mention the plant again later in the summary.\n"
    "- Focus only on what the plant may help with, in simple terms.\n"
    "- Mention the duration and population only if simple (e.g., adults, children, 12 weeks).\n"
    "- If safety is mentioned, say simply that it was well tolerated.\n"
    "\n"
    "End the summary with [PMID:{pmid}].\n"
)

_USER_TMPL = (
    "Study: {title} — {year} / {journal}\n\n"
    "Your task:\n"
    "Write a simple 3–5 sentence summary for the general public. DO NOT repeat any technical "
    "term, acronym, score name, product code, or study identifier, even if it appears in the "
    "context below. Replace all technical references with plain, simple descriptions.\n"
    "\n"
    "Focus clearly on the plant ('{plant}') and what the study suggests it may help with. "
    "Mention the plant more than once.\n"
    "\n"
    "Context:\n{context}\n\n"
    "Return ONLY the clean, simplified summary.\n"
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

# cache contexte par PMID
_CTX_CACHE: Dict[str, Tuple[float, Dict[str, str]]] = {}
_CTX_TTL = 60 * 60  # 1 hour

def _ctx_get(pmid: str) -> Optional[Dict[str, str]]:
    item = _CTX_CACHE.get(pmid)
    if not item:
        return None
    ts, payload = item
    if time.time() - ts > _CTX_TTL:
        _CTX_CACHE.pop(pmid, None)
        return None
    return payload

def _ctx_set(pmid: str, payload: Dict[str, str]) -> None:
    _CTX_CACHE[pmid] = (time.time(), payload)

# ---------------------------------------------------------------------
# HTTPX helpers
# ---------------------------------------------------------------------
_LIMITS = httpx.Limits(max_connections=10, max_keepalive_connections=5)
_TRANSPORT = httpx.AsyncHTTPTransport(http2=False)

def _client_for_headers() -> httpx.AsyncClient:
    # Pas de limite sur la lecture des EN-TÊTES (TTFB potentiellement long)
    return httpx.AsyncClient(
        timeout=httpx.Timeout(connect=20.0, read=None, write=120.0, pool=20.0),
        limits=_LIMITS,
        transport=_TRANSPORT,
        trust_env=False,
    )

def _client_for_body_long() -> httpx.Timeout:
    # Utilisé pour relâcher le timeout après réception des en-têtes
    return httpx.Timeout(connect=20.0, read=600.0, write=120.0, pool=20.0)

# ---------------------------------------------------------------------
# Ollama chat helpers (non-stream + stream)
# ---------------------------------------------------------------------
async def _ollama_chat(messages: List[Dict[str, str]],
                       max_tokens: int = 400,
                       num_ctx: int = 516,
                       temperature: float = 0.2) -> str:
    # Try /v1 first
    # try:
    #     print("messages for ollama v1:")
    #     print(messages)
    #     payload_v1 = {
    #         "model": OLLAMA_MODEL,
    #         "messages": messages,
    #         "options": {"num_ctx": num_ctx, "num_predict": max_tokens, "temperature": temperature, "keep_alive": "100m", "num_thread":  max(1, os.cpu_count() // 2)},
    #     }
    #     print("ollama v1 chat payload:", payload_v1)
    #     async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, limits=_LIMITS, transport=_TRANSPORT, trust_env=False) as client:
    #         r = await client.post(f"{OLLAMA_BASE}/v1/chat/completions", json=payload_v1)
    #         if r.status_code < 400:
    #             js = r.json()
    #             return (js.get("choices", [{}])[0].get("message", {}).get("content", "")) or ""
    # except Exception as e:
    #     print("api/ollama v1 chat error:", e)
    #     pass

    try:
        # print("ollama fallback generate payload:")


        # r = await client.post(f"{OLLAMA_BASE}/api/generate", json={
        #     "model": OLLAMA_MODEL,
        #     "prompt": "\n\n".join(f"{m['role'].upper()}:{ m['content']}" for m in messages) + "\n\nASSISTANT:",
        #     "max_tokens": max_tokens,
        #     "temperature": temperature,
        #     "stream": False,
        # })
        # r.raise_for_status()
        # js = r.json()
        # if "response" in js and js["response"]:
        #     return js["response"]

        # Fallback legacy /api/chat
        payload_legacy = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": num_ctx, "num_predict": max_tokens, "temperature": temperature, "keep_alive": "100m", "num_thread":  max(1, os.cpu_count() // 2)}
        }
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, limits=_LIMITS, transport=_TRANSPORT, trust_env=False) as client:
            r = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload_legacy)
            r.raise_for_status()
            js = r.json()

            # 2) format OpenAI /v1/chat/completions :
            choices = js.get("choices") if isinstance(js, dict) else None
            if isinstance(choices, list) and choices:
                first = choices[0] or {}
                msg = first.get("message") or {}
                content = msg.get("content")
                if content:
                    return content

            # 3) fallback debug : renvoyer tout l'objet JSON
            # (tu peux retirer ça plus tard si tout est stable)
            return json.dumps(js, ensure_ascii=False)
    except Exception as e:
        print("api/ollama chat error:", e)
        raise e

async def _ollama_stream(
    messages: List[Dict[str, str]],
    *,
    num_ctx: int = 516,
    temperature: float = 0.2,
    max_tokens: int = 320,
):
    """
    Try Ollama /v1 first (OpenAI-compat), then legacy /api/chat, then /api/generate.
    Yields plain text deltas (no SSE framing).
    """
    # --- helper: concat messages into a plain prompt for /api/generate
    def _as_prompt(msgs: List[Dict[str, str]]) -> str:
        parts = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role.upper()}: {content}")
        parts.append("ASSISTANT:")
        return "\n\n".join(parts)

    # 1) /v1/chat/completions (headers -> body, timeouts dédiés)
    try:
        payload_v1 = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": True,
            "options": {"num_ctx": num_ctx, "num_predict": max_tokens, "temperature": temperature, "keep_alive": "100m", "num_thread":  max(1, os.cpu_count() // 2)},
        }
        async with _client_for_headers() as c:
            try:
                req = c.build_request("POST", f"{OLLAMA_BASE}/v1/chat/completions", json=payload_v1)
                start_headers = time.perf_counter()
                r = await c.send(req, stream=True)  # attend les EN-TÊTES
                print(f"[perf] v1 headers in {time.perf_counter() - start_headers:.2f}s")

                if r.status_code < 400:
                    # Relâcher le timeout pour le CORPS du stream
                    try:
                        r._content_reader._timeout = _client_for_body_long()
                    except Exception:
                        pass
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        data = line[5:].strip() if line.startswith("data:") else line.strip()
                        if data == "[DONE]":
                            break
                        try:
                            js = json.loads(data)
                            delta = js.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if delta:
                                yield delta
                        except Exception:
                            pass
                    return
                else:
                    try:
                        body = await r.aread()
                        print(f"[ollama v1 {r.status_code}] {body.decode(errors='ignore')[:200]}")
                    except Exception:
                        pass
            except httpx.ReadTimeout:
                print("[ollama v1] header read timed out; falling back")
            except Exception as e:
                print(f"[ollama v1 error] {type(e).__name__}: {e}")
    except Exception as e:
        print(f"[ollama v1 outer error] {type(e).__name__}: {e}")

    # 2) Legacy /api/chat ------------------------------------------------------
    try:
        payload_legacy = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": True,
            "options": {"num_ctx": num_ctx, "num_predict": max_tokens, "temperature": temperature, "keep_alive": "100m", "num_thread":  max(1, os.cpu_count() // 2)},
        }
        async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT, limits=_LIMITS, transport=_TRANSPORT, trust_env=False) as c:
            async with c.stream("POST", f"{OLLAMA_BASE}/api/chat", json=payload_legacy) as r:
                if r.status_code < 400:
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        try:
                            js = json.loads(line)
                            chunk = js.get("message", {}).get("content", "")
                            if chunk:
                                yield chunk
                        except Exception:
                            pass
                    return
                else:
                    try:
                        body = await r.aread()
                        print(f"[ollama legacy /api/chat {r.status_code}] {body.decode(errors='ignore')[:200]}")
                    except Exception:
                        pass
    except Exception as e:
        print(f"[ollama legacy /api/chat error] {type(e).__name__}: {e}")

    # 3) Ultimate fallback: /api/generate -------------------------------------
    try:
        payload_gen = {"model": OLLAMA_MODEL, "prompt": _as_prompt(messages), "stream": True}
        async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT, limits=_LIMITS, transport=_TRANSPORT, trust_env=False) as c:
            async with c.stream("POST", f"{OLLAMA_BASE}/api/generate", json=payload_gen) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        js = json.loads(line)
                        if "response" in js and js["response"]:
                            yield js["response"]
                        if js.get("done"):
                            break
                    except Exception:
                        if line.startswith("data:"):
                            try:
                                js = json.loads(line[5:].strip())
                                if "response" in js and js["response"]:
                                    yield js["response"]
                                if js.get("done"):
                                    break
                            except Exception:
                                pass
        return
    except Exception as e:
        print(f"[ollama /api/generate error] {type(e).__name__}: {e}")

    # 4) Rien n'a marché
    yield "[Model unreachable or bad request — check OLLAMA_MODEL and endpoints]"

# ---------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------
def _sse_data(data: str) -> str:
    return f"data: {data}\n\n"

def _sse_comment(comment: str = "keep-alive") -> str:
    return f": {comment}\n\n"

# ---------------------------------------------------------------------
# Warmup (pré-charge le modèle)
# ---------------------------------------------------------------------
# @router.on_event("startup")
# async def _warm_ollama():
#     try:
#         msg = [{"role": "user", "content": "ok"}]
#         _ = await _ollama_chat(msg, max_tokens=1, num_ctx=256, temperature=0.0)
#         print("[warmup] ollama model preloaded")
#     except Exception as e:
#         print(f"[warmup] skip: {e}")

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




# --- helper utilisé par stream (avec cache) ---
async def _make_context_for_pmid(pmid: str) -> Dict[str, str]:
    cached = _ctx_get(pmid)
    if cached:
        return cached

    async with httpx.AsyncClient(timeout=60, limits=_LIMITS, transport=_TRANSPORT, trust_env=False) as http:
        t0 = time.perf_counter()
        arts = await efetch(http, [pmid])
        t1 = time.perf_counter()

    if not arts:
        payload = {"pmid": pmid, "title": "", "year": "", "journal": "", "context": ""}
        _ctx_set(pmid, payload)
        return payload

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
        payload = {**doc, "context": ""}
        _ctx_set(pmid, payload)
        return payload

    idx, X = _build_index(chunks)
    query = f"Key findings and limitations of: {doc['title']}"
    top = _retrieve(idx, X, chunks, query, top_k=3)  # top-3 suffit pour le stream
    context = "\n\n".join(f"[{t['id']}] {t['text']}" for t in top)
    t2 = time.perf_counter()
    print(f"[perf] (stream ctx) efetch={t1-t0:.2f}s rag={t2-t1:.2f}s (pmid={pmid})")

    payload = {**doc, "context": context}
    _ctx_set(pmid, payload)
    return payload

@router.get("/explore/stream")
async def explore_stream(
    pmid: str = Query(..., min_length=1),
    plant: Optional[str] = Query(None, min_length=1),
    request: Request = None,
):
    """
    SSE endpoint; each event ends with a blank line.
    Includes keep-alive comments so curl -N does not time out.
    """
    doc = await _make_context_for_pmid(pmid)
    plant_for_prompt = (plant or "unspecified").strip()

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Content-Type": "text/event-stream; charset=utf-8",
    }



    async def event_gen():
        # Healthcheck modèle (llama.cpp /v1 ; fallback Ollama /api)
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=8.0, read=8.0, write=8.0, pool=8.0),
                trust_env=False
            ) as c:
                pr = await c.get(f"{OLLAMA_BASE}/v1/models")
                if pr.status_code >= 400:
                    pr2 = await c.get(f"{OLLAMA_BASE}/api/tags")
                    if pr2.status_code >= 400:
                        msg = f"[WARN] LLM healthcheck failed at {OLLAMA_BASE} (/v1 and /api)"
                        print(msg)
                        yield _sse_comment(msg)
        except Exception as e:
            msg = f"[WARN] LLM healthcheck exception: {type(e).__name__}: {e} (base={OLLAMA_BASE})"
            print(msg)
            yield _sse_comment(msg)

        # En-tête initial
        yield _sse_data(f"[PMID:{pmid}] {doc.get('title') or 'Exploration'}")

        # Prépare messages
        messages = [
            {"role": "system", "content": _SYSTEM.replace("{pmid}", pmid).replace("{plant}", plant_for_prompt)},
            {"role": "user", "content": _USER_TMPL.format(
                title=doc.get("title", ""), year=doc.get("year", ""), journal=doc.get("journal", ""),
                context=doc.get("context", ""), plant=plant_for_prompt
            )},
        ]

        # Stream tokens avec buffering word-safe + keep-alive
        last_ping = time.monotonic()
        start = time.perf_counter()
        first_token_time = None

        buf_text: str = ""
        SOFT_FLUSH = 160   # seuil doux
        HARD_FLUSH = 400   # on force un flush même sans espace si vraiment trop gros

        def _maybe_flush(force: bool = False):
            nonlocal buf_text
            if not buf_text:
                return None
            if not force and len(buf_text) < SOFT_FLUSH and not any(buf_text.endswith(x) for x in (".", "!", "?", "\n")):
                return None
            # on cherche une frontière (espace/ponctuation) la plus à droite
            cut = max(buf_text.rfind(" "), buf_text.rfind("\n"), buf_text.rfind("\t"))
            if cut < 0:
                # si pas d'espace et qu'on dépasse HARD_FLUSH, on tranche tel quel
                if force or len(buf_text) >= HARD_FLUSH:
                    out = buf_text
                    buf_text = ""
                    return out
                return None
            out = buf_text[:cut+1]
            buf_text = buf_text[cut+1:]
            return out

        try:
            async for token in _ollama_stream(messages, num_ctx=4092, temperature=0.2, max_tokens=300):
                if first_token_time is None and token.strip():
                    first_token_time = time.perf_counter()
                    print(f"[perf] ttfb={first_token_time - start:.2f}s (stream)")

                # keep-alive
                now = time.monotonic()
                if now - last_ping > 15:
                    yield _sse_comment("keep-alive")
                    last_ping = now

                # accumulate + nettoyer
                buf_text += _clean_text(token)

                # soft flush si on a atteint un seuil ou une fin de phrase
                to_send = _maybe_flush(force=False)
                if to_send:
                    yield _sse_data(to_send)

            # flush final (forcer s'il reste des miettes sans espace)
            to_send = _maybe_flush(force=True)
            if to_send:
                yield _sse_data(to_send)

            yield _sse_data("[END]")
        except Exception as e:
            yield _sse_data(f"[ERROR] {type(e).__name__}: {e}")
            yield _sse_data("[END]")

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)



@router.get("/debug/ollama")
async def debug_ollama():
    out = {"base": OLLAMA_BASE, "model": OLLAMA_MODEL, "has_v1": OLLAMA_HAS_V1}
    try:
        r = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=3.0)
        out["/api/tags"] = r.status_code
    except Exception as e:
        out["/api/tags"] = f"error: {e}"
    try:
        r = httpx.get(f"{OLLAMA_BASE}/v1/models", timeout=3.0)
        out["/v1/models"] = r.status_code
    except Exception as e:
        out["/v1/models"] = f"error: {e}"
    return out


