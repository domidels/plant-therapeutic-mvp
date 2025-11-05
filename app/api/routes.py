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

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434").split("#", 1)[0].strip().split()[0].rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "biomistral")

_HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=10.0)
_STREAM_TIMEOUT = httpx.Timeout(connect=20.0, read=None, write=120.0, pool=20.0)

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

def _chunk_text(txt: str, max_len=1200, overlap=150) -> List[str]:
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
    import faiss
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
    "You are a careful biomedical literature explainer. "
    "Summarize only from the provided context. "
    "Cite with [PMID:{pmid}]. Do NOT give medical advice. "
    "Use cautious language ('may', 'suggests')."
)

_USER_TMPL = (
    "Study: {title} — {year} / {journal}\n\n"
    "Question: Summarize key findings and limitations for a general audience using the context.\n\n"
    "Context:\n{context}\n\n"
    "Return JSON with keys: summary (2-4 sentences), key_points (3-6 bullets), limitations (2-4 bullets)"
)

# ---------------------------------------------------------------------
# Plants DB + simple cache
# ---------------------------------------------------------------------
PLANTS_DB = load_plants(Path(__file__).resolve().parent.parent / "data" / "seed_plants.csv")

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
# Ollama chat helpers (non-stream + stream)
# ---------------------------------------------------------------------
async def _ollama_chat(messages: List[Dict[str, str]],
                       max_tokens: int = 400,
                       num_ctx: int = 516,
                       temperature: float = 0.2) -> str:
    # Try /v1 first
    try:
        payload_v1 = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": {"options": {"num_ctx": num_ctx, "keep_alive": "100m"}}
        }
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            r = await client.post(f"{OLLAMA_BASE}/v1/chat/completions", json=payload_v1)
            if r.status_code < 400:
                js = r.json()
                return (js.get("choices", [{}])[0].get("message", {}).get("content", "")) or ""
    except Exception:
        pass

    # Fallback legacy /api/chat
    payload_legacy = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": num_ctx, "temperature": temperature, "keep_alive": "10m"}
    }
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        r = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload_legacy)
        r.raise_for_status()
        js = r.json()
        if "message" in js and "content" in js["message"]:
            return js["message"]["content"]
        return json.dumps(js)

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

    # 1) /v1/chat/completions (Ollama v1)  ------------------------------------
    try:
        payload_v1 = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            # IMPORTANT: Ollama attend `options` au top-level (pas `extra_body`)
            "options": {"num_ctx": num_ctx, "keep_alive": "20m"},
        }
        async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as c:
            async with c.stream(
                "POST",
                f"{OLLAMA_BASE}/v1/chat/completions",
                json=payload_v1,
            ) as r:
                if r.status_code < 400:
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
                            # ignore unparsable line
                            pass
                    return
                else:
                    # log utile (en dev) : voir la raison du 400
                    try:
                        body = await r.aread()
                        print(f"[ollama v1 400] {body.decode(errors='ignore')}")
                    except Exception:
                        pass
    except Exception as e:
        # continue to legacy
        print(f"[ollama v1 error] {type(e).__name__}: {e}")

    # 2) Legacy /api/chat ------------------------------------------------------
    try:
        payload_legacy = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": True,
            "options": {"num_ctx": num_ctx, "temperature": temperature, "keep_alive": "20m"},
        }
        async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as c:
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
                        print(f"[ollama legacy /api/chat {r.status_code}] {body.decode(errors='ignore')}")
                    except Exception:
                        pass
    except Exception as e:
        print(f"[ollama legacy /api/chat error] {type(e).__name__}: {e}")

    # 3) Ultimate fallback: /api/generate (très robuste) ----------------------
    try:
        payload_gen = {
            "model": OLLAMA_MODEL,
            "prompt": _as_prompt(messages),
            "stream": True,
            # pas d'options exotiques ici pour éviter les 400
        }
        async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as c:
            async with c.stream("POST", f"{OLLAMA_BASE}/api/generate", json=payload_gen) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    # /api/generate renvoie souvent des lignes JSON {"response": "...", "done": bool}
                    try:
                        js = json.loads(line)
                        if "response" in js and js["response"]:
                            yield js["response"]
                        if js.get("done"):
                            break
                    except Exception:
                        # si jamais c'est déjà du 'data: {...}'
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
    # Single SSE event; caller must ensure small enough chunks
    return f"data: {data}\n\n"

def _sse_comment(comment: str = "keep-alive") -> str:
    # SSE comment (ping) — invisible to EventSource; useful with curl -N
    return f": {comment}\n\n"

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
    cached = _cache_get(key)
    if cached:
        return cached

    try:
        articles = await search_and_fetch(condition, str(from_year), str(to_year))
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
    _cache_set(key, payload)
    return payload

@router.get("/explore")
async def explore(pmid: str = Query(..., min_length=1)):
    async with httpx.AsyncClient(timeout=60) as http:
        arts = await efetch(http, [pmid])
    if not arts:
        return {"pmid": pmid, "summary": "", "key_points": [], "limitations": [], "references": []}
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
        return {"pmid": pmid, "summary": "", "key_points": [], "limitations": [], "references": []}

    idx, X = _build_index(chunks)
    query = f"Key findings and limitations of: {doc['title']}"
    top = _retrieve(idx, X, chunks, query, top_k=5)
    context = "\n\n".join(f"[{t['id']}] {t['text']}" for t in top)

    messages = [
        {"role": "system", "content": _SYSTEM.replace("{pmid}", pmid)},
        {"role": "user", "content": _USER_TMPL.format(
            title=doc["title"], year=doc["year"], journal=doc["journal"], context=context
        )},
    ]
    raw = await _ollama_chat(messages, max_tokens=700, num_ctx=4096, temperature=0.2)

    try:
        js = json.loads(raw)
    except Exception:
        js = {"summary": raw, "key_points": [], "limitations": []}

    js.setdefault("summary", "")
    js.setdefault("key_points", [])
    js.setdefault("limitations", [])
    js["pmid"] = pmid
    js["references"] = [f"PubMed: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"]
    return js

# --- small helper used by stream route ---
async def _make_context_for_pmid(pmid: str) -> Dict[str, str]:
    async with httpx.AsyncClient(timeout=60) as http:
        arts = await efetch(http, [pmid])
    if not arts:
        return {"pmid": pmid, "title": "", "year": "", "journal": "", "context": ""}
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
        return {**doc, "context": ""}
    idx, X = _build_index(chunks)
    query = f"Key findings and limitations of: {doc['title']}"
    top = _retrieve(idx, X, chunks, query, top_k=3)
    context = "\n\n".join(f"[{t['id']}] {t['text']}" for t in top)
    return {**doc, "context": context}

@router.get("/explore/stream")
async def explore_stream(pmid: str = Query(..., min_length=1), request: Request = None):
    """
    SSE endpoint; each event ends with a blank line.
    Includes keep-alive comments so curl -N does not time out.
    """
    doc = await _make_context_for_pmid(pmid)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # disable nginx buffering if present
    }

    async def event_gen():
        # Initial header message
        yield _sse_data(f"[PMID:{pmid}] {doc.get('title') or 'Exploration'}")

        # Prepare messages
        messages = [
            {"role": "system", "content": _SYSTEM.replace("{pmid}", pmid)},
            {"role": "user", "content": _USER_TMPL.format(
                title=doc.get("title", ""), year=doc.get("year", ""), journal=doc.get("journal", ""),
                context=doc.get("context", "")
            )},
        ]

        # Stream tokens and send keep-alive pings
        last_ping = time.monotonic()
        try:
            async for token in _ollama_stream(messages, num_ctx=516, temperature=0.2):
                # periodic keep-alive comment (every ~15s)
                now = time.monotonic()
                if now - last_ping > 15:
                    yield _sse_comment("keep-alive")
                    last_ping = now

                # write token as SSE data
                yield _sse_data(token)

            # graceful end
            yield _sse_data("[END]")
        except Exception as e:
            # propagate an error message as a final SSE event (optional)
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
