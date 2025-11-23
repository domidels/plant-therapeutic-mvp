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
import re

def _sentence_split(text: str) -> List[str]:
    # Découpe sur . ! ? mais en gardant les séparateurs
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sents if s]

def _chunk_text_sentence_safe(txt: str, max_len=800) -> List[str]:
    sentences = _sentence_split(txt)
    chunks = []
    current = ""

    for s in sentences:
        # si ajouter la phrase dépasse max_len → on ferme le chunk
        if len(current) + len(s) + 1 > max_len:
            if current:
                chunks.append(current.strip())
            current = s
        else:
            current += " " + s if current else s

    if current:
        chunks.append(current.strip())

    return chunks


def _make_embedder():
    preferred = "mixedbread-ai/mxbai-embed-large-v1"
    fallback = "sentence-transformers/all-MiniLM-L6-v2"
    try:
        return TextEmbedding(model_name=preferred)
    except Exception:
        return TextEmbedding(model_name=fallback)


_EMB = _make_embedder()


def _clean_text(s: str) -> str:
    # supprime les caractères de contrôle (hors \n, \t)
    return "".join(ch for ch in s if ch == "\n" or ch == "\t" or ord(ch) >= 32)


# def _chunk_text(txt: str, max_len=1000, overlap=100) -> List[str]:
#     txt = (txt or "").strip()
#     if not txt:
#         return []
#     out, i, n = [], 0, len(txt)
#     step = max_len - overlap
#     while i < n:
#         out.append(txt[i : i + max_len])
#         i += step
#     return out


def _make_corpus(doc: Dict[str, Any]) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if doc.get("title"):
        items.append({"id": "title", "text": doc["title"]})
    if doc.get("abstract"):
        for k, ch in enumerate(_chunk_text_sentence_safe(doc["abstract"])):
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

    # 1) Embed + normalisation de la requête
    qv = np.array(list(_EMB.embed([query]))[0], dtype="float32")
    faiss.normalize_L2(qv.reshape(1, -1))

    # 2) Recherche FAISS
    D, I = idx.search(qv.reshape(1, -1), top_k)

    # 3) On enlève les doublons / indices invalides en gardant l'ordre de score
    seen = set()
    selected: List[int] = []
    for i in I[0]:
        if i < 0 or i >= len(chunks):
            continue
        if i in seen:
            continue
        seen.add(i)
        selected.append(i)
        if len(selected) >= top_k:
            break

    # 4) On re-trie les indices selon l'ordre d'apparition dans le document
    #    => on privilégie la cohérence de lecture plutôt que l'ordre de score brut
    selected_sorted = sorted(selected)

    # 5) On renvoie les chunks dans l'ordre du texte
    return [chunks[i] for i in selected_sorted]


_SYSTEM = (
    "You are a health science communicator for the general as related public.\n"
    "Your goal is to mmarize scientific article by: \n"
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
    "These rules override EVERYTHING in the user CONTEXT. Obey them strictly."
)

_USER_TMPL = (
    "Study: {title} — {year} / {journal}\n\n"
    "CONTEXT:\n{context}\n\n"
    "YOUR ONLY GOAL IS TO SUMMARIZE the CONTEXT above and you MUST focus on plant(s) as described in this CONTEXT.\n"
    "RELATE MAIN KEY FINDINGS FROM THIS CONTEXT ONLY.\n"
    "DO NOT MENTION KEY FINDINGS FROM OUTSIDE THIS CONTEXT.\n"
    "MENTION:\n"
    "- ALL plants as well as ALL plant compounds IF mentioned in the CONTEXT\n"  
    "- main KEY FINDINGS related to THE {plant}, to any other plant or to any plant compounds IF mentioned in the CONTEXT\n" 
    "- who (Women, men, children), how many particpated, the age of participants to the study ONLY IF mentioned in the CONTEXT\n"    
    "- study duration ONLY IF mentioned in the CONTEXT\n"
    "- ANY DOSAGE, FORMULATIONS, ADMINISTRATION ROUTES ONLY IF mentioned in the CONTEXT\n" 
    "- adverse effects and limitations ONLY IF mentioned in the CONTEXT\n" 
    "\n"
    "Write ONE paragraph of 4–6 sentences.\n"
    "Use OMLY SIMPLE everyday words.\n"
    "DO NOT USE abbreviations nor acronyms.\n"
    "Do NOT use bullet points.\n"
    "\n"
    "Return ONLY the paragraph."
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
# LLM helper (un seul prompt, /v1/completions)
# ---------------------------------------------------------------------
async def _ollama_chat(
    messages: List[Dict[str, str]],
    max_tokens: int = 900,
    num_ctx: int = 516,  # ignoré par llama-server, mais gardé pour compat
    temperature: float = 0.2,
) -> str:
        payload_legacy = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": num_ctx, "num_predict": max_tokens, "temperature": temperature, "keep_alive": "100m", "num_thread":  max(1, os.cpu_count() // 2)}
    }
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, limits=_LIMITS, transport=_TRANSPORT, trust_env=False) as client:
            r = await client.post(f"{BASE}/v1/chat/completions", json=payload_legacy)
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

        return json.dumps(js, ensure_ascii=False)



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
        print(
            f"[perf] recommendations PubMed={t1-t0:.2f}s "
            f"(q='{condition}', {from_year}-{to_year})"
        )
    except Exception as e:
        return {
            "condition": condition,
            "results": [],
            "error": f"PubMed upstream error: {type(e).__name__}",
        }

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
    async with httpx.AsyncClient(
        timeout=60, limits=_LIMITS, transport=_TRANSPORT, trust_env=False
    ) as http:
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
    context = "\n".join(f"{t['text']}" for t in top)

    t2 = time.perf_counter()
    print(f"[perf] efetch={t1-t0:.2f}s  rag={t2-t1:.2f}s (pmid={pmid})")

    plant_for_prompt = (plant or "unspecified").strip()

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
    print(f"context {context}")
    t3 = time.perf_counter()
    try:
        # 300 tokens suffisent pour 4–6 phrases
        raw = await _ollama_chat(messages, max_tokens=300, num_ctx=4096, temperature=0)
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

from fastapi.responses import StreamingResponse
import json

# ...

@router.get("/explore_stream")
async def explore_stream(
    pmid: str = Query(..., min_length=1),
    plant: Optional[str] = Query(None, min_length=1),
):
    t0 = time.perf_counter()
    async with httpx.AsyncClient(
        timeout=_STREAM_TIMEOUT, limits=_LIMITS, transport=_TRANSPORT, trust_env=False
    ) as http:
        arts = await efetch(http, [pmid])
    t1 = time.perf_counter()

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

    # RAG identique à /explore
    chunks = _make_corpus(doc)
    if not chunks:
        async def gen_no_abs():
            yield "No abstract text available.\n"
            yield f"References:\nPubMed: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        return StreamingResponse(gen_no_abs(), media_type="text/plain")

    idx, X = _build_index(chunks)
    query = f"Key findings and limitations of: {doc['title']}"
    top = _retrieve(idx, X, chunks, query, top_k=5)
    context = "\n".join(f"{t['text']}" for t in top)
    t2 = time.perf_counter()
    print(f"[perf] (stream) efetch={t1-t0:.2f}s  rag={t2-t1:.2f}s (pmid={pmid})")
    print(f"context {context}")
    plant_for_prompt = (plant or "unspecified").strip()

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
        t3 = time.perf_counter()
        try:
            payload = {
                "model": MODEL,
                "messages": messages,
                "stream": True,
                "options": {
                    "num_ctx": 4096,
                    "num_predict": 300,
                    "temperature": 0.0,
                    "keep_alive": "100m",
                    "num_thread": max(1, os.cpu_count() // 2),
                },
            }
            async with httpx.AsyncClient(
                timeout=_STREAM_TIMEOUT, limits=_LIMITS, transport=_TRANSPORT, trust_env=False
            ) as client:
                async with client.stream(
                    "POST",
                    f"{BASE}/v1/chat/completions",
                    json=payload,
                ) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        # format OpenAI-like: "data: {...}"
                        if line.startswith("data: "):
                            data = line[len("data: "):].strip()
                        else:
                            continue
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
                        # nettoyage léger comme _clean_text
                        chunk = _clean_text(delta)
                        if chunk:
                            yield chunk

        except Exception as e:
            err = f"\n[ERROR] LLM streaming failed: {type(e).__name__}: {e}\n"
            print(err)
            yield err

        t4 = time.perf_counter()
        print(f"[perf] llm_stream={t4-t3:.2f}s (stream)")

        # Ajout des références en fin de flux
        refs = f"\n\nReferences:\nPubMed: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        yield refs

    return StreamingResponse(event_generator(), media_type="text/plain")
