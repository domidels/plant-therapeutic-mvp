# app/api/routes.py
from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastembed import TextEmbedding

from app.services.medline import get_medlineplus_fullsummary
from app.services.plants_v2 import find_plants_in_text, load_plants
from app.services.pubmed import efetch, search_and_fetch
from app.services.ranking import score_article, summarize_for_patients

# ---------------------------------------------------------------------
# Router & Config
# ---------------------------------------------------------------------
router = APIRouter()

# ---------------------------------------------------------------------
# Hugging Face Inference API config
# ---------------------------------------------------------------------



HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_BASE_URL = "https://router.huggingface.co/v1"
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2").strip()

# Timeouts: l'API peut "cold start" (503 + estimated_time), donc read assez large.
_HF_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=60.0, pool=10.0)
_HF_STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=60.0, pool=10.0)

_LIMITS = httpx.Limits(max_connections=10, max_keepalive_connections=5)
_TRANSPORT = httpx.AsyncHTTPTransport(http2=False)


def _hf_headers() -> Dict[str, str]:
    if not HF_TOKEN:
        # On laisse l'appel échouer clairement plus bas si la clé manque
        return {"Content-Type": "application/json"}
    return {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------
# Embeddings / RAG utils
# ---------------------------------------------------------------------
import re


def _sentence_split(text: str) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sents if s]


def _chunk_text_sentence_safe(txt: str, max_len=800) -> List[str]:
    sentences = _sentence_split(txt)
    chunks: List[str] = []
    current = ""

    for s in sentences:
        if len(current) + len(s) + 1 > max_len:
            if current:
                chunks.append(current.strip())
            current = s
        else:
            current += " " + s if current else s

    if current:
        chunks.append(current.strip())

    return chunks


# ---------------------------------------------------------------------
# Embeddings (LAZY INIT) - important for Vercel/serverless
# ---------------------------------------------------------------------
_EMB: Optional[TextEmbedding] = None


def _configure_fastembed_cache():
    """
    Sur Vercel/serverless:
    - /tmp est le seul endroit écrivable fiable
    - fastembed/huggingface cache doivent être dirigés vers /tmp
    """
    cache_root = os.getenv("FASTEMBED_CACHE_PATH") or "/tmp/fastembed_cache"

    os.environ.setdefault("FASTEMBED_CACHE_PATH", cache_root)
    os.environ.setdefault("FASTEMBED_CACHE_DIR", cache_root)

    os.environ.setdefault("HF_HOME", "/tmp/hf")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")


def _make_embedder() -> TextEmbedding:
    _configure_fastembed_cache()
    model_name = os.getenv("EMBED_MODEL") or "qdrant/all-MiniLM-L6-v2-onnx"
    return TextEmbedding(model_name=model_name)


def _get_embedder() -> TextEmbedding:
    global _EMB
    if _EMB is None:
        print("[embed] Initializing embedder (lazy)")
        _EMB = _make_embedder()
        print("[embed] Embedder initialized OK")
    return _EMB


def _clean_text(s: str) -> str:
    return "".join(ch for ch in s if ch == "\n" or ch == "\t" or ord(ch) >= 32)


def _make_corpus(doc: Dict[str, Any]) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if doc.get("title"):
        items.append({"id": "title", "text": doc["title"]})
    if doc.get("abstract"):
        for k, ch in enumerate(_chunk_text_sentence_safe(doc["abstract"])):
            items.append({"id": f"abs_{k}", "text": ch})
    return items


def _build_index(chunks: List[Dict[str, str]]):
    # lazy import: évite de casser tout le projet si faiss n'est pas dispo sur Vercel
    import faiss  # type: ignore

    emb = _get_embedder()
    vecs = list(emb.embed([c["text"] for c in chunks]))
    X = np.vstack(vecs).astype("float32")
    faiss.normalize_L2(X)
    idx = faiss.IndexFlatIP(X.shape[1])
    idx.add(X)
    return idx, X


def _retrieve(idx, X, chunks, query: str, top_k=5) -> List[Dict[str, str]]:
    import faiss  # type: ignore

    emb = _get_embedder()
    qv = np.array(list(emb.embed([query]))[0], dtype="float32")
    faiss.normalize_L2(qv.reshape(1, -1))

    D, I = idx.search(qv.reshape(1, -1), top_k)

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

    selected_sorted = sorted(selected)
    return [chunks[i] for i in selected_sorted]


# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Plants DB + caches
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
# Hugging Face "chat" helpers (messages -> prompt)
# ---------------------------------------------------------------------
def _format_messages_for_hf(messages: List[Dict[str, str]]) -> str:
    """
    La Serverless Inference API est typiquement "text-generation": elle prend un seul prompt (string).
    On concatène system/user avec un format simple et robuste.
    """
    system_parts: List[str] = []
    user_parts: List[str] = []
    other_parts: List[str] = []

    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
        else:
            other_parts.append(f"{role.upper()}:\n{content}")

    system_txt = "\n\n".join(system_parts).strip()
    user_txt = "\n\n".join(user_parts).strip()
    other_txt = "\n\n".join(other_parts).strip()

    # Format instruct très classique :
    prompt = ""
    if system_txt:
        prompt += f"[SYSTEM]\n{system_txt}\n\n"
    if user_txt:
        prompt += f"[USER]\n{user_txt}\n\n"
    if other_txt:
        prompt += f"{other_txt}\n\n"
    prompt += "[ASSISTANT]\n"
    return prompt


async def _hf_generate(
    prompt: str,
    *,
    max_new_tokens: int = 300,
    temperature: float = 0.2,
    do_sample: bool = False,
    top_p: float = 0.95,
) -> str:
    """
    Appel Hugging Face Serverless Inference API.
    - Peut renvoyer 503 "model is loading" avec estimated_time.
    - Certaines configs de modèles refusent certains paramètres: on reste minimal.
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is missing (set it in env vars).")

    payload: Dict[str, Any] = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "top_p": top_p,
            "return_full_text": False,
        },
        "options": {
            # Si True, HF attend que le modèle soit chargé (peut bloquer longtemps).
            # En free tier c'est souvent préférable de gérer nous-mêmes le retry/backoff.
            "wait_for_model": False
        },
    }

    headers = _hf_headers()

    # Retry simple si modèle en chargement (503 + {"estimated_time": ...})
    max_wait_s = 45.0
    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=_HF_TIMEOUT, limits=_LIMITS, transport=_TRANSPORT) as client:
        while True:
            r = await client.post(HF_BASE_URL, headers=headers, json=payload)

            # Cas: modèle en chargement
            if r.status_code == 503:
                try:
                    js = r.json()
                except Exception:
                    js = {}
                est = float(js.get("estimated_time") or 2.0)
                elapsed = time.perf_counter() - start
                if elapsed + est > max_wait_s:
                    raise HTTPException(
                        status_code=503,
                        detail=f"HuggingFace model loading too long (>{max_wait_s}s). Try again.",
                    )
                sleep_s = max(1.0, min(est, 8.0))
                print(f"[hf] Model loading. Retry in {sleep_s:.1f}s (estimated_time={est})")
                await asyncio_sleep(sleep_s)
                continue

            # Auth / quota / erreurs
            if r.status_code >= 400:
                try:
                    err = r.json()
                except Exception:
                    err = {"error": r.text}
                raise HTTPException(status_code=r.status_code, detail=err)

            data = r.json()

            # Formats fréquents:
            # - [{"generated_text": "..."}]
            # - {"generated_text": "..."} (plus rare)
            # - {"error": "..."}
            if isinstance(data, list) and data:
                first = data[0] or {}
                txt = first.get("generated_text")
                if isinstance(txt, str):
                    return txt
            if isinstance(data, dict):
                if isinstance(data.get("generated_text"), str):
                    return data["generated_text"]
                if data.get("error"):
                    raise HTTPException(status_code=502, detail=data)

            return json.dumps(data, ensure_ascii=False)


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
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data: "):
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


async def asyncio_sleep(seconds: float) -> None:
    # petite helper pour éviter d'importer asyncio en global si tu veux minimal
    import asyncio

    await asyncio.sleep(seconds)


# ---------------------------------------------------------------------
# Condition expansion cache
# ---------------------------------------------------------------------
_COND_EXPANSION_TTL = 60 * 60  # 1h
_COND_EXPANSION_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def _cond_cache_get(cond: str) -> Optional[Dict[str, Any]]:
    key = cond.strip().lower()
    item = _COND_EXPANSION_CACHE.get(key)
    if not item:
        return None
    ts, payload = item
    if time.time() - ts > _COND_EXPANSION_TTL:
        _COND_EXPANSION_CACHE.pop(key, None)
        return None
    return payload


def _cond_cache_set(cond: str, payload: Dict[str, Any]) -> None:
    key = cond.strip().lower()
    _COND_EXPANSION_CACHE[key] = (time.time(), payload)


# ---------------------------------------------------------------------
# Condition expansion (LLM) + MedlinePlus integration
# ---------------------------------------------------------------------
async def _expand_condition_with_llm(cond: str) -> Dict[str, Any]:
    cond = (cond or "").strip()
    print(f"[cond-llm] INPUT condition brut: {repr(cond)}")

    if not cond:
        payload = {"corrected": "", "search_query": "", "synonyms": []}
        _cond_cache_set(cond, payload)
        return payload

    cached = _cond_cache_get(cond)
    if cached:
        print(f"[cond-llm] Cache hit pour {repr(cond)}: {cached}")
        return cached

    fallback_system = (
        "You are a medical terminology assistant.\n"
        "Your ONLY job is to correct spelling mistakes in a disease or condition name.\n"
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
    synonyms_clean: List[str] = []

    try:
        raw = await _llm_chat(messages_txt, max_tokens=64, temperature=0.0)
        raw = _clean_text(raw).strip()

        for line in raw.splitlines():
            line_stripped = line.strip()
            if line_stripped.upper().startswith("CORRECTED:"):
                value = line_stripped[len("CORRECTED:") :].strip()
                if value:
                    corrected = value

        print(f"[cond-llm] parsed corrected='{corrected}', synonyms={synonyms_clean}")

    except Exception as e_txt:
        print("[cond-llm] ERREUR sur le TEXT mode")
        print(f"Exception: {type(e_txt).__name__}: {e_txt}")
        traceback.print_exc()

    # Query finale (ici, pas de synonyms)
    terms: List[str] = []
    if corrected:
        terms.append(corrected)

    if not terms:
        query = cond
    else:
        query = terms[0]

    payload = {
        "corrected": corrected or cond,
        "search_query": query,
        "synonyms": [],
    }
    _cond_cache_set(cond, payload)
    return payload


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@router.get("/condition_query")
async def condition_query(
    condition: str = Query(..., min_length=2),
):
    condition = (condition or "").strip()
    if not condition:
        raise HTTPException(status_code=400, detail="Empty condition")

    print("\n========== [/condition_query] ==========")
    print(f"[cond_api] condition reçue: {repr(condition)}")

    try:
        exp = await _expand_condition_with_llm(condition)
        corrected = (exp.get("corrected") or condition).strip()
        search_query = (exp.get("search_query") or corrected).strip()
        print(f"[cond_api] corrected={repr(corrected)}, search_query={repr(search_query)}")
    except Exception as e:
        print(f"[cond_api] ERREUR _expand_condition_with_llm: {type(e).__name__}: {e}")
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
        "condition": condition,
        "corrected": corrected,
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

    # 1) Query string (client fournie ou calculée)
    try:
        if llm_query:
            search_query = (llm_query or "").strip() or condition
            print(f"[reco] llm_query fourni par le client: {repr(search_query)}")
        else:
            exp = await _expand_condition_with_llm(condition)
            search_query = (exp.get("search_query") or condition).strip()
            print(f"[reco] search_query calculée par LLM côté serveur: {repr(search_query)}")
    except Exception as e:
        print(f"[reco] ERREUR cond-expansion: {type(e).__name__}: {e}")
        traceback.print_exc()
        search_query = condition

    # 2) PubMed
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

    results = []
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

    payload = {
        "condition": condition,
        "search_query": search_query,
        "results": results,
    }
    print("========== [/recommendations END] ==========\n")
    return payload


@router.get("/explore_stream")
async def explore_stream(
    pmid: str = Query(..., min_length=1),
    plant: Optional[str] = Query(None, min_length=1),
):
    print("\n========== [/explore_stream] ==========")
    print(f"[explore_stream] pmid reçu: {repr(pmid)}, plant={repr(plant)}")

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=_HF_STREAM_TIMEOUT, limits=_LIMITS, transport=_TRANSPORT, trust_env=False) as http:
        arts = await efetch(http, [pmid])
    t1 = time.perf_counter()
    print(f"[explore_stream] efetch terminé, durée={t1-t0:.2f}s, nb_arts={len(arts)}")

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

    query = f"Key findings and limitations of: {doc['title']}"

    try:
        idx, X = _build_index(chunks)
        top = _retrieve(idx, X, chunks, query, top_k=5)
        context = "\n".join(f"{t['text']}" for t in top)
    except Exception as e:
        print(f"[explore_stream] RAG failed, fallback to full abstract: {type(e).__name__}: {e}")
        traceback.print_exc()
        context = (doc.get("abstract") or doc.get("title") or "")[:6000]

    raw_plant = (plant or "").strip()
    if raw_plant:
        parts = [p.strip() for p in raw_plant.split(",") if p.strip()]
        if len(parts) == 1:
            plant_for_prompt = parts[0]
        elif len(parts) == 2:
            plant_for_prompt = " and ".join(parts)
        else:
            plant_for_prompt = ", ".join(parts[:-1]) + " and " + parts[-1]
    else:
        plant_for_prompt = "all plants mentioned in the CONTEXT"

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
        try:
            async for chunk in _llm_chat_stream(messages, max_tokens=300, temperature=0.0):
                yield chunk
        except Exception as e:
            err = f"\n[ERROR] LLM call failed: {type(e).__name__}: {e}\n"
            print(err)
            traceback.print_exc()
            yield err

        refs = f"\n\nReferences:\nPubMed: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        yield refs

    return StreamingResponse(event_generator(), media_type="text/plain")
