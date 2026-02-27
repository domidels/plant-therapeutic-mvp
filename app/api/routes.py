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
# import numpy as np  # (RAG local) plus nécessaire si on n'embarque plus embeddings+faiss sur Vercel
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

# -----------------------------
# (RAG local) IMPORTS DÉSACTIVÉS
# -----------------------------
# fastembed + faiss alourdissent énormément le bundle (limite 250MB sur Vercel serverless).
# On garde ces lignes en commentaire pour référence.
#
# from fastembed import TextEmbedding
#
# def _build_index(...):
#     import faiss
#     ...
#
# def _retrieve(...):
#     import faiss
#     ...

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
        return {"Content-Type": "application/json"}
    return {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------
# (RAG local) UTILITAIRES CONSERVÉS POUR INFO (non utilisés sans embeddings)
# ---------------------------------------------------------------------
# On garde ces helpers car ils sont utiles pour chunker/formatter un abstract,
# mais ils ne font plus de "retrieval" sans embeddings.

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


def _make_corpus(doc: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    (RAG local) À l'origine: corpus de chunks pour index faiss.
    Sans RAG, on peut encore l'utiliser pour décider de tronquer proprement.
    """
    items: List[Dict[str, str]] = []
    if doc.get("title"):
        items.append({"id": "title", "text": doc["title"]})
    if doc.get("abstract"):
        for k, ch in enumerate(_chunk_text_sentence_safe(doc["abstract"])):
            items.append({"id": f"abs_{k}", "text": ch})
    return items


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

    prompt = ""
    if system_txt:
        prompt += f"[SYSTEM]\n{system_txt}\n\n"
    if user_txt:
        prompt += f"[USER]\n{user_txt}\n\n"
    if other_txt:
        prompt += f"{other_txt}\n\n"
    prompt += "[ASSISTANT]\n"
    return prompt


def _clean_text(s: str) -> str:
    return "".join(ch for ch in s if ch == "\n" or ch == "\t" or ord(ch) >= 32)


async def asyncio_sleep(seconds: float) -> None:
    import asyncio
    await asyncio.sleep(seconds)


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

    try:
        raw = await _llm_chat(messages_txt, max_tokens=64, temperature=0.0)
        raw = _clean_text(raw).strip()

        for line in raw.splitlines():
            line_stripped = line.strip()
            if line_stripped.upper().startswith("CORRECTED:"):
                value = line_stripped[len("CORRECTED:") :].strip()
                if value:
                    corrected = value

        print(f"[cond-llm] parsed corrected='{corrected}'")

    except Exception as e_txt:
        print("[cond-llm] ERREUR sur le TEXT mode")
        print(f"Exception: {type(e_txt).__name__}: {e_txt}")
        traceback.print_exc()

    query = corrected or cond

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


# @router.get("/explore_stream")
# async def explore_stream(
#     pmid: str = Query(..., min_length=1),
#     plant: Optional[str] = Query(None, min_length=1),
# ):
#     """
#     Version SANS RAG local:
#     - On récupère l'article (title+abstract)
#     - On envoie directement l'abstract (ou un chunkage simple) au LLM HF
#     - Pas d'embeddings, pas de FAISS => beaucoup plus léger pour Vercel
#     """
#     print("\n========== [/explore_stream] ==========")
#     print(f"[explore_stream] pmid reçu: {repr(pmid)}, plant={repr(plant)}")

#     t0 = time.perf_counter()
#     async with httpx.AsyncClient(timeout=_HF_STREAM_TIMEOUT, limits=_LIMITS, transport=_TRANSPORT, trust_env=False) as http:
#         arts = await efetch(http, [pmid])
#     t1 = time.perf_counter()
#     print(f"[explore_stream] efetch terminé, durée={t1-t0:.2f}s, nb_arts={len(arts)}")

#     if not arts:
#         async def gen_empty():
#             yield "No abstract found.\n"
#             yield f"References:\nPubMed: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
#         return StreamingResponse(gen_empty(), media_type="text/plain")

#     a = arts[0]
#     doc = {
#         "pmid": pmid,
#         "title": a.get("title", ""),
#         "abstract": a.get("abstract", ""),
#         "journal": a.get("journal", ""),
#         "year": a.get("year", ""),
#     }

#     # (RAG local) On gardait des chunks + retrieval. Maintenant on fait juste un contexte simple.
#     # On garde un chunkage "safe" pour éviter d'envoyer un abstract énorme.
#     chunks = _make_corpus(doc)
#     if not chunks:
#         async def gen_no_abs():
#             yield "No abstract text available.\n"
#             yield f"References:\nPubMed: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
#         return StreamingResponse(gen_no_abs(), media_type="text/plain")

#     # --- CONTEXT (no-RAG) ---
#     # Strategy:
#     # - include title
#     # - include first N chunks of abstract (deterministic)
#     # - truncate to a safe max
#     max_chars = 6000
#     parts: List[str] = []
#     if doc["title"]:
#         parts.append(f"TITLE: {doc['title']}")
#     # add up to first 6 abstract chunks (tweak if needed)
#     abs_chunks = [c["text"] for c in chunks if c["id"].startswith("abs_")]
#     for ch in abs_chunks[:6]:
#         parts.append(ch)
#     context = "\n".join(parts)
#     context = context[:max_chars]

#     raw_plant = (plant or "").strip()
#     if raw_plant:
#         parts_pl = [p.strip() for p in raw_plant.split(",") if p.strip()]
#         if len(parts_pl) == 1:
#             plant_for_prompt = parts_pl[0]
#         elif len(parts_pl) == 2:
#             plant_for_prompt = " and ".join(parts_pl)
#         else:
#             plant_for_prompt = ", ".join(parts_pl[:-1]) + " and " + parts_pl[-1]
#     else:
#         plant_for_prompt = "all plants mentioned in the CONTEXT"

#     messages = [
#         {"role": "system", "content": _SYSTEM},
#         {
#             "role": "user",
#             "content": _USER_TMPL.format(
#                 title=doc["title"],
#                 year=doc["year"],
#                 journal=doc["journal"],
#                 context=context,
#                 pmid_study=pmid,  # (note: pas utilisé dans le template actuellement)
#                 plant=plant_for_prompt,
#             ),
#         },
#     ]

#     async def event_generator():
#         try:
#             async for chunk in _llm_chat_stream(messages, max_tokens=300, temperature=0.0):
#                 yield chunk
#         except Exception as e:
#             err = f"\n[ERROR] LLM call failed: {type(e).__name__}: {e}\n"
#             print(err)
#             traceback.print_exc()
#             yield err

#         refs = f"\n\nReferences:\nPubMed: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
#         yield refs

#     return StreamingResponse(event_generator(), media_type="text/plain")


@router.get("/explore_stream")
async def explore_stream(
    pmid: str = Query(..., min_length=1),
    plant: Optional[str] = Query(None, min_length=1),
):
    print("\n========== [/explore_stream] ==========")
    print(f"[explore_stream] pmid reçu: {repr(pmid)}, plant={repr(plant)}")

    pub_id = f"pub_{pmid}"

    # 1) Cache Turso
    try:
        cached = await get_resume_by_pub_id(pub_id)
    except Exception as e:
        print(f"[turso] cache read failed: {type(e).__name__}: {e}")
        cached = None

    if cached:
        # increment searched (best-effort)
        try:
            await increment_searched(pub_id)
        except Exception as e:
            print(f"[turso] increment failed: {type(e).__name__}: {e}")

        async def gen_cached():
            yield cached
            yield f"\n\nReferences:\nPubMed: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        return StreamingResponse(gen_cached(), media_type="text/plain")

    # 2) Sinon: ton flux actuel (efetch + LLM stream)
    async with httpx.AsyncClient(timeout=_HF_STREAM_TIMEOUT, limits=_LIMITS, transport=_TRANSPORT, trust_env=False) as http:
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

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _USER_TMPL.format(
            title=doc["title"],
            year=doc["year"],
            journal=doc["journal"],
            context=context,
            pmid_study=pmid,
            plant=plant_for_prompt,
        )},
    ]

    async def event_generator():
        # Accumule pour sauvegarder dans Turso à la fin
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

        # Ajoute refs côté client (comme avant)
        refs = f"\n\nReferences:\nPubMed: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        yield refs

        # Sauvegarde Turso (best-effort)
        full_resume = "".join(buf_parts).strip()
        if full_resume:
            try:
                await insert_resume(pub_id, full_resume)  # searched=1 via SQL
            except Exception as e:
                print(f"[turso] insert failed: {type(e).__name__}: {e}")

    return StreamingResponse(event_generator(), media_type="text/plain")