# app/api/routes.py
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import traceback

import httpx
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from fastembed import TextEmbedding
from app.services.medline import get_medlineplus_fullsummary
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
    "PROVIDE SIMPLE INFORMATION AS PER THE FOLLOWING INSTRUCTIONS:\n"
    "- main KEY FINDINGS related to {plant} (one or several plants) and any plant compound IF mentioned in the CONTEXT.\n"
    "- who (Women, men, children), how many particpated, the age of participants to the study ONLY IF mentioned in the CONTEXT.\n"
    "- study duration ONLY IF mentioned in the CONTEXT.\n"
    "- ANY DOSAGE, FORMULATIONS, ADMINISTRATION ROUTES ONLY IF mentioned in the CONTEXT.\n"
    "- adverse effects and limitations ONLY IF mentioned in the CONTEXT.\n"
    "\n"
    "Write ONE paragraph of 4–6 sentences.\n"
    "Use ONLY SIMPLE everyday words.\n"
    "DO NOT USE abbreviations, acronyms, or codes.\n"
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
        "options": {
            "num_ctx": num_ctx,
            "num_predict": max_tokens,
            "temperature": temperature,
            "keep_alive": "100m",
            "num_thread": max(1, os.cpu_count() // 2),
        },
    }
    async with httpx.AsyncClient(
        timeout=_HTTP_TIMEOUT,
        limits=_LIMITS,
        transport=_TRANSPORT,
        trust_env=False,
    ) as client:
        r = await client.post(f"{BASE}/v1/chat/completions", json=payload_legacy)
        r.raise_for_status()
        js = r.json()

        # format OpenAI /v1/chat/completions :
        choices = js.get("choices") if isinstance(js, dict) else None
        if isinstance(choices, list) and choices:
            first = choices[0] or {}
            msg = first.get("message") or {}
            content = msg.get("content")
            if content:
                return content

    return json.dumps(js, ensure_ascii=False)

_COND_EXPANSION_TTL = 60 * 60  # 1h pour le cache
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
    """
    Prend ce que l'utilisateur a tapé (cond) et renvoie un dict:
      {
        "corrected": "<nom corrigé pour la maladie>",
        "search_query": "<requête élargie pour PubMed>",
        "synonyms": ["..."]
      }
    """
    cond = (cond or "").strip()
    print(f"[cond-llm] INPUT condition brut: {repr(cond)}")

    if not cond:
        print("[cond-llm] condition vide → on renvoie un payload vide.")
        payload = {"corrected": "", "search_query": "", "synonyms": []}
        _cond_cache_set(cond, payload)
        return payload

    # cache éventuel
    cached = _cond_cache_get(cond)
    if cached:
        print(f"[cond-llm] Cache hit pour {repr(cond)}: {cached}")
        return cached

    # ---------- LLM en mode texte simple ----------
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

    print("[cond-llm] Messages envoyés au LLM (TEXT mode):")
    print(json.dumps(messages_txt, ensure_ascii=False, indent=2))

    corrected = cond
    synonyms_clean: List[str] = []

    try:
        raw = await _ollama_chat(
            messages_txt,
            max_tokens=256,
            num_ctx=512,
            temperature=0.0,
        )
        print(f"[cond-llm] RAW réponse LLM (TEXT) avant nettoyage: {repr(raw)}")
        raw = _clean_text(raw).strip()
        print(f"[cond-llm] RAW réponse LLM (TEXT) après _clean_text: {repr(raw)}")

        for line in raw.splitlines():
            line_stripped = line.strip()
            up = line_stripped.upper()
            if up.startswith("CORRECTED:"):
                value = line_stripped[len("CORRECTED:"):].strip()
                if value:
                    corrected = value

        print(f"[cond-llm] parsed corrected='{corrected}', synonyms={synonyms_clean}")

    except Exception as e_txt:
        print("[cond-llm] ERREUR sur le TEXT mode")
        print(f"Exception: {type(e_txt).__name__}: {e_txt}")
        traceback.print_exc()
        # on garde corrected = cond, synonyms=[]

    # ---------- Construction de la query finale ----------
    MAX_SYNONYMS = 3

    cleaned_synonyms: List[str] = []
    for s in synonyms_clean:
        if not s:
            continue
        if corrected and s.lower() == corrected.lower():
            continue
        if any(s.lower() == t.lower() for t in cleaned_synonyms):
            continue
        cleaned_synonyms.append(s)
        if len(cleaned_synonyms) >= MAX_SYNONYMS:
            break

    terms: List[str] = []
    if corrected:
        terms.append(corrected)
    terms.extend(cleaned_synonyms)

    print(f"[cond-llm] terms (corrected + synonyms uniques, limited)={terms}")

    if not terms:
        query = cond
    else:
        if len(terms) == 1:
            query = terms[0]
        else:
            query = " OR ".join(f'"{t}"' for t in terms)

    print(f"[cond-llm] OUTPUT query finale: {repr(query)}")

    payload = {
        "corrected": corrected or cond,
        "search_query": query,
        "synonyms": cleaned_synonyms,
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
    """
    Appelle seulement le LLM pour corriger la condition et élargir en requête,
    puis récupère un résumé MedlinePlus.

    Retourne:
      {
        "condition": "<tel que tapé (trim)>",
        "corrected": "<nom corrigé>",
        "search_query": "<requête élargie pour PubMed>",
        "medline_html": "<résumé HTML de la maladie si dispo>"
      }
    """
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
        print(f"[cond_api] Fallback corrected/search_query={repr(search_query)}")

    # --- Appel MedlinePlus avec le nom corrigé ---
    medline_html = ""
    try:
        # Si get_medlineplus_fullsummary est synchrone, enlève le "await".
        medline_html = await get_medlineplus_fullsummary(corrected)
        print(f"[cond_api] MedlinePlus summary OK (len={len(medline_html or '')})")
    except TypeError:
        # Cas où la fonction est synchrone (pas awaitable)
        try:
            medline_html = get_medlineplus_fullsummary(corrected)
            print(f"[cond_api] MedlinePlus summary OK (sync, len={len(medline_html or '')})")
        except Exception as e_sync:
            print(f"[cond_api] ERREUR get_medlineplus_fullsummary (sync): {type(e_sync).__name__}: {e_sync}")
            traceback.print_exc()
            medline_html = ""
    except Exception as e:
        print(f"[cond_api] ERREUR get_medlineplus_fullsummary: {type(e).__name__}: {e}")
        traceback.print_exc()
        medline_html = ""

    payload = {
        "condition": condition,          # ce que l'utilisateur a tapé
        "corrected": corrected,         # nom corrigé LLM
        "search_query": search_query,   # requête PubMed
        "medline_html": medline_html,   # résumé HTML MedlinePlus (ou "")
    }
    print(f"[cond_api] Payload final: keys={list(payload.keys())}")
    print("========== [/condition_query END] ==========\n")
    return payload


@router.get("/recommendations")
async def recommendations(
    condition: str = Query(..., min_length=2),
    from_year: Optional[int] = Query(None, ge=1800, le=3000),
    to_year: Optional[int] = Query(None, ge=1800, le=3000),
    llm_query: Optional[str] = Query(None),  # <--- nouveau
):
    print("\n========== [/recommendations] ==========")
    print(f"[reco] condition reçue (brute): {repr(condition)}")
    print(f"[reco] from_year={from_year}, to_year={to_year}")

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

    print(f"[reco] Intervalle final: {from_year}-{to_year}")

    # 1) LLM → query string (ou bien on reçoit déjà la query calculée)
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
        print(f"[reco] Fallback search_query={repr(search_query)}")

    key = (search_query.lower(), from_year, to_year)
    print(f"[reco] Cache key principal: {key}")

    # 2) PubMed
    try:
        t0 = time.perf_counter()
        print(f"[reco] Appel search_and_fetch(query={repr(search_query)}, from={from_year}, to={to_year})")
        articles = await search_and_fetch(search_query, str(from_year), str(to_year))
        t1 = time.perf_counter()
        print(
            f"[perf] recommendations PubMed={t1-t0:.2f}s "
            f"(q={repr(search_query)}, {from_year}-{to_year})"
        )
        print(f"[reco] Nombre d'articles récupérés: {len(articles)}")
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
        pmid = a.get("pmid", "?")
        title = a.get("title", "")[:80]
        print(f"[reco] Analyse article PMID={pmid}, title≈{repr(title)}")

        text = f"{a.get('title','')} {a.get('abstract','')}"
        plants_found = list(find_plants_in_text(text, PLANTS_DB))

        # Nettoyage: unique + tri pour des labels stables
        plants_unique = sorted(set(plants_found))
        print(f"[reco]  → plantes trouvées (uniques/triées): {plants_unique}")

        if not plants_unique:
            # aucun végétal pertinent → on ignore l'article pour cette vue
            continue

        if len(plants_unique) == 1:
            # Article avec UNE seule plante → on garde le comportement "par plante"
            group_label = plants_unique[0]
        else:
            # Article avec PLUSIEURS plantes → on crée un groupe combinaison
            # Exemple: "cannabidiol, ginger"
            group_label = ", ".join(plants_unique)

        grp = plant_groups.get(group_label)
        if not grp:
            grp = {
                "plants": plants_unique,  # liste des plantes de ce groupe
                "articles": [],
            }
            plant_groups[group_label] = grp

        grp["articles"].append(a)

    print(f"[reco] Nombre de groupes plante/combinaison: {len(plant_groups)}")

    results = []
    for label, grp in plant_groups.items():
        arts = grp["articles"]
        plants_list = grp["plants"]
        print(f"[reco] Traitement groupe='{label}' avec {len(arts)} articles")

        # On garde ta logique de score: somme des scores des 5 meilleurs articles du groupe
        scored = sorted(arts, key=score_article, reverse=True)
        plant_score = sum(score_article(a) for a in scored[:5])
        print(f"[reco]  → plant_score (top 5)={plant_score}")

        # On peut passer le label complet à summarize_for_patients
        # (ex: "cannabidiol, ginger") -> le prompt verra la liste de plantes.
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
                "plant": label,          # ex: "cannabidiol" ou "cannabidiol, ginger"
                "score": round(plant_score, 2),
                "summary": summary,
                "top_studies": items,
                # si un jour tu en as besoin côté front:
                # "plant_list": plants_list,
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)

    payload = {
        "condition": condition,       # ce que l'utilisateur a tapé
        "search_query": search_query, # query réelle utilisée pour PubMed
        "results": results,
    }
    print(
        f"[reco] Payload final (résumé): condition={payload['condition']}, "
        f"search_query={payload['search_query']}, "
        f"#results={len(payload['results'])}"
    )
    print("========== [/recommendations END] ==========\n")
    return payload



from fastapi.responses import StreamingResponse
import json

# # ...

@router.get("/explore_stream")
async def explore_stream(
    pmid: str = Query(..., min_length=1),
    plant: Optional[str] = Query(None, min_length=1),
):
    print("\n========== [/explore_stream] ==========")
    print(f"[explore_stream] pmid reçu: {repr(pmid)}, plant={repr(plant)}")

    t0 = time.perf_counter()
    async with httpx.AsyncClient(
        timeout=_STREAM_TIMEOUT, limits=_LIMITS, transport=_TRANSPORT, trust_env=False
    ) as http:
        print(f"[explore_stream] Appel efetch pour PMID={pmid}")
        arts = await efetch(http, [pmid])
    t1 = time.perf_counter()
    print(f"[explore_stream] efetch terminé, durée={t1-t0:.2f}s, nb_arts={len(arts)}")

    if not arts:
        print(f"[explore_stream] Aucun article trouvé pour PMID={pmid}")
        async def gen_empty():
            yield "No abstract found.\n"
            yield f"References:\nPubMed: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        print("========== [/explore_stream END] (no arts) ==========\n")
        return StreamingResponse(gen_empty(), media_type="text/plain")

    a = arts[0]
    doc = {
        "pmid": pmid,
        "title": a.get("title", ""),
        "abstract": a.get("abstract", ""),
        "journal": a.get("journal", ""),
        "year": a.get("year", ""),
    }
    print(
        f"[explore_stream] Doc récupéré pour PMID={pmid}: "
        f"title={repr(doc['title'][:80])}, year={doc['year']}, journal={repr(doc['journal'])}"
    )

    # RAG identique à /explore
    chunks = _make_corpus(doc)
    print(f"[explore_stream] Nb de chunks dans le corpus: {len(chunks)}")
    if not chunks:
        print("[explore_stream] Aucun chunk (pas de titre/abstract exploitable)")
        async def gen_no_abs():
            yield "No abstract text available.\n"
            yield f"References:\nPubMed: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        print("========== [/explore_stream END] (no chunks) ==========\n")
        return StreamingResponse(gen_no_abs(), media_type="text/plain")

    idx, X = _build_index(chunks)
    query = f"Key findings and limitations of: {doc['title']}"
    print(f"[explore_stream] RAG query: {repr(query)}")
    top = _retrieve(idx, X, chunks, query, top_k=5)
    print(f"[explore_stream] Nb de chunks top-k pour contexte: {len(top)}")

    context = "\n".join(f"{t['text']}" for t in top)
    t2 = time.perf_counter()
    print(f"[perf] (stream) efetch={t1-t0:.2f}s  rag={t2-t1:.2f}s (pmid={pmid})")
    print(
        f"[explore_stream] Longueur contexte (caractères): {len(context)}"
    )
    print(
        f"[explore_stream] Extrait contexte (200 premiers caractères): "
        f"{repr(context[:200])}"
    )
    
    raw_plant = (plant or "").strip()
    if raw_plant:
        # Exemple: "cannabidiol, ginger, turmeric"
        parts = [p.strip() for p in raw_plant.split(",") if p.strip()]
        if len(parts) == 1:
            plant_for_prompt = parts[0]                     # "ginger"
        elif len(parts) == 2:
            plant_for_prompt = " and ".join(parts)          # "cannabidiol and ginger"
        else:
            # "cannabidiol, ginger and turmeric"
            plant_for_prompt = ", ".join(parts[:-1]) + " and " + parts[-1]
    else:
        # fallback neutre quand aucune plante n’est fournie par le front
        plant_for_prompt = "all plants mentioned in the CONTEXT"

    print(f"[explore_stream] plant_for_prompt={repr(plant_for_prompt)}")

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
    print("[explore_stream] Messages envoyés au LLM (non-stream, structure):")
    print(json.dumps(messages, ensure_ascii=False, indent=2)[:2000])  # pour éviter de tout spammer

    async def event_generator():
        t3 = time.perf_counter()
        print("[explore_stream] event_generator démarré")
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
            print("[explore_stream] Payload envoyé à /v1/chat/completions (stream):")
            print(json.dumps(payload, ensure_ascii=False, indent=2)[:2000])

            async with httpx.AsyncClient(
                timeout=_STREAM_TIMEOUT, limits=_LIMITS, transport=_TRANSPORT, trust_env=False
            ) as client:
                print(f"[explore_stream] Connexion streaming vers {BASE}/v1/chat/completions")
                async with client.stream(
                    "POST",
                    f"{BASE}/v1/chat/completions",
                    json=payload,
                ) as r:
                    print(f"[explore_stream] Status LLM streaming HTTP: {r.status_code}")
                    r.raise_for_status()
                    line_count = 0
                    token_count = 0
                    async for line in r.aiter_lines():
                        line_count += 1
                        if not line:
                            continue
                        # log brut des premières lignes
                        if line_count <= 10:
                            print(f"[explore_stream] Ligne brute #{line_count}: {repr(line)}")

                        # format OpenAI-like: "data: {...}"
                        if line.startswith("data: "):
                            data = line[len("data: "):].strip()
                        else:
                            continue
                        if data == "[DONE]":
                            print("[explore_stream] Reçu [DONE] du LLM")
                            break
                        try:
                            js = json.loads(data)
                        except json.JSONDecodeError as e:
                            print(f"[explore_stream] JSONDecodeError sur chunk streaming: {e}")
                            print(f"[explore_stream]   data brut: {repr(data)[:200]}")
                            continue
                        choices = js.get("choices") or []
                        if not choices:
                            print("[explore_stream] Chunk sans choices, ignoré")
                            continue
                        delta = (choices[0].get("delta") or {}).get("content")
                        if not delta:
                            continue
                        # nettoyage léger comme _clean_text
                        chunk = _clean_text(delta)
                        if chunk:
                            token_count += 1
                            if token_count <= 10:
                                print(f"[explore_stream] Chunk texte #{token_count}: {repr(chunk)}")
                            yield chunk

        except Exception as e:
            err = f"\n[ERROR] LLM streaming failed: {type(e).__name__}: {e}\n"
            print(err)
            traceback.print_exc()
            yield err

        t4 = time.perf_counter()
        print(f"[perf] llm_stream={t4-t3:.2f}s (stream)")
        print("[explore_stream] Fin du stream, ajout des références.")

        # Ajout des références en fin de flux
        refs = f"\n\nReferences:\nPubMed: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        yield refs
        print("========== [/explore_stream END] ==========\n")

    return StreamingResponse(event_generator(), media_type="text/plain")
