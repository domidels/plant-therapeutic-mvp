"""
app/services/mesh.py
--------------------
Candidate synonyms for a condition from the MeSH thesaurus (NCBI E-utilities, db=mesh).

Why: LLM-generated synonyms vary between calls and "exact synonym" prompts drop
closely related forms that dominate the literature — "eczema" alone finds ~27% of
the articles that "eczema OR atopic dermatitis" finds. MeSH reliably surfaces those
forms: "Dermatitis, Atopic" lists "Atopic Eczema" among its entry terms.

MeSH alone can't decide which candidates to keep: the right term is sometimes in
another tree branch (eczema → atopic dermatitis, anxiety → generalized anxiety
disorder) while same-branch siblings can be unrelated (migraine → cluster headache).
So this module only returns a stable candidate list with PubMed counts; the caller
lets the LLM pick from that closed list (see ``routes._select_synonyms_with_llm``).
"""
from __future__ import annotations

import asyncio
import re
from typing import Dict, List, NamedTuple

from app.services.pubmed import BASE, _client, _common_params, _get_with_retry, _headers, build_query

MAX_CANDIDATES = 10

# MeSH tree categories that describe conditions: C = diseases, F01 = behavior and
# behavior mechanisms (anxiety, fatigue…), F03 = mental disorders. Excludes drugs (D),
# procedures (E), psychological scales (F04)…
_CONDITION_TREES = ("C", "F01", "F03")


class MeshCandidate(NamedTuple):
    name: str
    count: int      # articles the app's query finds for this term (all years)
    is_main: bool   # the descriptor that has the searched term as an exact entry term


def _natural_name(meshterms: List[str]) -> str:
    """Un-invert the preferred term: "Dermatitis, Atopic" → "Atopic Dermatitis".

    The preferred term (first) is used rather than the first comma-free entry term,
    which can be an obscure synonym ("Dermatitis Venenata" for contact dermatitis).
    """
    head, sep, tail = meshterms[0].partition(", ")
    return f"{tail} {head}" if sep else head


async def _pubmed_count(client, term: str) -> int:
    """Number of articles the app's own query would find for ``term`` (all years)."""
    params = {
        "db": "pubmed",
        "retmode": "json",
        "retmax": "0",
        "term": build_query(term, "1900", "3000"),
        **_common_params(),
    }
    r = await _get_with_retry(client, f"{BASE}/esearch.fcgi", params=params, headers=_headers())
    return int((r.json().get("esearchresult") or {}).get("count", 0) or 0)


async def mesh_candidates(term: str) -> List[MeshCandidate]:
    """
    Return MeSH condition descriptors whose name or entry terms contain ``term`` as a
    whole word, most productive first, excluding ``term`` itself and candidates with
    no article. ``is_main`` flags the descriptor MeSH files ``term`` under (e.g.
    "insomnia" → "sleep initiation and maintenance disorders"), which callers should
    always keep. Deterministic for a given MeSH release. Raises on network errors so
    callers can avoid caching a degraded result.
    """
    term_l = (term or "").strip().lower()
    if not term_l:
        return []
    word_re = re.compile(rf"(?<![\w-]){re.escape(term_l)}(?![\w-])", re.IGNORECASE)

    async with _client() as client:
        params = {"db": "mesh", "retmode": "json", "term": term_l, "retmax": "30", **_common_params()}
        r = await _get_with_retry(client, f"{BASE}/esearch.fcgi", params=params, headers=_headers())
        ids = (r.json().get("esearchresult") or {}).get("idlist") or []
        if not ids:
            return []

        await asyncio.sleep(0.35)  # be polite to NCBI
        params = {"db": "mesh", "retmode": "json", "id": ",".join(ids), **_common_params()}
        r = await _get_with_retry(client, f"{BASE}/esummary.fcgi", params=params, headers=_headers())
        result = r.json().get("result") or {}

        names: List[str] = []
        main_name = ""
        for i in ids:  # keep MeSH relevance order
            d = result.get(i) or {}
            if not str(d.get("ds_meshui", "")).startswith("D"):
                continue  # skip supplementary concepts / qualifiers
            meshterms = d.get("ds_meshterms") or []
            trees = [x["treenum"] for x in d.get("ds_idxlinks") or [] if x.get("treenum")]
            if not meshterms or not any(t.startswith(_CONDITION_TREES) for t in trees):
                continue
            if not any(word_re.search(t) for t in meshterms):
                continue  # matched through stemming only (e.g. "migrainous")
            name = _natural_name(meshterms).lower()
            if not main_name and term_l in (t.lower() for t in meshterms):
                main_name = name
            if name != term_l and name not in names:
                names.append(name)
        names = names[:MAX_CANDIDATES]

        counts: Dict[str, int] = {}
        for name in names:
            await asyncio.sleep(0.35)
            counts[name] = await _pubmed_count(client, name)

    found = [MeshCandidate(n, counts[n], n == main_name) for n in names if counts[n] > 0]
    return sorted(found, key=lambda c: c.count, reverse=True)
