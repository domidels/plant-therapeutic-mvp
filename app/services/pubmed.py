import os
import re
import random
import asyncio
from typing import List, Dict, Tuple

import httpx
from app.core.config import settings

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Optional: NCBI API key for higher rate limits
API_KEY = getattr(settings, "ncbi_api_key", None) or os.getenv("NCBI_API_KEY", None)


def _headers():
    ua = (
        settings.user_agent.format(email=settings.ncbi_email)
        if getattr(settings, "user_agent", None)
        else f"natural-therapeutics/1.0 ({settings.ncbi_email})"
    )
    return {
        "User-Agent": ua,
        "Accept": "application/json, text/xml;q=0.9, */*;q=0.8",
        "Connection": "close",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }


def _client() -> httpx.AsyncClient:
    # http2=False: NCBI sometimes drops HTTP/2 connections
    return httpx.AsyncClient(http2=False, timeout=httpx.Timeout(35.0))


def build_query(condition: str, from_year: str, to_year: str) -> str:
    # condition may contain synonyms joined by " OR "
    # e.g. "urticaria OR hives OR allergic urticaria"
    raw_terms = [t.strip() for t in condition.split(" OR ") if t.strip()]
    condition_filter = " OR ".join(
        f'"{t}"[Title] OR "{t}"[MeSH Major Topic:noexp]' for t in raw_terms
    )
    terms = [
        # Require condition (or a synonym) in the title or as primary MeSH topic.
        f'({condition_filter})',
        '(herb OR herbal OR plant OR botanical OR phytotherapy OR extract OR supplement)',
        '(randomized OR randomised OR trial OR meta-analysis OR systematic)',
    ]
    query = " AND ".join(terms)
    return f'{query} AND ("{from_year}"[PDAT] : "{to_year}"[PDAT])'


async def _sleep_backoff(attempt: int):
    # Bounded exponential backoff with jitter
    base = min(1.7**attempt, 12.0)
    await asyncio.sleep(base + random.random() * 0.4)


async def _get_with_retry(
    client: httpx.AsyncClient, url: str, params: dict, headers: dict, max_tries: int = 5
) -> httpx.Response:
    last_exc = None
    for attempt in range(max_tries):
        try:
            r = await client.get(url, params=params, headers=headers)
            if r.status_code in (429, 500, 502, 503, 504):
                await _sleep_backoff(attempt)
                continue
            r.raise_for_status()
            return r
        except (
            httpx.RemoteProtocolError,
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.WriteError,
            httpx.HTTPStatusError,
        ) as e:
            last_exc = e
            await _sleep_backoff(attempt)
            continue
    raise last_exc or RuntimeError("HTTP retry budget exhausted")


def _common_params() -> dict:
    base = {
        "tool": settings.ncbi_tool,
        "email": settings.ncbi_email,
    }
    if API_KEY:
        base["api_key"] = API_KEY
    return base


async def esearch_page(
    client: httpx.AsyncClient, query: str, retmax: int, retstart: int = 0
) -> Tuple[int, List[str]]:
    params = {
        "db": "pubmed",
        "retmode": "json",
        "term": query,
        "retmax": str(retmax),
        "retstart": str(retstart),
        "sort": "pub+date",
        **_common_params(),
    }
    r = await _get_with_retry(client, f"{BASE}/esearch.fcgi", params=params, headers=_headers())
    js = r.json()
    es = js.get("esearchresult", {}) or {}
    count = int(es.get("count", 0) or 0)
    ids = es.get("idlist", []) or []
    return count, ids


async def esearch_all(
    client: httpx.AsyncClient, query: str, page_size: int = 60, cap: int = 120
) -> List[str]:
    """
    Low cap for fast display — we want a sufficient sample to score and render.
    """
    pmids: List[str] = []
    retstart = 0
    count, ids = await esearch_page(client, query, retmax=page_size, retstart=retstart)
    pmids.extend(ids)
    retstart += len(ids)

    while retstart < count and len(pmids) < cap:
        await asyncio.sleep(0.35)  # be polite to NCBI
        _count, ids = await esearch_page(client, query, retmax=page_size, retstart=retstart)
        if not ids:
            break
        pmids.extend(ids)
        retstart += len(ids)

    return pmids[:cap]


async def efetch(client: httpx.AsyncClient, pmids: List[str]) -> List[Dict]:
    if not pmids:
        return []
    params = {
        "db": "pubmed",
        "retmode": "xml",
        "id": ",".join(pmids),
        **_common_params(),
    }
    r = await _get_with_retry(client, f"{BASE}/efetch.fcgi", params=params, headers=_headers())

    from lxml import etree

    root = etree.fromstring(r.content)
    out: List[Dict] = []
    YEAR_RE = re.compile(r"(19|20|21)\d{2}")

    for art in root.xpath(".//PubmedArticle"):

        def txt(xpath):
            node = art.xpath(xpath)
            return node[0].text.strip() if node and node[0].text else ""

        title = "".join(art.xpath(".//ArticleTitle//text()")) or ""
        abstract = " ".join(art.xpath(".//Abstract//AbstractText//text()")) or ""
        journal = txt(".//Journal/Title")
        pmid = txt(".//PMID")

        # Robust year extraction — try multiple fields
        year = txt(".//Journal/JournalIssue/PubDate/Year") or txt(".//Article/ArticleDate/Year")
        if not year:
            for st in ("entrez", "pubmed", "medline"):
                y = txt(f".//PubMedPubDate[@PubStatus='{st}']/Year")
                if y:
                    year = y
                    break
        if not year:
            md = txt(".//Journal/JournalIssue/PubDate/MedlineDate")
            if md:
                m = YEAR_RE.search(md)
                if m:
                    year = m.group(0)

        pubtype = " ".join(
            [t.text for t in art.xpath(".//PublicationTypeList/PublicationType") if t.text]
        ) or ""

        out.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "year": year or "",
                "pubtype": pubtype,
            }
        )
    return out


async def search_and_fetch(condition: str, from_year: str, to_year: str) -> List[Dict]:
    query = build_query(condition, from_year, to_year)
    async with _client() as client:
        ids = await esearch_all(client, query, page_size=60, cap=120)
        out_efetch: List[Dict] = []
        # Small batches to reduce network errors
        for i in range(0, len(ids), 20):
            subids = ids[i : i + 20]
            out_efetch.extend(await efetch(client, subids))
            await asyncio.sleep(0.2)
        return out_efetch
