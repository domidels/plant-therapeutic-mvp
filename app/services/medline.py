import requests
import xml.etree.ElementTree as ET
from typing import Optional

MEDLINEPLUS_SEARCH_URL = "https://wsearch.nlm.nih.gov/ws/query"

def get_medlineplus_fullsummary(
    disease_name: str,
    *,
    retmax: int = 1,
    timeout: int = 10
) -> Optional[str]:
    """
    Query the MedlinePlus API (healthTopics) and return the FullSummary
    of the first matching topic, or None if not found.

    Parameters
    ----------
    disease_name : str
        Name of the condition (e.g. "asthma", "type 2 diabetes", "migraine").
    retmax : int
        Maximum number of documents returned by the API (only the first is used).
    timeout : int
        HTTP timeout in seconds.

    Returns
    -------
    Optional[str]
        The FullSummary content (HTML) or None if not found.
    """
    params = {
        "db": "healthTopics",
        "term": disease_name,
        "retmax": retmax,
        # "rettype": "full",  # optional — the API already returns FullSummary by default
    }

    resp = requests.get(MEDLINEPLUS_SEARCH_URL, params=params, timeout=timeout)
    resp.raise_for_status()

    # Parse XML
    root = ET.fromstring(resp.content)

    # Iterate documents in rank order
    for doc in root.findall(".//document"):
        # Look for <content name="FullSummary"> (case-insensitive)
        for content in doc.findall("content"):
            name = content.get("name", "")
            if name and name.lower() == "fullsummary":
                # Content is HTML as plain text (already escaped) — return as-is
                return "".join(content.itertext()).strip()

    # No FullSummary found
    return None


if __name__ == "__main__":
    summary = get_medlineplus_fullsummary("Irritable bowel syndrome")
    if summary:
        print(summary[:1000])  # truncated for display
    else:
        print("No FullSummary found for this condition.")
