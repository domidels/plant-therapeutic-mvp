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
    Interroge l'API MedlinePlus (healthTopics) et renvoie le FullSummary
    de la première maladie trouvée, ou None si absent.

    Parameters
    ----------
    disease_name : str
        Nom de la maladie (ex: "asthma", "type 2 diabetes", "migraine").
    retmax : int
        Nombre max de documents renvoyés par l'API (on prend le premier).
    timeout : int
        Timeout HTTP en secondes.

    Returns
    -------
    Optional[str]
        Le contenu de FullSummary (HTML) ou None si non trouvé.
    """
    params = {
        "db": "healthTopics",
        "term": disease_name,
        "retmax": retmax,
        # "rettype": "full",  # optionnel, par défaut l’API renvoie déjà FullSummary
    }

    resp = requests.get(MEDLINEPLUS_SEARCH_URL, params=params, timeout=timeout)
    resp.raise_for_status()

    # Parse XML
    root = ET.fromstring(resp.content)

    # Parcourt les documents dans l’ordre (rank)
    for doc in root.findall(".//document"):
        # Cherche un <content name="FullSummary"> ou "fullSummary"
        for content in doc.findall("content"):
            name = content.get("name", "")
            if name and name.lower() == "fullsummary":
                # Le contenu est du HTML sous forme de texte (déjà échappé)
                # On renvoie tel quel ; tu pourras ensuite le nettoyer ou l'afficher.
                return "".join(content.itertext()).strip()

    # Si aucun FullSummary trouvé
    return None


if __name__ == "__main__":
    summary = get_medlineplus_fullsummary("Irritable bowel syndrome")
    if summary:
        print(summary[:1000])  # on tronque juste pour l'affichage
    else:
        print("Aucun FullSummary trouvé pour cette maladie.")
