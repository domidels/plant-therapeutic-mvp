import csv
from pathlib import Path
from typing import Dict, List
import unicodedata
import regex as re  # pip install regex

# Mots-contexte utiles pour lever les ambiguïtés (sage, tea, etc.)
CONTEXT_WORDS = {
    "leaf","leaves","root","bark","seed","seeds","oil","extract","herb",
    "powder","capsule","capsules","tea","infusion","tincture","ointment","cream","topical"
}

# Alias ambigus qui exigent contexte ou binôme latin
AMBIGUOUS = {"sage","rheum","mate","maca","tea","bay","cola","pepper","willow","nettle","licorice","ginseng"}

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def norm(s: str) -> str:
    s = (s or "").strip().casefold()
    s = strip_accents(s)
    s = s.replace("’","'").replace("–","-").replace("—","-")
    s = re.sub(r"\s+", " ", s)
    return s

def load_plants(csv_path: Path) -> List[Dict]:
    """
    Charge le CSV au format: canonical,aliases
    - normalise en minuscules
    - split des aliases sur la virgule
    - dédoublonne les aliases
    """
    plants: List[Dict] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            can = norm(row.get("canonical",""))
            if not can:
                continue
            al_raw = row.get("aliases","") or ""
            aliases = {can}
            for a in al_raw.split(","):
                a = norm(a)
                if a and a != can:
                    aliases.add(a)
            plants.append({"canonical": can, "aliases": sorted(aliases)})
    return plants

def token_regex(term: str) -> str:
    """
    Construit un motif robuste:
    - tolère espaces/tirets/underscores entre tokens
    - impose bornes non-alphanum (\p{L}\p{N}) aux extrémités
    """
    parts = [re.escape(p) for p in re.split(r"[\s\-_/]+", term) if p]
    inner = r"[-\s_]*".join(parts)
    return rf"(?<![\p{{L}}\p{{N}}]){inner}(?![\p{{L}}\p{{N}}])"

def near_context(tokens: List[str], idx: int, window: int = 4) -> bool:
    i0 = max(0, idx - window)
    i1 = min(len(tokens), idx + window + 1)
    return any(t in CONTEXT_WORDS for t in tokens[i0:i1])

def find_plants_in_text(text: str, plant_db: List[Dict]) -> List[str]:
    """
    - insensible à la casse/accents
    - évite les sous-chaînes internes (bornes de mots Unicode)
    - accepte tirets/espaces (evening-primrose ~ evening primrose)
    - limite les alias ambigus via contexte proche ou binôme latin
    """
    t = norm(text)
    tokens = t.split()
    hits: List[str] = []

    for p in plant_db:
        found = False
        norm_aliases = [norm(a) for a in p["aliases"] if norm(a)]
        has_binomial = any(re.match(r"^[a-z]+(?:\s+|-|_)[a-z]+$", a) for a in norm_aliases)

        # 1) binôme latin prioritaire (très spécifique)
        if has_binomial:
            for term in norm_aliases:
                if re.match(r"^[a-z]+(?:\s+|-|_)[a-z]+$", term):
                    if re.search(token_regex(term), t, flags=re.IGNORECASE):
                        found = True
                        break

        # 2) sinon/ensuite alias communs avec anti-ambiguïté
        if not found:
            for term in norm_aliases:
                m = re.search(token_regex(term), t, flags=re.IGNORECASE)
                if not m:
                    continue
                base = term.split()[0] if " " in term else term

                if base in AMBIGUOUS or term in AMBIGUOUS:
                    # Contexte requis autour du match ambigu
                    start = m.start()
                    idx = len(norm(t[:start]).split())
                    if near_context(tokens, idx, window=3):
                        found = True
                        break
                    else:
                        continue
                else:
                    found = True
                    break

        if found:
            hits.append(p["canonical"])

    # dédoublonne en conservant l'ordre
    seen = set(); out = []
    for h in hits:
        if h not in seen:
            out.append(h); seen.add(h)
    return out
