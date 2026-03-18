import csv
from pathlib import Path
from typing import Dict, List
import unicodedata
import regex as re  # pip install regex

# ---------------------------------------------------------
#           CONTEXT & DISAMBIGUATION CONSTANTS
# ---------------------------------------------------------

CONTEXT_WORDS = {
    "leaf", "leaves", "root", "bark", "seed", "seeds", "oil", "extract", "herb",
    "powder", "capsule", "capsules", "tea", "infusion", "tincture", "ointment",
    "cream", "topical", "plant", "compound"
}

# Keywords signalling genuine toxicity / adverse outcomes
NEGATIVE_KEYWORDS = {
    "toxicity", "toxic", "poisoning", "poison", "overdose",
    "induced liver injury", "liver injury", "liver damage", "hepatotoxicity", "hepatotoxic",
    "harmful", "damage", "injury", "drug-induced",
    "herb-induced", "herbal-induced", "disorder", "death", "fatal",
    "implicating",
    "adverse drug reaction", "adverse reaction",
    "side effect",
    "risk of bleeding",
    "drug interaction",
    "safety concern",
    "fatal outcome",
    "suspected",
}

# Typical control/placebo group contexts
# NOTE: "placebo" alone is intentionally excluded to avoid false positives on arm-enumeration sentences.
CONTROL_KEYWORDS = {
    "control group", "controls", "placebo group", "vehicle", "carrier oil",
    "base oil", "odourless", "neutral oil", "standard care",
    "sham", "matched placebo"
}

# Ambiguous aliases that require surrounding context before being accepted
AMBIGUOUS = {
    "sage", "rheum", "mate", "maca", "tea", "bay", "cola", "pepper", "willow",
    "nettle", "licorice", "ginseng", "cbd"
}

# ---------------------------------------------------------
#                   GENERIC HELPERS
# ---------------------------------------------------------

def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )

def norm(s: str) -> str:
    s = (s or "").strip().casefold()
    s = strip_accents(s)
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s

def token_regex(term: str) -> str:
    """Build a robust regex pattern that matches the alias as a whole token (no substring captures)."""
    parts = [re.escape(p) for p in re.split(r"[\s\-_/]+", term) if p]
    inner = r"[-\s_]*".join(parts)
    return rf"(?<![\p{{L}}\p{{N}}]){inner}(?![\p{{L}}\p{{N}}])"

def near_context(tokens: List[str], idx: int, window: int = 4) -> bool:
    i0 = max(0, idx - window)
    i1 = min(len(tokens), idx + window + 1)
    return any(t in CONTEXT_WORDS for t in tokens[i0:i1])


# ---------------------------------------------------------
#      CSV LOADING & NESTED-PLANT CLEANUP
# ---------------------------------------------------------

def load_plants(csv_path: Path) -> List[Dict]:
    plants: List[Dict] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            can = norm(row.get("canonical", ""))
            if not can:
                continue
            al_raw = row.get("aliases", "") or ""
            aliases = {can}
            for a in al_raw.split(","):
                a = norm(a)
                if a and a != can:
                    aliases.add(a)
            plants.append({"canonical": can, "aliases": sorted(aliases)})
    return plants

def _filter_nested_plants(plants: List[str]) -> List[str]:
    """Remove canonical names that are entirely contained within a longer canonical name."""
    sorted_plants = sorted(plants, key=len, reverse=True)
    kept = []
    for p in sorted_plants:
        if any(p in q for q in kept):
            continue
        kept.append(p)
    return [p for p in plants if p in kept]


# ---------------------------------------------------------
#         CLAUSE SPLITTING FOR CONTEXT ANALYSIS
# ---------------------------------------------------------

def _split_clauses(text: str) -> List[str]:
    """
    Robust clause splitting:
      - sentence boundaries (.?!)
      - contrast connectors (while, although, whereas…)
      - handles: "While X, Y."
    """
    t = norm(text)
    sentences = re.split(r'(?<=[.!?])\s+', t)

    clauses: List[str] = []

    connector_mid = (
        r'\b(while|whereas|but|however|nevertheless|nonetheless|yet|although|though)\b'
    )
    connector_start = r'^(while|although|whereas)\b'

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        # Connector at the start of the sentence — split on the first comma
        if re.match(connector_start, s):
            idx = s.find(',')
            if idx != -1:
                first = s[:idx].strip()
                rest = s[idx+1:].strip()
                if first:
                    clauses.append(first)
                if rest:
                    clauses.append(rest)
                continue

        # Otherwise split on internal contrast connectors
        s_marked = re.sub(connector_mid, r' <SEP>\1', s)
        parts = [p.strip() for p in s_marked.split("<SEP>") if p.strip()]
        clauses.extend(parts)

    return clauses


# ---------------------------------------------------------
#      CONTEXT HEURISTICS (control / placebo group detection)
# ---------------------------------------------------------

def _is_group_enumeration_clause(c_norm: str) -> bool:
    """
    Heuristic: detect arm-enumeration clauses such as
    "participants were randomly allocated to one of three groups: X, Y, or placebo."
    Plants mentioned here should NOT be flagged as 'control'.
    """
    has_randomization = bool(
        re.search(r"\b(allocated|assigned|randomized|randomised|divided)\b", c_norm)
    )
    has_group_word = bool(
        re.search(r"\b(group|groups|arm|arms)\b", c_norm)
    )
    has_placebo = "placebo" in c_norm
    return has_randomization and has_group_word and has_placebo

def _describes_control_treatment(c_norm: str) -> bool:
    """
    Heuristic: detect clauses that describe a treatment received by a control/placebo group,
    e.g. "patients in the placebo group received neutral oil".
    """
    has_group = (
        "control group" in c_norm
        or "placebo group" in c_norm
        or "controls" in c_norm
    )
    has_treatment_verb = bool(
        re.search(r"\b(received|given|administered|treated|provided)\b", c_norm)
    )
    return has_group and has_treatment_verb


# ---------------------------------------------------------
#      CONTEXT CLASSIFICATION (toxicity / control)
# ---------------------------------------------------------

def _classify_context(text: str, plants: List[str], plant_db: List[Dict]):
    """Classify each detected plant clause by clause for toxicity and control signals."""
    clauses = _split_clauses(text)
    ctx = {p: {"negative": False, "control": False} for p in plants}

    alias_to_can = {}
    for p in plant_db:
        can = p["canonical"]
        if can not in plants:
            continue
        for a in p["aliases"]:
            alias_to_can[norm(a)] = can

    for cl in clauses:
        c_norm = norm(cl)

        plants_here = {
            can for alias, can in alias_to_can.items()
            if alias and alias in c_norm
        }
        if not plants_here:
            continue

        # Toxicity: keyword presence is sufficient
        has_neg = any(k in c_norm for k in NEGATIVE_KEYWORDS)

        # Control group: apply finer heuristics
        if _is_group_enumeration_clause(c_norm):
            # Arm enumeration (M. piperita, E. cardamomum, placebo) → not a control context
            has_ctrl = False
        elif _describes_control_treatment(c_norm):
            # Explicit description of what the placebo / control group received
            has_ctrl = True
        else:
            # Fallback: generic control keywords (neutral oil, matched placebo, etc.)
            has_ctrl = any(k in c_norm for k in CONTROL_KEYWORDS)

        for can in plants_here:
            if has_neg:
                ctx[can]["negative"] = True
            if has_ctrl:
                ctx[can]["control"] = True

    return ctx

def _filter_by_context(text: str, plants: List[str], plant_db: List[Dict]) -> List[str]:
    """
    Final filtering rules:
       - exclude plants appearing only in toxic/adverse contexts
       - exclude plants appearing only as control/placebo
       - keep everything else (no positive-signal requirement)
    """
    ctx = _classify_context(text, plants, plant_db)
    out = []

    for p in plants:
        neg  = ctx[p]["negative"]
        ctrl = ctx[p]["control"]

        if neg:
            continue
        if ctrl:
            continue

        out.append(p)

    return out


# ---------------------------------------------------------
#      MAIN PIPELINE: PLANT DETECTION IN TEXT
# ---------------------------------------------------------

def find_plants_in_text(text: str, plant_db: List[Dict]) -> List[str]:
    """
    Full detection pipeline:
      1) robust lexical matching (Latin binomials, aliases, ambiguous terms)
      2) nested-name deduplication
      3) toxicity / control filtering
    Instrumented with debug prints.
    """

    t = norm(text)
    tokens = t.split()
    hits: List[str] = []

    print("---- STEP 1: LEXICAL DETECTION ----")

    for p in plant_db:
        can = p["canonical"]
        norm_aliases = [norm(a) for a in p["aliases"] if norm(a)]

        found = False

        # Check whether any alias is a Latin binomial (two words)
        has_binomial = any(
            re.match(r"^[a-z]+(?:[\s\-_]+)[a-z]+$", a)
            for a in norm_aliases
        )

        # 1) Latin binomial takes priority
        if has_binomial:
            for term in norm_aliases:
                if re.match(r"^[a-z]+(?:[\s\-_]+)[a-z]+$", term):
                    if re.search(token_regex(term), t, flags=re.IGNORECASE):
                        print(term, "      ✔ LATIN BINOMIAL MATCH")
                        found = True
                        break

        # 2) Common aliases
        if not found:
            for term in norm_aliases:
                m = re.search(token_regex(term), t, flags=re.IGNORECASE)
                if not m:
                    continue

                base = term.split()[0] if " " in term else term

                # Ambiguous alias: surrounding context required
                if base in AMBIGUOUS or term in AMBIGUOUS:
                    start = m.start()
                    idx = len(norm(t[:start]).split())
                    print(term, f"      (ambiguous) index={idx}, checking context…")
                    if near_context(tokens, idx, window=3):
                        print("      ✔ context found → accepted")
                        found = True
                        break
                    else:
                        continue

                found = True
                break

        if found:
            hits.append(can)

    # Deduplicate while preserving order
    seen = set()
    out = []
    for h in hits:
        if h not in seen:
            out.append(h)
            seen.add(h)

    print("\n---- STEP 2: NESTED NAME REMOVAL ----")
    print("before _filter_nested_plants:", out)
    filtered_nested = _filter_nested_plants(out)
    print("after  _filter_nested_plants:", filtered_nested)

    out = filtered_nested

    # 3) Toxicity + control filter
    if out:
        print("\n---- STEP 3: TOXICITY + CONTROL FILTER ----")
        print("before _filter_by_context:", out)
        filtered_context = _filter_by_context(text, out, plant_db)
        print("after  _filter_by_context:", filtered_context)
        out = filtered_context
    else:
        print("\n---- STEP 3: (skipped) no plants to filter ----")

    print("\n==================== END DEBUG find_plants_in_text ====================\n")

    return out
