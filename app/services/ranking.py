from typing import Dict, List

def score_article(a: Dict) -> float:
    t = f"{a.get('title','')} {a.get('abstract','')} {a.get('pubtype','')}".lower()
    score = 0.0
    if "meta-analysis" in t or "systematic review" in t: score += 5
    if "randomized" in t or "randomised" in t: score += 4
    if "clinical trial" in t: score += 2
    if "cohort" in t or "case-control" in t: score += 1
    # slight recency bump
    try:
        y = int(a.get("year") or 0)
        score += max(0, (y - 2015) * 0.2)
    except: pass
    return score

def summarize_for_patients(plant: str, articles: List[Dict]) -> str:
    # ultra-simple bullet style (no LLM)
    if not articles:
        return f"No strong human evidence found yet for {plant} in this condition. Studies may be limited or preliminary."
    # heuristic signals
    has_meta = any("meta-analysis" in (a["pubtype"]+a["title"]).lower() for a in articles)
    has_rct  = any(("randomized" in (a["title"]+a["abstract"]).lower()) for a in articles)
    lines = []
    if has_meta: lines.append("• Includes systematic reviews/meta-analyses.")
    if has_rct:  lines.append("• Includes randomized clinical trials.")
    lines.append("• See key studies below; discuss with your clinician before use.")
    return " ".join(lines)
