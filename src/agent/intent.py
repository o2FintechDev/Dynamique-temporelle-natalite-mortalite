from __future__ import annotations

def detect_intent(text: str) -> str:
    t = text.lower().strip()

    if any(k in t for k in ["qualité", "couverture", "trous", "missing", "completude", "complétude"]):
        return "quality"
    if any(k in t for k in ["compare", "vs", "corr", "relation", "liaison"]):
        return "compare"
    if any(k in t for k in ["synthèse", "résume", "resume", "bilan", "conclusion"]):
        return "synthesize"
    return "explore"
