#agent/intent.py
from __future__ import annotations

import re
from src.agent.schemas import Intent

# Intent = Literal["exploration","methodologie","modelisation","resultats","anthropologie","export"]

_PATTERNS: list[tuple[Intent, list[str]]] = [
    ("export", [r"\bexport\b", r"\blatex\b", r"\bpdf\b", r"\breport\b", r"\brapport\b"]),
    ("anthropologie", [r"\btodd\b", r"\banthrop", r"\bstructure(s)? famil", r"\bnormes?\b"]),
    ("resultats", [r"\bcointegr", r"\bengle\b", r"\bjohansen\b", r"\bvecm\b", r"\birf\b", r"\bfevd\b"]),
    ("modelisation", [r"\barima\b", r"\barma\b", r"\bvar\b", r"\bgranger\b", r"\bmodel", r"\bfit\b"]),
    ("methodologie", [r"\badf\b", r"\bpp\b", r"\bstation", r"\bacf\b", r"\bpacf\b", r"\btest\b"]),
    ("exploration", [r"\bexplor", r"\bplot\b", r"\bgraph", r"\bvisual", r"\bdescrib", r"\bcoverage\b", r"\bprofil"]),
]

def classify_intent(text: str) -> Intent:
    t = (text or "").lower().strip()
    if not t:
        return "exploration"
    for intent, pats in _PATTERNS:
        for p in pats:
            if re.search(p, t):
                return intent
    # fallback neutre: exploration
    return "exploration"

