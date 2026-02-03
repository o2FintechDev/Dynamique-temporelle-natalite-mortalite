#agent/intent.py
from __future__ import annotations
import re
from src.agent.schemas import Intent

_PATTERNS: list[tuple[Intent, list[str]]] = [
    ("export", [r"\bexport\b", r"\blatex\b", r"\bpdf\b", r"\breport\b"]),
    ("explore", [r"\bexplor", r"\bplot\b", r"\bgraph", r"\bvisual", r"\bdescrib", r"\bcoverage\b"]),
    ("diagnose", [r"\badf\b", r"\bpp\b", r"\bstation", r"\bacf\b", r"\bpacf\b", r"\btest\b"]),
    ("model", [r"\barima\b", r"\bvar\b", r"\bmodel", r"\bfit\b"]),
    ("summarize", [r"\bsummar", r"\bsynth", r"\bnarrat"]),
]

def classify_intent(text: str) -> Intent:
    t = (text or "").lower().strip()
    if not t:
        return "unknown"
    for intent, pats in _PATTERNS:
        for p in pats:
            if re.search(p, t):
                return intent
    return "unknown"
