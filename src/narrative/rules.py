# src/narrative/rules.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Sentence:
    text: str
    artefact_ids: list[str]

def render_sentence(s: Sentence) -> str:
    # Format d’audit strict: chaque phrase porte ses références
    refs = " ".join([f"[artefact:{aid}]" for aid in s.artefact_ids])
    return f"{s.text} {refs}".strip()

def audit_narrative(text: str, known_ids: set[str]) -> tuple[bool, list[str]]:
    # Chaque phrase doit contenir au moins un [artefact:...]
    problems: list[str] = []
    for i, line in enumerate([l.strip() for l in text.split("\n") if l.strip()], start=1):
        if "[artefact:" not in line:
            problems.append(f"Ligne {i}: pas de référence artefact.")
        else:
            # Vérifier que les ids référencés existent
            parts = [p for p in line.split("[artefact:") if "]" in p]
            for p in parts[1:]:
                aid = p.split("]")[0].strip()
                if aid not in known_ids:
                    problems.append(f"Ligne {i}: référence inconnue {aid}.")
    return (len(problems) == 0), problems
