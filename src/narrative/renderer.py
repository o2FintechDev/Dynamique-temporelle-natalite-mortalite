from __future__ import annotations
from typing import Any
import pandas as pd

from src.utils import SessionStore

from .rules import narrative_from_artefacts

def build_evidence_bundle(store: SessionStore) -> dict[str, Any]:
    bundle: dict[str, Any] = {}

    # récupérer derniers artefacts "kind"
    wide = next((a for a in reversed(store.list()) if a.meta.get("kind") == "wide"), None)
    cov = next((a for a in reversed(store.list()) if a.meta.get("kind") == "coverage"), None)
    desc = next((a for a in reversed(store.list()) if a.meta.get("kind") == "describe"), None)

    if wide:
        bundle["wide_id"] = wide.artefact_id
        bundle["wide"] = wide.payload
    if cov:
        bundle["coverage_id"] = cov.artefact_id
        bundle["coverage"] = cov.payload
    if desc:
        bundle["describe_id"] = desc.artefact_id
        bundle["describe"] = desc.payload

    return bundle

def render_constrained_narrative(store: SessionStore) -> str:
    bundle = build_evidence_bundle(store)
    lines = narrative_from_artefacts(bundle)

    # audit: chaque phrase doit contenir un tag [aXXXX]
    for ln in lines:
        if "[" not in ln or "]" not in ln:
            raise ValueError("Audit narration échoué: phrase sans référence artefact.")
    return "\n".join(f"- {ln}" for ln in lines)
