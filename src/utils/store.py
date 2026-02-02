from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal
import time as _time

ArtefactType = Literal["dataframe", "figure", "metrics", "text", "json"]

@dataclass
class Artefact:
    artefact_id: str
    type: ArtefactType
    title: str
    payload: Any
    created_at: float = field(default_factory=lambda: _time.time())
    meta: dict[str, Any] = field(default_factory=dict)

class SessionStore:
    """
    Stockage en mémoire (session Streamlit) des artefacts produits par l’agent.
    """
    def __init__(self) -> None:
        self.artefacts: list[Artefact] = []

    def add(self, type: ArtefactType, title: str, payload: Any, meta: dict[str, Any] | None = None) -> Artefact:
        artefact_id = f"a{len(self.artefacts)+1:04d}"
        art = Artefact(
            artefact_id=artefact_id,
            type=type,
            title=title,
            payload=payload,
            meta=meta or {},
        )
        self.artefacts.append(art)
        return art

    def get(self, artefact_id: str) -> Artefact:
        for a in self.artefacts:
            if a.artefact_id == artefact_id:
                return a
        raise KeyError(f"Artefact introuvable: {artefact_id}")

    def list(self) -> list[Artefact]:
        return list(self.artefacts)
