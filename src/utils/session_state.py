from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import time

@dataclass
class Artifact:
    artifact_id: str
    kind: str  # "table" | "figure" | "metric" | "report" | "text"
    title: str
    payload: Any
    created_at: float = field(default_factory=lambda: time.time())

def ensure_session_state(st) -> None:
    if "history" not in st.session_state:
        st.session_state.history = []  # list[dict]: {"role": "user/agent", "content": str}
    if "artifacts" not in st.session_state:
        st.session_state.artifacts = []  # list[Artifact]
    if "last_plan" not in st.session_state:
        st.session_state.last_plan = None
