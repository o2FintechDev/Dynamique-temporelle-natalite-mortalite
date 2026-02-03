from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class AppSession:
    current_run_id: str | None = None
    last_plan: dict[str, Any] | None = None
    last_narrative: str | None = None
    last_artefacts: list[dict[str, Any]] = field(default_factory=list)

def ensure_session_state(st) -> None:
    if "anthrodem" not in st.session_state:
        st.session_state["anthrodem"] = AppSession()

def get_session(st) -> AppSession:
    ensure_session_state(st)
    return st.session_state["anthrodem"]
