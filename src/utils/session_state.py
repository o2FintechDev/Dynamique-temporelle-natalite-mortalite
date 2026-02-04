from __future__ import annotations
from dataclasses import dataclass

@dataclass
class AppState:
    selected_run_id: str | None = None

_STATE = AppState()

def get_state() -> AppState:
    return _STATE
