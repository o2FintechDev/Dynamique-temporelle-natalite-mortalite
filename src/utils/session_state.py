from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import streamlit as st

@dataclass
class AnthroDemState:
    current_run_id: str | None = None
    selected_run_id: str | None = None
    last_user_query: str | None = None
    last_intent: str | None = None

_KEY = "anthrodem_state"

def get_state() -> AnthroDemState:
    if _KEY not in st.session_state:
        st.session_state[_KEY] = AnthroDemState()
    return st.session_state[_KEY]

def set_current_run_id(run_id: str) -> None:
    s = get_state()
    s.current_run_id = run_id
    s.selected_run_id = run_id

def set_last_query(query: str, intent: str | None = None) -> None:
    s = get_state()
    s.last_user_query = query
    if intent is not None:
        s.last_intent = intent
