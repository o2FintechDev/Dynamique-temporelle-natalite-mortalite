# src/utils/session_state.py
from __future__ import annotations
from dataclasses import dataclass
import streamlit as st


@dataclass
class AppState:
    selected_run_id: str | None = None


def get_state() -> AppState:
    # Crée l'état une seule fois par session Streamlit
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState(selected_run_id=None)
    return st.session_state.app_state
