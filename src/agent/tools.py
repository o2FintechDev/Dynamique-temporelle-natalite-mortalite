from __future__ import annotations

from typing import Callable, Any

from src.utils import get_logger

log = get_logger("agent.tools")

# IMPORTANT:
# - Mapping strict tool_name -> callable
# - Les callables doivent retourner un dict sérialisable (metrics, tables, refs, etc.)
# - L'executor gère la persistance via RunWriter (pas les pages Streamlit).

TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {}

def register(name: str) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
    def deco(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        if name in TOOL_REGISTRY:
            raise KeyError(f"Outil déjà enregistré: {name}")
        TOOL_REGISTRY[name] = fn
        return fn
    return deco

def get_tool(name: str) -> Callable[..., dict[str, Any]]:
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Outil inconnu: {name}. Disponibles: {sorted(TOOL_REGISTRY)}")
    return TOOL_REGISTRY[name]
