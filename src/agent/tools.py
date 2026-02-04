from __future__ import annotations

from typing import Callable, Any

import pandas as pd

from src.utils import get_logger
from src.data_pipeline.loader import load_clean_dataset
from src.data_pipeline.harmonize import harmonize
from src.data_pipeline.coverage_report import coverage_report
from src.data_pipeline.profile import profile_dataset

log = get_logger("agent.tools")

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

@register("pipeline_load_and_profile")
def pipeline_load_and_profile(*, variables: list[str], **params: Any) -> dict[str, Any]:
    """
    Charge (source unique) -> harmonise -> profile (desc/missing/coverage).
    Retourne tables+metrics sérialisables, l'executor persiste.
    """
    df_raw = load_clean_dataset()
    df = harmonize(df_raw)

    prof = profile_dataset(df, variables=variables)
    cov = coverage_report(df, variables=variables)

    tables: dict[str, pd.DataFrame] = {
        "desc_stats": prof.desc,
        "missing_report": prof.missing,
        "coverage_report": cov,
    }
    metrics = {
        "dataset_meta": prof.meta,
    }
    return {"tables": tables, "metrics": metrics}
