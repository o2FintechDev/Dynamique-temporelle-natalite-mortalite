# agent/tools.py
from __future__ import annotations

from typing import Callable, Any

import pandas as pd

from src.utils import get_logger
from src.data_pipeline.loader import load_clean_dataset
from src.data_pipeline.harmonize import harmonize
from src.data_pipeline.coverage_report import coverage_report
from src.data_pipeline.profile import profile_dataset
from src.econometrics.api import diagnostics_pack, modelisation_pack, resultats_pack
from src.narrative.renderer import render_anthropology

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



@register("eco_diagnostics")
def eco_diagnostics(*, variables: list[str], y: str, lags: int = 24, **params: Any) -> dict[str, Any]:
    df = harmonize(load_clean_dataset())
    return diagnostics_pack(df, y=y)  # tables+metrics+figures

@register("eco_modelisation")
def eco_modelisation(*, variables: list[str], y: str, x: list[str] | None = None, with_granger: bool = True, **params: Any) -> dict[str, Any]:
    df = harmonize(load_clean_dataset())
    return modelisation_pack(df, y=y, x=x, with_granger=with_granger)

@register("eco_resultats")
def eco_resultats(*, variables: list[str], vars: list[str], **params: Any) -> dict[str, Any]:
    df = harmonize(load_clean_dataset())
    return resultats_pack(df, vars=vars)
from src.narrative.renderer import render_anthropology

@register("narrative_anthropology")
def narrative_anthropology(*, variables: list[str], y: str, **params: Any) -> dict[str, Any]:
    # faits minimaux (tirés des metrics existantes si présentes, sinon méta)
    facts = {"y": y, "note": "Analyse anthropologique basée sur artefacts de la run (offline)."}
    out = render_anthropology(facts=facts)
    return {"metrics": {"anthropology_markdown": out}}
