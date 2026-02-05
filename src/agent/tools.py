# src/agent/tools.py
from __future__ import annotations
from typing import Callable, Any
import pandas as pd
import matplotlib.figure as mpl_fig

from src.utils.logger import get_logger
from src.data_pipeline.loader import load_clean_dataset
from src.data_pipeline.harmonize import harmonize
from src.data_pipeline.coverage_report import coverage_report
from src.data_pipeline.profile import profile_dataset

from src.econometrics.api import (
    step2_descriptive_pack,
    step3_stationarity_pack,
    step4_univariate_pack,
    step5_var_pack,
    step6_cointegration_pack,
)

from src.narrative.renderer import render_anthropology
from src.utils.session_state import get_state
from src.utils.run_reader import get_run_files, read_manifest, RunManager, read_metric_json


log = get_logger("agent.tools")

TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {}

Y_CANON = "Croissance_Naturelle"

def _enforce_y(y: str) -> None:
    if y != Y_CANON:
        raise ValueError(f"Variable cible interdite: {y}. Seule variable autorisée: {Y_CANON}")

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

@register("step1_load_and_profile")
def step1_load_and_profile(*, variables: list[str], y: str, **params: Any) -> dict[str, Any]:
    _enforce_y(y)
    df = harmonize(load_clean_dataset())

    prof = profile_dataset(df, variables=[y])
    cov = coverage_report(df, variables=[y])

    # ---- NOTE STEP1 (persistée) ----
    miss = float(prof.missing.loc[y, "missing_rate"]) if y in prof.missing.index else None
    nobs = int(prof.meta.get("nobs", len(df)))
    start = prof.meta.get("start")
    end = prof.meta.get("end")
    note = (
        f"**Étape 1 — Traitement des données** : série `{y}` construite (taux_naissances − taux_deces), "
        f"mensuelle 1975–2025. Observations: **{nobs}**, période: **{start} → {end}**, "
        f"taux de valeurs manquantes: **{miss:.2%}**."
    )

    return {
        "tables": {
            "tbl.data.desc_stats": prof.desc,
            "tbl.data.missing_report": prof.missing,
            "tbl.data.coverage_report": cov,
        },
        "metrics": {
            "m.data.dataset_meta": prof.meta,
            "m.note.step1": {"markdown": note},
        },
    }

@register("step2_descriptive")
def step2_descriptive(*, variables: list[str], y: str, **params: Any) -> dict[str, Any]:
    _enforce_y(y)
    df = harmonize(load_clean_dataset())
    return step2_descriptive_pack(df, y=y, **params)

@register("step3_stationarity")
def step3_stationarity(*, variables: list[str], y: str, lags: int = 24, **params: Any) -> dict[str, Any]:
    _enforce_y(y)
    df = harmonize(load_clean_dataset())
    return step3_stationarity_pack(df, y=y, lags=lags, **params)

@register("step4_univariate")
def step4_univariate(*, variables: list[str], y: str, **params: Any) -> dict[str, Any]:
    _enforce_y(y)
    df = harmonize(load_clean_dataset())
    return step4_univariate_pack(df, y=y, **params)

@register("step5_var")
def step5_var(*, variables: list[str], y: str, vars_mode: str = "decomp", **params: Any) -> dict[str, Any]:
    _enforce_y(y)
    df = harmonize(load_clean_dataset())
    return step5_var_pack(df, y=y, vars_mode=vars_mode, **params)

@register("step6_cointegration")
def step6_cointegration(*, variables: list[str], y: str, vars_mode: str = "decomp", **params: Any) -> dict[str, Any]:
    _enforce_y(y)
    df = harmonize(load_clean_dataset())
    return step6_cointegration_pack(df, y=y, vars_mode=vars_mode, **params)

@register("step7_anthropology")
def step7_anthropology(*, variables: list[str], y: str, **params: Any) -> dict[str, Any]:
    _enforce_y(y)
    state = get_state()
    run_id = state.selected_run_id
    refs: list[str] = []
    facts: dict[str, Any] = {"y": y, "_refs": refs}

    if not run_id:
        facts["note"] = "Aucun run sélectionné."
        out = render_anthropology(facts=facts)
        return {"metrics": {"m.anthro.todd_analysis": out}}

    def _try(label: str) -> dict[str, Any] | None:
        p = RunManager.get_artefact_path(label, run_id=run_id)
        if not p:
            return None
        refs.append(label)
        return read_metric_json(p)

    for k in ["m.diag.ts_vs_ds", "m.uni.best", "m.var.meta", "m.coint.meta"]:
        v = _try(k)
        if v:
            facts[k] = v

    out = render_anthropology(facts=facts)
    return {"metrics": {"m.anthro.todd_analysis": out}}

@register("export_latex_pdf")
def export_latex_pdf(*, variables: list[str], run_id: str, **params: Any) -> dict[str, Any]:
    from src.narrative.latex_report import export_report_tex_from_manifest, try_compile_pdf
    rf = get_run_files(run_id)
    manifest = read_manifest(run_id)
    if not manifest:
        return {"metrics": {"m.report.export": {"ok": False, "error": "manifest introuvable", "run_id": run_id}}}

    narrative_md = None
    p_todd = RunManager.get_artefact_path("m.anthro.todd_analysis", run_id=run_id)
    if p_todd:
        payload = read_metric_json(p_todd)
        narrative_md = payload.get("markdown")

    tex_path = export_report_tex_from_manifest(run_root=rf.root, manifest=manifest, narrative_markdown=narrative_md)
    pdf_path, log_text = try_compile_pdf(run_root=rf.root, tex_path=tex_path)

    out = {
        "ok": True,
        "run_id": run_id,
        "tex_path": str(tex_path),
        "pdf_path": str(pdf_path) if pdf_path else None,
        "note": "PDF compilé si pdflatex disponible; sinon report.tex seulement.",
    }
    if log_text:
        out["pdflatex_log_head"] = log_text[:2000]
    return {"metrics": {"m.report.export": out}}
