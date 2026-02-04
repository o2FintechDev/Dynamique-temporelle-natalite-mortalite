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
from src.utils.session_state import get_state
from src.utils.run_reader import read_manifest, RunManager, read_metric_json
from pathlib import Path
from src.utils.run_reader import get_run_files, read_manifest, RunManager, read_metric_json
from src.narrative.latex_report import export_report_tex_from_manifest, try_compile_pdf


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
    """
    Lit les metrics du run courant (lookup) et produit un JSON audité.
    Persisté par l'Executor via label: 'todd_analysis'.
    """
    state = get_state()
    run_id = state.selected_run_id
    if not run_id:
        facts = {"y": y, "note": "Aucun run sélectionné: impossible de lire les métriques.", "_refs": []}
        out = render_anthropology(facts=facts)
        return {"metrics": {"todd_analysis": out}}

    manifest = read_manifest(run_id)
    lookup = manifest.get("lookup") or {}

    refs: list[str] = []
    facts: dict[str, Any] = {"y": y, "_refs": refs}

    # 1) Récupérer metric ARIMA best si présent
    best_label = f"univariate_best_{y}"
    p_best = RunManager.get_artefact_path(best_label, run_id=run_id)
    if p_best:
        m = read_metric_json(p_best)
        refs.append(best_label)
        # structure: {"best": {...}, ...} ou {"univariate_best_...": ...} selon implémentation
        best = m.get("best") if isinstance(m, dict) else None
        if isinstance(best, dict):
            facts["ARIMA_best_order"] = best.get("order")
            facts["ARIMA_best_aic"] = best.get("aic")
            facts["ARIMA_best_bic"] = best.get("bic")
        else:
            # fallback: dump brut limité
            facts["ARIMA_metric"] = str(m)[:400]

    # 2) VAR meta (si tu l’as dans metrics)
    # run_var_pack produit "var_meta" (label exact)
    p_var_meta = RunManager.get_artefact_path("var_meta", run_id=run_id)
    if p_var_meta:
        refs.append("var_meta")
        vm = read_metric_json(p_var_meta)
        if isinstance(vm, dict):
            facts["VAR_selected_lag_aic"] = vm.get("selected_lag_aic")
            facts["VAR_vars"] = vm.get("vars")
            facts["VAR_nobs"] = vm.get("nobs")

    # 3) Cointegration meta (si présent)
    p_coint_meta = RunManager.get_artefact_path("cointegration_meta", run_id=run_id)
    if p_coint_meta:
        refs.append("cointegration_meta")
        cm = read_metric_json(p_coint_meta)
        if isinstance(cm, dict):
            facts["Cointegration_vars"] = cm.get("vars")
            facts["Cointegration_nobs"] = cm.get("nobs")

    # 4) Impulse meta (si présent)
    p_imp_meta = RunManager.get_artefact_path("impulse_meta", run_id=run_id)
    if p_imp_meta:
        refs.append("impulse_meta")
        im = read_metric_json(p_imp_meta)
        if isinstance(im, dict):
            facts["IRF_selected_lag_aic"] = im.get("selected_lag_aic")
            facts["IRF_horizon"] = im.get("horizon")

    # Si rien trouvé: note explicite
    if len(refs) == 0:
        facts["note"] = "Aucune métrique clé trouvée dans lookup. Lancer une run Modélisation/Résultats avant Anthropologie."

    out = render_anthropology(facts=facts)
    return {"metrics": {"todd_analysis": out}}

@register("export_latex_pdf")
def export_latex_pdf(*, variables: list[str], run_id: str, **params: Any) -> dict[str, Any]:
    """
    Parcourt manifest['artefacts'] du run_id et génère report.tex + tentative report.pdf.
    Persisté par l'Executor sous forme de metric: export_report
    """
    rf = get_run_files(run_id)
    manifest = read_manifest(run_id)
    if not manifest:
        return {"metrics": {"export_report": {"ok": False, "error": "manifest introuvable", "run_id": run_id}}}

    # narrative si existante (todd_analysis)
    narrative_md = None
    p_todd = RunManager.get_artefact_path("todd_analysis", run_id=run_id)
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
    # log séparé pour ne pas exploser le JSON
    if log_text:
        out["pdflatex_log_head"] = log_text[:2000]

    return {"metrics": {"export_report": out}}
