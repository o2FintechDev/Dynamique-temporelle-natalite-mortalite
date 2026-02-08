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
from pathlib import Path
from src.narrative import build_narrative_from_run, save_narrative_packet
from src.narrative.schema import NarrativePacket
from src.narrative.latex_renderer import render_all_section_blocks
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
    """
    Étape 4 — Univarié (ARIMA)
    Articulation: récupère automatiquement le verdict TS/DS (ADF-only) depuis le run courant
    via la métrique persistée "m.diag.ts_vs_ds" et le passe à step4_univariate_pack.
    """
    _enforce_y(y)
    df = harmonize(load_clean_dataset())

    # --- Auto-wire Step3 -> Step4 (verdict ADF-only) ---
    state = get_state()
    run_id = state.selected_run_id

    if run_id:
        p = RunManager.get_artefact_path("m.diag.ts_vs_ds", run_id=run_id)
        if p:
            tsds = read_metric_json(p)
            if isinstance(tsds, dict):
                verdict = tsds.get("verdict")
                if verdict in ("TS", "DS"):
                    params = dict(params)
                    params["ts_ds_verdict"] = verdict

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
    """
    Export LaTeX/PDF (mémoire-like) :
    - génère les blocks LaTeX par section (plan stable) depuis manifest + metrics
    - compile uniquement latex/master.tex (wrapper unique)
    - nettoie les fichiers legacy (run_root/narrative.json et latex/report.tex)
    """
    from pathlib import Path
    from src.narrative.latex_report import try_compile_pdf
    from src.narrative.latex_renderer import render_all_section_blocks

    rf = get_run_files(run_id)
    manifest = read_manifest(run_id)
    if not manifest:
        return {"metrics": {"m.report.export": {"ok": False, "error": "manifest introuvable", "run_id": run_id}}}

    run_root = Path(rf.root)

    # --- Nettoyage legacy à la racine du run ---
    legacy_narr = run_root / "narrative.json"
    if legacy_narr.exists():
        try:
            legacy_narr.unlink()
        except Exception:
            pass

    # --- Nettoyage legacy report.tex si présent ---
    legacy_report = run_root / "latex" / "report.tex"
    if legacy_report.exists():
        try:
            legacy_report.unlink()
        except Exception:
            pass

    # 1) Génère les blocks par section (latex/blocks/sec_*.tex)
    #    => chaque block inclut : intro méthodo + analyse unitaire figures/tables + conclusion (metrics)
    blocks_map = render_all_section_blocks(run_root, manifest)

    # 2) Compile uniquement le wrapper master.tex
    tex_master = run_root / "latex" / "master.tex"
    if not tex_master.exists():
        return {"metrics": {"m.report.export": {"ok": False, "error": "latex/master.tex introuvable", "run_id": run_id}}}

    pdf_path, log_text = try_compile_pdf(run_root=run_root / "latex", tex_path=tex_master, runs=1)

    out = {
        "ok": True,
        "run_id": run_id,
        "master_tex": str(tex_master),
        "blocks_generated": blocks_map,  # dict sec_key -> relpath
        "pdf_path": str(pdf_path) if pdf_path else None,
        "note": "Compilation sur latex/master.tex (plan stable via blocks sec_*.tex).",
    }
    if log_text:
        out["pdflatex_log_head"] = log_text[:2000]
    return {"metrics": {"m.report.export": out}}


@register("build_narrative")
def build_narrative(*, variables: list[str], run_id: str, **params: Any) -> dict[str, Any]:
    rf = get_run_files(run_id)
    manifest = read_manifest(run_id)
    if not manifest:
        return {"metrics": {"m.narrative.build": {"ok": False, "error": "manifest introuvable", "run_id": run_id}}}

    run_root = Path(rf.root)

    # --- CLEAN legacy narrative at run root ---
    legacy = run_root / "narrative.json"
    if legacy.exists():
        try:
            legacy.unlink()
        except Exception:
            pass

    packet = build_narrative_from_run(run_root, manifest)
    out_path = save_narrative_packet(run_root, packet)

    return {"metrics": {"m.narrative.build": {"ok": True, "run_id": run_id, "narrative_path": str(out_path)}}}