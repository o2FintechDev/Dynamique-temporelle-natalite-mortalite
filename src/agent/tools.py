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

from src.visualization.tables import save_table_csv_and_tex
from src.utils.run_writer import RunWriter
from src.utils.run_reader import get_run_files

log = get_logger("agent.tools")

TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {}

Y_CANON = "Croissance_Naturelle"
VAR_COLS_CANON = ["Croissance_Naturelle", "Nb_mariages", "IPC", "Masse_monetaire"]


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

    # --------------------------------------------------
    # Chargement et harmonisation des données
    # --------------------------------------------------
    df = harmonize(load_clean_dataset())

    requested = sorted(set([y] + VAR_COLS_CANON))
    present = [c for c in requested if c in df.columns]
    missing = [c for c in requested if c not in df.columns]

    # --------------------------------------------------
    # Profil statistique et couverture
    # --------------------------------------------------
    prof = profile_dataset(df, variables=present if present else [y])
    cov = coverage_report(df, variables=present if present else [y])

    # --------------------------------------------------
    # Méta-données et note d’audit
    # --------------------------------------------------
    miss_rate = float(prof.missing.loc[y, "missing_rate"]) if y in prof.missing.index else None

    meta = dict(prof.meta)
    meta["missing_rate"] = miss_rate
    meta["columns_available"] = list(map(str, df.columns))
    meta["requested_vars"] = requested
    meta["present_vars"] = present
    meta["missing_vars"] = missing

    note = (
        f"**Étape 1 — Traitement des données** : `{y}` chargée. "
        f"Audit des variables — présentes : {present}, manquantes : {missing}."
    ).replace("−", "-")

    # --------------------------------------------------
    # Table d’audit des variables
    # --------------------------------------------------
    tbl_vars_audit = pd.DataFrame(
        [{
            "requested": ", ".join(requested),
            "present": ", ".join(present),
            "missing": ", ".join(missing),
        }],
        index=pd.Index(["vars_audit"])
    )

    # --------------------------------------------------
    # ÉCRITURE DES TABLES LaTeX + ENREGISTREMENT MANIFEST
    # --------------------------------------------------
    state = get_state()
    run_id = state.selected_run_id

    if run_id:
        rf = get_run_files(run_id)
        run_root = Path(rf.root)
        rw = RunWriter(base_runs_dir=run_root.parent, run_id=run_id)


        run_root = rw.paths.run_dir
        tables_dir = rw.paths.tables_dir

        # --- Tableau 1 : Statistiques descriptives
        _, tex_desc = save_table_csv_and_tex(
            prof.desc,
            tables_dir / "tbl_data_desc_stats.csv",
        )
        rw.register_artefact(
            kind="tables",
            lookup_key="tbl.data.desc_stats",
            rel_path=f"artefacts/tables/{tex_desc.name}",
        )

        # --- Tableau 2 : Valeurs manquantes
        _, tex_miss = save_table_csv_and_tex(
            prof.missing,
            tables_dir / "tbl_data_missing_report.csv",
        )
        rw.register_artefact(
            kind="tables",
            lookup_key="tbl.data.missing_report",
            rel_path=f"artefacts/tables/{tex_miss.name}",
        )

        # --- Tableau 3 : Couverture temporelle
        _, tex_cov = save_table_csv_and_tex(
            cov,
            tables_dir / "tbl_data_coverage_report.csv",
        )
        rw.register_artefact(
            kind="tables",
            lookup_key="tbl.data.coverage_report",
            rel_path=f"artefacts/tables/{tex_cov.name}",
        )

        # --- Tableau 4 : Audit des variables
        _, tex_vars = save_table_csv_and_tex(
            tbl_vars_audit,
            tables_dir / "tbl_data_vars_audit.csv",
        )
        rw.register_artefact(
            kind="tables",
            lookup_key="tbl.data.vars_audit",
            rel_path=f"artefacts/tables/{tex_vars.name}",
        )

        # --------------------------------------------------
        # Retour agent (inchangé)
        # --------------------------------------------------
        return {
            "tables": {
                "tbl.data.desc_stats": prof.desc,
                "tbl.data.missing_report": prof.missing,
                "tbl.data.coverage_report": cov,
                "tbl.data.vars_audit": tbl_vars_audit,
            },
            "metrics": {
                "m.data.dataset_meta": meta,
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
    from pathlib import Path
    import time
    import re

    from src.narrative.latex_report import build_pdf
    from src.narrative.latex_renderer import build_section_blocks_from_manifest
    from src.narrative.interpretation_engine import build_snippets_from_run
    from src.narrative.tex_snippets import write_snippets
    from src.utils.run_writer import RunWriter

    # NEW: mailer (à créer dans src/utils/mailer.py)
    # def send_pdf_mail(*, to: str, subject: str, body: str, pdf_path: Path) -> None: ...
    from src.utils.mailer import send_pdf_mail

    rf = get_run_files(run_id)
    manifest = read_manifest(run_id)
    if not manifest:
        return {"metrics": {"m.report.export": {"ok": False, "error": "manifest introuvable", "run_id": run_id}}}

    run_root = Path(rf.root)

    # ---- params email (optionnels, fournis par Streamlit) ----
    to_email = (params.get("to_email") or "").strip()
    mail_subject = (params.get("mail_subject") or "Rapport économétrique (PDF)").strip()
    mail_body = (params.get("mail_body") or "Veuillez trouver le rapport en pièce jointe.").strip()

    def _is_email(s: str) -> bool:
        return bool(re.match(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$", s))

    mail_requested = bool(to_email)
    if mail_requested and not _is_email(to_email):
        return {"metrics": {"m.report.export": {
            "ok": False,
            "error": "Adresse email invalide",
            "run_id": run_id,
            "to_email": to_email,
        }}}

    # 0) Nettoyage legacy
    for p in [run_root / "narrative.json", run_root / "latex" / "report.tex"]:
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    # 1) SNIPPETS IA (artefacts/text/*.tex) AVANT blocks
    try:
        snippets = build_snippets_from_run(run_id=run_id, y=Y_CANON)
        snippets_audit = write_snippets(run_root, snippets)
    except Exception as e:
        snippets_audit = {"_error": f"{type(e).__name__}: {e}"}

    # 2) Blocks sec_*.tex
    patch_blocks = build_section_blocks_from_manifest(run_root, manifest)
    blocks_audit = (patch_blocks.get("metrics") or {}).get("m.report.blocks", {})

    # 3) Compile master.tex
    tex_master = run_root / "latex" / "master.tex"
    if not tex_master.exists():
        return {"metrics": {"m.report.export": {
            "ok": False,
            "error": "latex/master.tex introuvable",
            "run_id": run_id,
            "snippets_written": snippets_audit,
            "blocks": blocks_audit,
        }}}

    t0 = time.time()
    pdf_path, log_text = build_pdf(tex_master, runs=2)
    elapsed_s = round(time.time() - t0, 3)
    # normaliser Path
    pdf_path = Path(pdf_path) if pdf_path else None
    elapsed_s = round(time.time() - t0, 3)

    export_audit: dict[str, Any] = {
        "ok": True,
        "run_id": run_id,
        "master_tex": str(tex_master),
        "latex_dir": str(run_root / "latex"),
        "pdf_path": str(pdf_path) if pdf_path else None,
        "compile_ok": bool(pdf_path),
        "compile_elapsed_s": elapsed_s,
        "snippets_written": snippets_audit,
        "blocks": blocks_audit,
        "note": "Compilation sur latex/master.tex (blocks + snippets artefacts/text).",
        "mail_requested": mail_requested,
        "mail_sent": False,
        "to_email": to_email or None,
    }
    tail = "\n".join(log_text.splitlines()[-120:])
    export_audit["pdflatex_log_tail"] = tail
    export_audit["pdf_path"] = str(pdf_path) if pdf_path else None
    export_audit["compile_ok"] = bool(pdf_path)
    export_audit["compile_elapsed_s"] = elapsed_s
    if log_text:
        export_audit["pdflatex_log_head"] = log_text[:2000]
        export_audit["pdflatex_log_tail"] = "\n".join(log_text.splitlines()[-120:])
    # 4) Envoi mail (si demandé et PDF OK)
    if mail_requested:
        if not pdf_path:
            export_audit["mail_error"] = "PDF non généré => envoi impossible"
        else:
            try:
                send_pdf_mail(
                    to=to_email,
                    subject=mail_subject,
                    body=mail_body,
                    pdf_path=pdf_path,   # <-- déjà Path
                )
                export_audit["mail_sent"] = True
            except Exception as e:
                export_audit["mail_error"] = f"{type(e).__name__}: {e}"
    # 5) Persist audit dans manifest
    base_runs_dir = run_root.parent
    try:
        rw = RunWriter(base_runs_dir=base_runs_dir, run_id=run_id)
        rw.update_manifest({"metrics": {"m.report.export": export_audit}})
    except Exception:
        pass

    return {"metrics": {"m.report.export": export_audit}}


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