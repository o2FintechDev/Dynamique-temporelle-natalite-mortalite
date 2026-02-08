from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from src.narrative.sections.base import (
    SectionSpec, lookup, read_json, md_basic_to_tex, escape_tex,
    include_table_tex, narr_call
)

# ... imports identiques

def render_sec_data(*, run_root: Path, manifest: Dict[str, Any], sec: SectionSpec, metrics_cache: Dict[str, Dict[str, Any]]) -> str:
    meta = metrics_cache.get("m.data.dataset_meta") or {}
    note = metrics_cache.get("m.note.step1") or {}

    freq = meta.get("freq") or meta.get("frequency") or "NA"
    start = meta.get("start") or meta.get("start_date") or "NA"
    end = meta.get("end") or meta.get("end_date") or "NA"
    nobs = meta.get("nobs") or meta.get("n") or "NA"
    missing = meta.get("missing_rate")
    missing_txt = f"{100*missing:.2f}%" if isinstance(missing, (int, float)) else "NA"

    note_md = ""
    if isinstance(note, dict):
        note_md = (note.get("markdown") or note.get("text") or note.get("summary") or "")

    t_desc = lookup(manifest, "tables", "tbl.data.desc_stats")
    t_miss = lookup(manifest, "tables", "tbl.data.missing_report")
    t_cov = lookup(manifest, "tables", "tbl.data.coverage_report")

    lines: list[str] = []
    lines += [
        r"\section{" + escape_tex(sec.title) + r"}",
        "",
        md_basic_to_tex(sec.intro_md),
        "",
        r"\subsection{Synthèse quantitative}",
        md_basic_to_tex(
            f"**Période** : **{start} -> {end}** ; **fréquence** : **{freq}** ; "
            f"**observations** : **{nobs}** ; **valeurs manquantes** : **{missing_txt}**."
        ),
        "",
        r"\subsection{Cadre, construction et définition}",
        md_basic_to_tex(
            "Le solde naturel synthétise la dynamique endogène de renouvellement de la population. "
            "La construction en taux (lorsqu’elle est disponible) vise la comparabilité intertemporelle "
            "et limite l’hétéroscédasticité liée à l’échelle démographique."
        ),
        "",
        r"\begin{equation}",
        r"Y_t = B_t - D_t",
        r"\end{equation}",
        "",
    ]

    if t_desc:
        lines += [
            r"\subsection{Tableau : tbl.data.desc\_stats}",
            include_table_tex(run_root=run_root, tbl_rel=t_desc, caption="tbl.data.desc_stats", label="tab:tbl-data-desc-stats"),
            narr_call("tbl.data.desc_stats"),
            "",
        ]
    if t_miss:
        lines += [
            r"\subsection{Tableau : tbl.data.missing\_report}",
            include_table_tex(run_root=run_root, tbl_rel=t_miss, caption="tbl.data.missing_report", label="tab:tbl-data-missing-report"),
            narr_call("tbl.data.missing_report"),
            "",
        ]
    if t_cov:
        lines += [
            r"\subsection{Tableau : tbl.data.coverage\_report}",
            include_table_tex(run_root=run_root, tbl_rel=t_cov, caption="tbl.data.coverage_report", label="tab:tbl-data-coverage-report"),
            narr_call("tbl.data.coverage_report"),
            "",
        ]

    if note_md.strip():
        lines += [
            r"\subsection{Note de contrôle (Step1)}",
            md_basic_to_tex(note_md),
            narr_call("m.note.step1"),
            "",
        ]

    lines += [
        r"\subsection{Conclusion de section}",
        md_basic_to_tex(
            "La couverture et la complétude conditionnent la puissance des tests et la stabilité des modèles. "
            "Toute discontinuité non traitée se répercute sur les résidus et fragilise les diagnostics."
        ),
        "",
    ]
    return "\n".join(lines).strip() + "\n"

