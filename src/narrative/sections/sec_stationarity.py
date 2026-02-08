from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from src.narrative.sections.base import (
    SectionSpec, lookup, md_basic_to_tex, escape_tex,
    include_table_tex, include_figure, narr_call
)

def render_sec_stationarity(*, run_root: Path, manifest: Dict[str, Any], sec: SectionSpec, metrics_cache: Dict[str, Dict[str, Any]]) -> str:
    tsds = metrics_cache.get("m.diag.ts_vs_ds") or {}
    verdict = tsds.get("verdict", "NA")
    p_c = tsds.get("adf_p_c", "NA")
    p_ct = tsds.get("adf_p_ct", "NA")

    note = (metrics_cache.get("m.note.step3") or {}).get("markdown") or ""

    lines: list[str] = [
        r"\section{" + escape_tex(sec.title) + r"}", "",
        md_basic_to_tex(sec.intro_md), "",
        r"\subsection{Synthèse quantitative}",
        md_basic_to_tex(f"Verdict ADF-only : **{verdict}** (ADF(c) p={p_c}, ADF(ct) p={p_ct})."),
        "",
    ]

    for fk in sec.figure_keys:
        rel = lookup(manifest, "figures", fk)
        if not rel:
            continue
        lines += [
            r"\subsection{Figure : " + escape_tex(fk) + r"}",
            include_figure(fig_rel=rel, caption=fk, label=f"fig:{fk}"),
            narr_call(fk),
            "",
        ]

    for tk in sec.table_keys:
        rel = lookup(manifest, "tables", tk)
        if not rel:
            continue
        lines += [
            r"\subsection{Tableau : " + escape_tex(tk) + r"}",
            include_table_tex(run_root=run_root, tbl_rel=rel, caption=tk, label=f"tab:{tk}"),
            narr_call(tk),
            "",
        ]

    lines += [
        r"\subsection{Conclusion de section}",
        md_basic_to_tex(
            f"La décision opérationnelle est **{verdict}** : elle pilote le degré d’intégration et le traitement des niveaux "
            "pour les étapes ARIMA/VAR/VECM. Un mauvais diagnostic de stationnarité contamine tous les modèles aval."
        ),
        narr_call("m.diag.ts_vs_ds"),
        "",
    ]

    if note.strip():
        lines += [md_basic_to_tex(note), narr_call("m.note.step3"), ""]

    return "\n".join(lines).strip() + "\n"
