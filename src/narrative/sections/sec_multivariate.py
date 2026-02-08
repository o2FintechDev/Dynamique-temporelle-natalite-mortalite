from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from src.narrative.sections.base import (
    SectionSpec, lookup, md_basic_to_tex, escape_tex,
    include_table_tex, include_figure, narr_call
)

def render_sec_multivariate(*, run_root: Path, manifest: Dict[str, Any], sec: SectionSpec, metrics_cache: Dict[str, Dict[str, Any]]) -> str:
    meta = metrics_cache.get("m.var.meta") or {}
    note = (metrics_cache.get("m.note.step5") or {}).get("markdown") or ""

    lines: list[str] = [
        r"\section{" + escape_tex(sec.title) + r"}", "",
        md_basic_to_tex(sec.intro_md), "",
        r"\subsection{Synthèse quantitative}",
        md_basic_to_tex(
            "Le VAR caractérise les interdépendances dynamiques via sélection de retard, causalités de Granger, tests de Sims et FEVD."
        ),
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
            "L’analyse multivariée fournit une lecture causale (au sens prédictif) et une décomposition des chocs via IRF/FEVD. "
            "Les conclusions ne sont retenues que si les diagnostics et la spécification du retard sont cohérents."
        ),
        narr_call("m.var.meta"),
        "",
    ]

    if note.strip():
        lines += [md_basic_to_tex(note), narr_call("m.note.step5"), ""]

    return "\n".join(lines).strip() + "\n"
