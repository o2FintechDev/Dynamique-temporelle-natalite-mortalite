from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from src.narrative.sections.base import (
    SectionSpec, lookup, md_basic_to_tex, escape_tex,
    include_table_tex, include_figure, narr_call
)

def render_sec_descriptive(*, run_root: Path, manifest: Dict[str, Any], sec: SectionSpec, metrics_cache: Dict[str, Dict[str, Any]]) -> str:
    m_strength = (metrics_cache.get("m.desc.seasonal_strength") or {}).get("value")
    m_type = (metrics_cache.get("m.desc.seasonality_type") or {}).get("value")
    note = (metrics_cache.get("m.note.step2") or {}).get("markdown") or ""

    lines: list[str] = [
        r"\section{" + escape_tex(sec.title) + r"}", "",
        md_basic_to_tex(sec.intro_md), "",
        r"\subsection{Synthèse quantitative}",
        md_basic_to_tex(f"Force saisonnière (STL) : **{m_strength}**. Qualification : **{m_type}**."),
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

    lines += [r"\subsection{Conclusion de section}"]
    if note.strip():
        lines += [md_basic_to_tex(note), narr_call("m.note.step2"), ""]
    else:
        lines += [md_basic_to_tex("La décomposition isole tendance, saisonnalité et résidu et structure les choix de modélisation."), ""]

    return "\n".join(lines).strip() + "\n"
