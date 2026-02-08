from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from src.narrative.sections.base import (
    SectionSpec, lookup, md_basic_to_tex, escape_tex,
    include_table_tex, narr_call
)

def render_sec_cointegration(*, run_root: Path, manifest: Dict[str, Any], sec: SectionSpec, metrics_cache: Dict[str, Dict[str, Any]]) -> str:
    coint = metrics_cache.get("m.coint.meta") or {}
    choice = coint.get("choice", "NA")
    rank = coint.get("rank", "NA")
    note = (metrics_cache.get("m.note.step6") or {}).get("markdown") or ""

    lines: list[str] = [
        r"\section{" + escape_tex(sec.title) + r"}", "",
        md_basic_to_tex(sec.intro_md), "",
        r"\subsection{Synthèse quantitative}",
        md_basic_to_tex(f"Choix : **{choice}** ; rang (Johansen) : **{rank}**."),
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
            f"Le module long-terme retient **{choice}**. "
            "En présence de cointégration, la dynamique de court terme doit être modélisée avec un terme de correction d’erreur (VECM)."
        ),
        narr_call("m.coint.meta"),
        "",
    ]

    if note.strip():
        lines += [md_basic_to_tex(note), narr_call("m.note.step6"), ""]

    return "\n".join(lines).strip() + "\n"
