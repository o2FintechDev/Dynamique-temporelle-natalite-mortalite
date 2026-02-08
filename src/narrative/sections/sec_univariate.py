from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from src.narrative.sections.base import (
    SectionSpec, lookup, md_basic_to_tex, escape_tex,
    include_table_tex, include_figure, narr_call
)

def render_sec_univariate(*, run_root: Path, manifest: Dict[str, Any], sec: SectionSpec, metrics_cache: Dict[str, Dict[str, Any]]) -> str:
    uni = metrics_cache.get("m.uni.best") or {}
    kp = uni.get("key_points") or {}
    order = kp.get("order") or (uni.get("best") or {}).get("order") or "NA"
    aic = kp.get("aic") or (uni.get("best") or {}).get("aic")
    bic = kp.get("bic") or (uni.get("best") or {}).get("bic")

    tsds = metrics_cache.get("m.diag.ts_vs_ds") or {}
    verdict = tsds.get("verdict", "NA")
    d_force = kp.get("d_force")
    if d_force is None:
        d_force = 1 if verdict == "DS" else 0 if verdict == "TS" else "auto"

    note = (metrics_cache.get("m.note.step4") or {}).get("markdown") or ""

    lines: list[str] = [
        r"\section{" + escape_tex(sec.title) + r"}", "",
        md_basic_to_tex(sec.intro_md), "",
        r"\subsection{Synthèse quantitative}",
        md_basic_to_tex(f"Modèle candidat : **ARIMA{order}** avec $d={d_force}$ ; AIC={aic}, BIC={bic}."),
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
            f"Le modèle **ARIMA{order}** est retenu sous la contrainte $d={d_force}$ (verdict {verdict}). "
            "La validité est jugée sur diagnostics résiduels (blancheur, normalité, hétéroscédasticité)."
        ),
        narr_call("m.uni.best"),
        "",
    ]

    if note.strip():
        lines += [md_basic_to_tex(note), narr_call("m.note.step4"), ""]

    return "\n".join(lines).strip() + "\n"
