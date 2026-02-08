# src/narrative/tex_include.py
from __future__ import annotations
from pathlib import Path

def _looks_like_full_table(tex: str) -> bool:
    t = tex.lower()
    return ("\\begin{table" in t) or ("\\end{table" in t) or ("\\begin{longtable" in t) or ("\\end{longtable" in t)

def include_table_tex(*, run_root: Path, tbl_rel: str, caption: str, label: str) -> str:
    """
    Inclusion robuste d'un artefact .tex.
    - Si le .tex contient déjà un environnement (table/longtable) => input direct (pas de wrapper).
    - Sinon => wrapper table + adjustbox + caption/label.
    """
    tbl_path = (run_root / tbl_rel).resolve()
    try:
        content = tbl_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        content = ""

    fname = Path(tbl_rel).name.replace(" ", "_")

    if _looks_like_full_table(content):
        # Type A : déjà un environnement complet
        return "\n".join([
            r"\input{../artefacts/tables/" + fname + r"}",
            "",
        ])

    # Type B : fragment => wrapper standardisé
    return "\n".join([
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\begin{adjustbox}{max width=\linewidth,center}",
        r"\input{../artefacts/tables/" + fname + r"}",
        r"\end{adjustbox}",
        r"\end{table}",
        "",
    ])
