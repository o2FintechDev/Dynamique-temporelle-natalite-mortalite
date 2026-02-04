# src/narrative/latex_report.py
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import shutil
import subprocess

import pandas as pd


LATEX_PREAMBLE = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{hyperref}
\usepackage{float}
\geometry{margin=2.2cm}
\title{AnthroDem Lab — Rapport}
\date{}
\begin{document}
\maketitle
"""

LATEX_END = r"\end{document}"


def _latex_escape(s: str) -> str:
    # Escape minimal défensif
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("$", r"\$")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _csv_to_longtable(csv_path: Path, caption: str, *, max_rows: int = 200) -> str:
    if not csv_path.exists():
        return r"\emph{Fichier CSV introuvable.}"

    df = pd.read_csv(csv_path)
    if df.empty:
        return r"\emph{Table vide.}"

    # Limite défensive
    df = df.head(max_rows)

    cols = "l" * len(df.columns)
    lines: list[str] = []
    lines.append(r"\begin{longtable}{" + cols + r"}")
    lines.append(r"\caption{" + _latex_escape(caption) + r"}\\")
    lines.append(r"\toprule")
    lines.append(" & ".join(_latex_escape(str(c)) for c in df.columns) + r" \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        lines.append(" & ".join(_latex_escape(str(v)) for v in row.values) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def export_report_tex_from_manifest(
    *,
    run_root: Path,
    manifest: dict[str, Any],
    narrative_markdown: str | None = None,
) -> Path:
    """
    Génère report.tex dans run_root à partir du manifest.
    - Inclut figures .png
    - Convertit tables .csv en longtable
    - Ajoute narrative_markdown en résumé si fourni
    """
    artefacts = manifest.get("artefacts") or []
    if not isinstance(artefacts, list):
        artefacts = []

    # Sélection par kind
    figs = [a for a in artefacts if isinstance(a, dict) and a.get("kind") == "figure"]
    tabs = [a for a in artefacts if isinstance(a, dict) and a.get("kind") == "table"]
    mets = [a for a in artefacts if isinstance(a, dict) and a.get("kind") == "metric"]

    tex: list[str] = [LATEX_PREAMBLE]

    tex.append(r"\section*{Résumé exécutif}")
    if narrative_markdown:
        # markdown -> texte latex brut (simple): on escape et on conserve les retours
        tex.append(_latex_escape(narrative_markdown))
    else:
        tex.append(r"Synthèse indisponible (aucune narration fournie).")

    tex.append(r"\section*{Manifest}")
    tex.append(r"\begin{itemize}")
    for k in ["run_id", "created_at_utc", "intent", "user_query"]:
        if k in manifest:
            tex.append(r"\item " + _latex_escape(f"{k}: {manifest.get(k)}"))
    tex.append(r"\end{itemize}")

    tex.append(r"\section*{Tables (CSV)}")
    if not tabs:
        tex.append(r"Aucune table disponible.")
    else:
        for a in tabs[:25]:  # limite défensive
            p = Path(a.get("path", ""))
            label = a.get("label") or p.stem
            tex.append(_csv_to_longtable(p, caption=f"Table: {label}"))

    tex.append(r"\section*{Figures (PNG)}")
    if not figs:
        tex.append(r"Aucune figure disponible.")
    else:
        for a in figs[:20]:
            p = Path(a.get("path", ""))
            if not p.exists():
                continue
            label = a.get("label") or p.stem

            # Chemin relatif au run_root (portable)
            try:
                rel = p.relative_to(run_root)
                rel_str = str(rel).replace("\\", "/")
            except Exception:
                rel_str = str(p).replace("\\", "/")

            tex.append(r"\begin{figure}[H]\centering")
            tex.append(r"\includegraphics[width=0.95\linewidth]{" + _latex_escape(rel_str) + r"}")
            tex.append(r"\caption{" + _latex_escape(label) + r"}")
            tex.append(r"\end{figure}")

    tex.append(r"\section*{Métriques (JSON)}")
    if not mets:
        tex.append(r"Aucune métrique disponible.")
    else:
        tex.append(r"\begin{itemize}")
        for a in mets[:50]:
            p = Path(a.get("path", ""))
            label = a.get("label") or p.stem
            tex.append(r"\item " + _latex_escape(label))
        tex.append(r"\end{itemize}")

    tex.append(LATEX_END)

    report_path = run_root / "report.tex"
    report_path.write_text("\n\n".join(tex), encoding="utf-8")
    return report_path


def try_compile_pdf(*, run_root: Path, tex_path: Path) -> tuple[Path | None, str]:
    """
    Tente pdflatex. Retourne (pdf_path, log_text).
    Fallback si pdflatex absent.
    """
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        return None, "pdflatex absent: PDF non généré (report.tex disponible)."

    cmd = [pdflatex, "-interaction=nonstopmode", tex_path.name]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(run_root),
            capture_output=True,
            text=True,
            timeout=120,
        )
        log_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        pdf_path = run_root / tex_path.with_suffix(".pdf").name
        if pdf_path.exists():
            return pdf_path, log_text
        return None, "pdflatex exécuté mais PDF introuvable.\n" + log_text
    except Exception as e:
        return None, f"Erreur compilation pdflatex: {e}"
