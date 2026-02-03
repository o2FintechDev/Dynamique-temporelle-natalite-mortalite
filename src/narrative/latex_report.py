from __future__ import annotations
from pathlib import Path
import csv

from src.agent.schemas import Artefact

LATEX_PREAMBLE = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{hyperref}
\geometry{margin=2.2cm}
\title{AnthroDem Lab — Rapport}
\date{}
\begin{document}
\maketitle
"""

LATEX_END = r"\end{document}"

def _csv_to_longtable(csv_path: Path, caption: str) -> str:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return ""

    header = rows[0]
    body = rows[1:]
    cols = "l" * len(header)

    lines = []
    lines.append(r"\begin{longtable}{" + cols + r"}")
    lines.append(r"\caption{" + caption + r"}\\")
    lines.append(r"\toprule")
    lines.append(" & ".join([h.replace("_", r"\_") for h in header]) + r" \\")
    lines.append(r"\midrule")
    for r in body[:500]:  # limite défensive
        lines.append(" & ".join([c.replace("_", r"\_") for c in r]) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)

def export_report_tex(run_dir: Path, run_id: str, artefacts: list[Artefact], narrative: str, audit: dict) -> Path:
    # Sélection d’artefacts clés
    cov = next((a for a in artefacts if a.name == "coverage_report" and a.kind == "table"), None)
    desc = next((a for a in artefacts if a.name == "describe_stats" and a.kind == "table"), None)
    figs = [a for a in artefacts if a.kind == "figure"]

    tex = [LATEX_PREAMBLE]
    tex.append(r"\section*{Résumé exécutif}")
    tex.append(narrative.replace("_", r"\_") if narrative else "Synthèse indisponible (audit ou absence d’artefacts).")
    tex.append(r"\subsection*{Audit}")
    tex.append(f"Audit OK: {audit.get('ok')} (phrases={audit.get('n_sentences')}).".replace("_", r"\_"))

    tex.append(r"\section*{Data Coverage Report}")
    if cov:
        tex.append(_csv_to_longtable(Path(cov.path), "Couverture des variables (start/end, manquants)"))
    else:
        tex.append("Coverage report indisponible.")

    tex.append(r"\section*{Statistiques descriptives}")
    if desc:
        tex.append(_csv_to_longtable(Path(desc.path), "Describe (base)"))
    else:
        tex.append("Describe indisponible.")

    tex.append(r"\section*{Figures clés}")
    for a in figs[:12]:
        p = Path(a.path)
        # chemin relatif (portable)
        rel = p.relative_to(run_dir)
        tex.append(r"\begin{figure}[ht]\centering")
        tex.append(r"\includegraphics[width=0.95\linewidth]{" + str(rel).replace("\\", "/").replace("_", r"\_") + r"}")
        tex.append(r"\caption{" + a.name.replace("_", r"\_") + r"}")
        tex.append(r"\end{figure}")

    tex.append(LATEX_END)

    report_path = run_dir / "report.tex"
    report_path.write_text("\n\n".join(tex), encoding="utf-8")
    return report_path
