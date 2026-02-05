# src/narrative/latex_report.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import shutil
import subprocess
import datetime


def _escape_tex(s: str) -> str:
    if s is None:
        return ""
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def export_report_tex_from_manifest(
    *,
    run_root: str | Path,
    manifest: Dict[str, Any],
    narrative_markdown: Optional[str] = None,
    tex_name: str = "report.tex",
    title: str = "Rapport économétrique",
    author: str = "",
) -> Path:
    """
    Génère un .tex dans <run_root>/<tex_name> à partir du manifest + (optionnel) narrative_markdown.
    Signature ALIGNÉE avec src/agent/tools.py.
    """
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    tex_path = run_root / tex_name

    run_id = manifest.get("run_id", "") or manifest.get("meta", {}).get("run_id", "")
    created = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    artefacts = manifest.get("artefacts", {}) or {}
    figures = artefacts.get("figures", []) or []
    tables = artefacts.get("tables", []) or []
    metrics = artefacts.get("metrics", []) or []
    models = artefacts.get("models", []) or []

    lookup = manifest.get("lookup", {}) or {}

    def resolve_path(item: dict) -> str:
        p = item.get("path") or item.get("relpath") or ""
        if not p:
            k = item.get("key") or item.get("label")
            if k and isinstance(lookup, dict):
                # lookup typé
                for bucket in lookup.values() if any(isinstance(v, dict) for v in lookup.values()) else []:
                    if isinstance(bucket, dict) and k in bucket:
                        p = bucket[k]
                        break
                # lookup plat
                if not p and k in lookup and isinstance(lookup[k], str):
                    p = lookup[k]
        return str(p or "")

    def include_graphic(path_str: str) -> str:
        p = Path(path_str)
        abs_p = (run_root / p).resolve() if not p.is_absolute() else p
        try:
            rel = abs_p.relative_to(run_root.resolve())
            return str(rel).replace("\\", "/")
        except Exception:
            return str(abs_p).replace("\\", "/")
    
    def tex_path_str(path_str: str) -> str:
        """
        Chemin LaTeX robuste : convertit en relatif au run_root + protège caractères spéciaux.
        Utilise \detokenize{...} pour éviter les soucis avec _, %, #, espaces, etc.
        """
        p = Path(path_str)
        abs_p = (run_root / p).resolve() if not p.is_absolute() else p
        try:
            rel = abs_p.relative_to(run_root.resolve())
            rel_s = str(rel).replace("\\", "/")
        except Exception:
            rel_s = str(abs_p).replace("\\", "/")
        return r"\detokenize{" + rel_s + "}"


    # IMPORTANT: lines défini dans le scope parent
    lines: list[str] = []
    lines += [
    r"\documentclass[11pt,a4paper]{article}",
    r"\usepackage[utf8]{inputenc}",
    r"\usepackage[T1]{fontenc}",
    r"\usepackage[french]{babel}",
    r"\usepackage{geometry}",
    r"\geometry{margin=2.2cm}",
    r"\usepackage{graphicx}",
    r"\usepackage{float}",          # pour [H]
    r"\usepackage{booktabs}",
    r"\usepackage{longtable}",
    r"\usepackage{xcolor}",
    r"\usepackage{hyperref}",
    r"\usepackage{pgfplotstable}",  # pour CSV -> table
    r"\pgfplotsset{compat=1.18}",
    r"\graphicspath{{./artefacts/figures/}{./}}",
    r"\DeclareGraphicsExtensions{.pdf,.png,.jpg,.jpeg}",
    r"\begin{document}",
    r"\title{" + _escape_tex(title) + r"}",
    r"\author{" + _escape_tex(author) + r"}",
    r"\date{" + _escape_tex(created) + r"}",
    r"\maketitle",
]

    if run_id:
        lines += [r"\textbf{Run ID:} " + _escape_tex(run_id) + r"\\", ""]

    if narrative_markdown:
        lines += [r"\section{Interprétation}", ""]
        lines += [
            r"\begin{verbatim}",
            narrative_markdown,
            r"\end{verbatim}",
            "",
        ]

    def section_block(lines_ref: list[str], name: str, items: list[dict]) -> None:
        if not items:
            return

        lines_ref.append(r"\section{" + _escape_tex(name) + r"}")

        for it in items:
            label = it.get("key") or it.get("label") or it.get("name") or ""
            path_str = resolve_path(it)

            # IDs LaTeX stables
            safe_id = (label or path_str or "artefact").lower().replace(" ", "-")
            safe_id = "".join(ch for ch in safe_id if ch.isalnum() or ch in "-_")[:60]

            lines_ref.append(r"\subsection{" + _escape_tex(label or path_str or "Artefact") + r"}")

            # Affichage du chemin (optionnel) - ok en verbatim court
            if path_str:
                lines_ref.append(r"\texttt{" + _escape_tex(path_str) + r"}\\")
            else:
                lines_ref.append(r"\textit{Chemin indisponible dans le manifest.}\\")
                lines_ref.append("")
                continue

            ext = path_str.lower()

            # --- FIGURES ---
            if ext.endswith((".png", ".jpg", ".jpeg", ".pdf")):
                p_tex = tex_path_str(path_str)
                lines_ref += [
                    r"\begin{figure}[H]",
                    r"\centering",
                    rf"\IfFileExists{{{p_tex}}}{{",
                    rf"\includegraphics[width=0.95\linewidth]{{{p_tex}}}",
                    r"}{",
                    rf"\fbox{{\textbf{{Figure manquante :}} \texttt{{{_escape_tex(path_str)}}}}}",
                    r"}",
                    r"\caption{" + _escape_tex(label or "Figure") + r"}",
                    r"\label{fig:" + _escape_tex(safe_id) + r"}",
                    r"\end{figure}",
                    "",
                    r"\paragraph{Analyse automatique (IA).}",
                    r"\begin{quote}",
                    r"Référence : Figure~\ref{fig:" + _escape_tex(safe_id) + r"}.",
                    r"\end{quote}",
                    "",
                ]
                continue

            # --- TABLES ---
            # 1) si .tex -> \input
            if ext.endswith(".tex"):
                p_tex = tex_path_str(path_str)
                lines_ref += [
                    r"\begin{table}[H]",
                    r"\centering",
                    r"\caption{" + _escape_tex(label or "Tableau") + r"}",
                    r"\label{tab:" + _escape_tex(safe_id) + r"}",
                    rf"\IfFileExists{{{p_tex}}}{{",
                    rf"\input{{{p_tex}}}",
                    r"}{",
                    rf"\fbox{{\textbf{{Table manquante :}} \texttt{{{_escape_tex(path_str)}}}}}",
                    r"}",
                    r"\end{table}",
                    "",
                    r"\paragraph{Analyse automatique (IA).}",
                    r"\begin{quote}",
                    r"Référence : Tableau~\ref{tab:" + _escape_tex(safe_id) + r"}.",
                    r"\end{quote}",
                    "",
                ]
                continue

            # 2) si .csv -> pgfplotstable
            if ext.endswith(".csv"):
                p_tex = tex_path_str(path_str)
                lines_ref += [
                    r"\begin{table}[H]",
                    r"\centering",
                    r"\caption{" + _escape_tex(label or "Tableau (CSV)") + r"}",
                    r"\label{tab:" + _escape_tex(safe_id) + r"}",
                    rf"\IfFileExists{{{p_tex}}}{{",
                    r"\pgfplotstabletypeset[",
                    r"  col sep=comma,",
                    r"  string type,",
                    r"  header=true,",
                    r"  every head row/.style={before row=\toprule, after row=\midrule},",
                    r"  every last row/.style={after row=\bottomrule}",
                    r"]{" + p_tex + r"}",
                    r"}{",
                    rf"\fbox{{\textbf{{CSV manquant :}} \texttt{{{_escape_tex(path_str)}}}}}",
                    r"}",
                    r"\end{table}",
                    "",
                    r"\paragraph{Analyse automatique (IA).}",
                    r"\begin{quote}",
                    r"Référence : Tableau~\ref{tab:" + _escape_tex(safe_id) + r"}.",
                    r"\end{quote}",
                    "",
                ]
                continue

            # fallback : artefact non géré
            lines_ref.append(r"\textit{Type de fichier non géré pour rendu LaTeX.}\\")
            lines_ref.append("")



    # Appels safe
    section_block(lines, "Figures", figures)
    section_block(lines, "Tables", tables)
    section_block(lines, "Métriques", metrics)
    section_block(lines, "Modèles", models)

    lines.append(r"\end{document}")

    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return tex_path


def try_compile_pdf(
    *,
    run_root: str | Path,
    tex_path: str | Path,
    runs: int = 1,
) -> Tuple[Optional[Path], Optional[str]]:
    """
    Compile avec pdflatex si dispo.
    Retourne (pdf_path|None, log_text|None).
    """
    run_root = Path(run_root)
    tex_path = Path(tex_path)
    if not tex_path.is_absolute():
        tex_path = (run_root / tex_path).resolve()

    if shutil.which("pdflatex") is None:
        return None, None

    work_dir = tex_path.parent
    log_path = work_dir / (tex_path.stem + ".pdflatex.log")
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]

    with log_path.open("w", encoding="utf-8") as logf:
        rc = 0
        for _ in range(max(1, runs)):
            p = subprocess.run(cmd, cwd=str(work_dir), stdout=logf, stderr=logf, check=False)
            rc = p.returncode
            if rc != 0:
                break

    pdf_path = work_dir / (tex_path.stem + ".pdf")
    log_text = None
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        log_text = None

    if rc != 0 or not pdf_path.exists():
        return None, log_text

    return pdf_path, log_text
