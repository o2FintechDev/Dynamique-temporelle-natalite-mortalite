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
    tex_name: str = "main.tex",  # Overleaf convention
    title: str = "Rapport économétrique",
    author: str = "",
) -> Path:
    """
    Génère un .tex dans <run_root>/<tex_name> à partir du manifest + (optionnel) narrative_markdown.

    CIBLE OVERLEAF (strict, sans CSV):
    - artefacts/figures/*.png
    - artefacts/tables/*.tex   (tables prêtes à \input)
    - artefacts/metrics/*.json (non rendues automatiquement ici)
    - Aucun \IfFileExists : LaTeX tente directement d’inclure.

    RÈGLE CENTRAGE:
    - Figures et tables centrées systématiquement (sans centrer tout le texte).
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

    # --- Overleaf layout (fixed folders) ---
    OVERLEAF_FIG_DIR = "artefacts/figures"
    OVERLEAF_TAB_DIR = "artefacts/tables"
    OVERLEAF_MET_DIR = "artefacts/metrics"

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

    def filename_only(path_str: str) -> str:
        return Path(path_str).name

    def latex_filename(fname: str) -> str:
        # \input/\includegraphics : garder "_" brut (pas \_) et éviter espaces
        return fname.replace(" ", "_")

    def safe_latex_id(label: str, path_str: str) -> str:
        # IMPORTANT: destiné à \label{} => doit contenir UNIQUEMENT des caractères "sûrs"
        base = (label or path_str or "artefact").lower().replace(" ", "-")
        base = "".join(ch for ch in base if ch.isalnum() or ch in "-_")[:60]
        return base or "artefact"

    def is_tex(path_str: str) -> bool:
        return path_str.lower().endswith(".tex")

    def is_fig(path_str: str) -> bool:
        return path_str.lower().endswith((".png", ".jpg", ".jpeg", ".pdf"))

    def overleaf_target_path(path_str: str) -> str:
        fname = filename_only(path_str)
        ext = path_str.lower()
        if ext.endswith((".png", ".jpg", ".jpeg", ".pdf")):
            return f"{OVERLEAF_FIG_DIR}/{fname}"
        if ext.endswith(".tex"):
            return f"{OVERLEAF_TAB_DIR}/{fname}"
        return f"{OVERLEAF_MET_DIR}/{fname}"

    lines: list[str] = []
    lines += [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[french]{babel}",
        r"\usepackage{geometry}",
        r"\geometry{margin=2.2cm}",
        r"\usepackage{float}",

        r"\usepackage{graphicx}",
        r"\setkeys{Gin}{draft=false}",

        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{adjustbox}",

        r"\usepackage{xcolor}",
        r"\usepackage{hyperref}",

        # Centrage automatique des floats (sans centrer tout le texte)
        r"\usepackage{etoolbox}",
        r"\AtBeginEnvironment{figure}{\centering}",
        r"\AtBeginEnvironment{table}{\centering}",

        # Images
        r"\graphicspath{{./artefacts/figures/}}",
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
        lines += [r"\begin{verbatim}", narrative_markdown, r"\end{verbatim}", ""]

    def section_block(lines_ref: list[str], name: str, items: list[dict]) -> None:
        if not items:
            return

        lines_ref.append(r"\section{" + _escape_tex(name) + r"}")

        for it in items:
            label = it.get("key") or it.get("label") or it.get("name") or ""
            src_path = resolve_path(it)

            safe_id = safe_latex_id(label, src_path)
            title_txt = label or filename_only(src_path) or "Artefact"
            lines_ref.append(r"\subsection{" + _escape_tex(title_txt) + r"}")

            if not src_path:
                lines_ref.append(r"\textit{Chemin indisponible dans le manifest.}\\")
                lines_ref.append("")
                continue

            target = overleaf_target_path(src_path)
            lines_ref.append(r"\texttt{" + _escape_tex(target) + r"}\\")
            lines_ref.append("")

            # --- FIGURES ---
            if is_fig(src_path):
                fname = latex_filename(filename_only(target))
                lines_ref += [
                    r"\begin{figure}[H]",
                    r"\includegraphics[width=0.95\linewidth]{" + fname + r"}",
                    r"\caption{" + _escape_tex(title_txt) + r"}",
                    # IMPORTANT: pas de _escape_tex ici (sinon \_ dans \label -> crash)
                    r"\label{fig:" + safe_id + r"}",
                    r"\end{figure}",
                    "",
                ]
                continue

            # --- TABLES (.tex) ---
            if is_tex(src_path):
                fname = latex_filename(filename_only(src_path))
                lines_ref += [
                    r"\begin{table}[H]",
                    r"\caption{" + _escape_tex(title_txt) + r"}",
                    r"\label{tab:" + safe_id + r"}",
                    r"\begin{adjustbox}{max width=\linewidth,center}",
                    # IMPORTANT: pas de \detokenize dans \input
                    r"\input{" + f"{OVERLEAF_TAB_DIR}/{fname}" + r"}",
                    r"\end{adjustbox}",
                    r"\end{table}",
                    "",
                ]
                continue

            lines_ref.append(r"\textit{Type de fichier non géré pour rendu LaTeX.}\\")
            lines_ref.append("")

    section_block(lines, "Figures", figures)
    section_block(lines, "Tables", tables)

    # Metrics: non rendues automatiquement (JSON/texte). On liste les chemins pour traçabilité.
    if metrics:
        lines.append(r"\section{Métriques}")
        for it in metrics:
            label = it.get("key") or it.get("label") or it.get("name") or ""
            src_path = resolve_path(it)
            title_txt = label or filename_only(src_path) or "Métrique"
            lines.append(r"\subsection{" + _escape_tex(title_txt) + r"}")
            if src_path:
                target = overleaf_target_path(src_path)
                lines.append(r"\texttt{" + _escape_tex(target) + r"}\\")
                lines.append(r"\textit{Métrique non rendue automatiquement (JSON/texte).}\\")
            else:
                lines.append(r"\textit{Chemin indisponible dans le manifest.}\\")
            lines.append("")

    # Models: généralement txt/pkl/json => non rendus
    if models:
        lines.append(r"\section{Modèles}")
        for it in models:
            label = it.get("key") or it.get("label") or it.get("name") or ""
            src_path = resolve_path(it)
            title_txt = label or filename_only(src_path) or "Modèle"
            lines.append(r"\subsection{" + _escape_tex(title_txt) + r"}")
            if src_path:
                lines.append(r"\texttt{" + _escape_tex(src_path) + r"}\\")
                lines.append(r"\textit{Artefact modèle non rendu (binaire / non-LaTeX).}\\")
            else:
                lines.append(r"\textit{Chemin indisponible dans le manifest.}\\")
            lines.append("")

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
