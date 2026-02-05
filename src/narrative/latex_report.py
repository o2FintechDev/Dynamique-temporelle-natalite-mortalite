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
        r"\usepackage{booktabs}",
        r"\usepackage{hyperref}",
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

            lines_ref.append(r"\subsection{" + _escape_tex(label or path_str) + r"}")
            if path_str:
                lines_ref.append(r"\texttt{" + _escape_tex(path_str) + r"}\\")
            else:
                lines_ref.append(r"\textit{Chemin indisponible dans le manifest.}\\")
            if path_str.lower().endswith((".png", ".jpg", ".jpeg")):
                inc = include_graphic(path_str)
                lines_ref += [
                    r"\begin{center}",
                    r"\includegraphics[width=0.95\linewidth]{" + _escape_tex(inc) + r"}",
                    r"\end{center}",
                ]
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
