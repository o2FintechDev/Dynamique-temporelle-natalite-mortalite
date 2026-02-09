# src/narrative/latex_report.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import shutil
import subprocess
import datetime

from src.narrative.tex_snippets import normalize_key
from src.narrative.sections.sec_data import render_sec_data

def write_blocks_from_manifest(*, run_root: Path, manifest: Dict[str, Any], y_name: str) -> Dict[str, str]:
    blocks_dir = run_root / "latex" / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)

    # Chapitre 1
    (blocks_dir / "sec_data.tex").write_text(
        render_sec_data(run_root=run_root, manifest=manifest, y_name=y_name),
        encoding="utf-8",
    )

    # retourne éventuellement un mapping pour log/manifest
    return {"sec_data": "latex/blocks/sec_data.tex"}

def _escape_tex(s: str) -> str:
    """Escape TEX for normal text (NOT for file paths, NOT for \\label)."""
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


# ---------- small helpers ----------
def _filename_only(path_str: str) -> str:
    return Path(path_str).name


def _latex_filename(fname: str) -> str:
    # for \input/\includegraphics: keep "_" raw, avoid spaces
    return fname.replace(" ", "_")


def _safe_latex_id(label: str, path_str: str) -> str:
    # for \label{...}: strict safe charset
    base = (label or path_str or "artefact").lower().replace(" ", "-")
    base = "".join(ch for ch in base if ch.isalnum() or ch in "-_")[:60]
    return base or "artefact"


def _is_tex(path_str: str) -> bool:
    return path_str.lower().endswith(".tex")


def _is_fig(path_str: str) -> bool:
    return path_str.lower().endswith((".png", ".jpg", ".jpeg", ".pdf"))


def export_report_tex_from_manifest(
    *,
    run_root: str | Path,
    manifest: Dict[str, Any],
    narrative_markdown: Optional[str] = None,
    tex_name: str = "main.tex",  # Overleaf convention
    title: str = "Rapport économétrique",
    author: str = "",
    # NEW:
    modular: bool = True,
    narrative_snippets: bool = True,
    strict_snippet_includes: bool = False,
) -> Path:
    """
    Génère un rapport LaTeX à partir du manifest.

    MODE MODULAR :
    - écrit main.tex (preamble + \\input{sec_*.tex})
    - écrit des blocs sec_*.tex générés automatiquement
    - supporte des snippets IA en artefacts/text/<normalized_key>.tex via \\Narrative{normalized_key}

    Snippets IA:
    - narrative_snippets=True :
        - strict_snippet_includes=False -> inclusion conditionnelle (robuste)
        - strict_snippet_includes=True  -> \\input direct (échoue si absent)
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

    # Overleaf folders (fixed, relative to run_root because main.tex is written in run_root)
    OVERLEAF_FIG_DIR = "artefacts/figures"
    OVERLEAF_TAB_DIR = "artefacts/tables"
    OVERLEAF_MET_DIR = "artefacts/metrics"
    OVERLEAF_TXT_DIR = "artefacts/text"

    def resolve_path(item: dict) -> str:
        p = item.get("path") or item.get("relpath") or ""
        if not p:
            k = item.get("key") or item.get("label")
            if k and isinstance(lookup, dict):
                # lookup typé (dict of dict)
                if any(isinstance(v, dict) for v in lookup.values()):
                    for bucket in lookup.values():
                        if isinstance(bucket, dict) and k in bucket:
                            p = bucket[k]
                            break
                # lookup plat
                if not p and k in lookup and isinstance(lookup[k], str):
                    p = lookup[k]
        return str(p or "")

    def overleaf_target_path(path_str: str) -> str:
        fname = _filename_only(path_str)
        ext = path_str.lower()
        if ext.endswith((".png", ".jpg", ".jpeg", ".pdf")):
            return f"{OVERLEAF_FIG_DIR}/{fname}"
        if ext.endswith(".tex"):
            return f"{OVERLEAF_TAB_DIR}/{fname}"
        return f"{OVERLEAF_MET_DIR}/{fname}"

    # ---------- write blocks ----------
    def write_block(name: str, lines: List[str]) -> Path:
        p = run_root / name
        p.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        return p

    def narrative_macro_lines() -> List[str]:
        """
        Define \\Narrative{key} where key is already normalized (normalize_key()).
        Paths are relative to main.tex (run_root), so artefacts/text is correct.
        """
        if not narrative_snippets:
            return []
        if strict_snippet_includes:
            return [
                r"% --- IA narrative snippets (strict) ---",
                r"\newcommand{\Narrative}[1]{\input{" + f"{OVERLEAF_TXT_DIR}/" + r"#1.tex}}",
                "",
            ]
        return [
            r"% --- IA narrative snippets (robust) ---",
            r"\newcommand{\Narrative}[1]{%",
            r"  \IfFileExists{" + f"{OVERLEAF_TXT_DIR}/" + r"#1.tex}{\input{" + f"{OVERLEAF_TXT_DIR}/" + r"#1.tex}}{}%",
            r"}",
            "",
        ]

    def _narr_key(raw: str) -> str:
        # NEVER TeX-escape here. Normalize to filesystem-safe key.
        return normalize_key(raw)

    def section_figures() -> List[str]:
        if not figures:
            return []
        out: List[str] = []
        out += [r"\section{Figures}", ""]
        for it in figures:
            key = it.get("key") or it.get("label") or it.get("name") or ""
            src_path = resolve_path(it)
            safe_id = _safe_latex_id(key, src_path)
            title_txt = key or _filename_only(src_path) or "Figure"

            if not src_path:
                out += [r"\textit{Chemin indisponible dans le manifest.}\\", ""]
                continue

            target = overleaf_target_path(src_path)
            out += [r"\subsection{" + _escape_tex(title_txt) + r"}", r"\texttt{" + _escape_tex(target) + r"}\\", ""]

            if _is_fig(src_path):
                fname = _latex_filename(_filename_only(target))
                out += [
                    r"\begin{figure}[H]",
                    r"\includegraphics[width=0.95\linewidth]{" + fname + r"}",
                    r"\caption{" + _escape_tex(title_txt) + r"}",
                    r"\label{fig:" + safe_id + r"}",
                    r"\end{figure}",
                    "",
                ]
                if key:
                    out += [r"\Narrative{" + _narr_key(key) + r"}", ""]
            else:
                out += [r"\textit{Type de fichier non géré pour rendu figure.}\\", ""]

        return out

    def section_tables() -> List[str]:
        if not tables:
            return []
        out: List[str] = []
        out += [r"\section{Tables}", ""]
        for it in tables:
            key = it.get("key") or it.get("label") or it.get("name") or ""
            src_path = resolve_path(it)
            safe_id = _safe_latex_id(key, src_path)
            title_txt = key or _filename_only(src_path) or "Table"

            if not src_path:
                out += [r"\textit{Chemin indisponible dans le manifest.}\\", ""]
                continue

            target = overleaf_target_path(src_path)
            out += [r"\subsection{" + _escape_tex(title_txt) + r"}", r"\texttt{" + _escape_tex(target) + r"}\\", ""]

            if _is_tex(src_path):
                fname = _latex_filename(_filename_only(src_path))
                out += [
                    r"\begin{table}[H]",
                    r"\caption{" + _escape_tex(title_txt) + r"}",
                    r"\label{tab:" + safe_id + r"}",
                    r"\begin{adjustbox}{max width=\linewidth,center}",
                    r"\input{" + f"{OVERLEAF_TAB_DIR}/{fname}" + r"}",
                    r"\end{adjustbox}",
                    r"\end{table}",
                    "",
                ]
                if key:
                    out += [r"\Narrative{" + _narr_key(key) + r"}", ""]
            else:
                out += [r"\textit{Type de fichier non géré pour rendu table.}\\", ""]

        return out

    def section_metrics() -> List[str]:
        if not metrics:
            return []
        out: List[str] = []
        out += [r"\section{Métriques}", ""]
        for it in metrics:
            key = it.get("key") or it.get("label") or it.get("name") or ""
            src_path = resolve_path(it)
            title_txt = key or _filename_only(src_path) or "Métrique"
            out += [r"\subsection{" + _escape_tex(title_txt) + r"}"]
            if src_path:
                target = overleaf_target_path(src_path)
                out += [
                    r"\texttt{" + _escape_tex(target) + r"}\\",
                    r"\textit{Métrique non rendue automatiquement (JSON/texte).}\\",
                    "",
                ]
                if key:
                    out += [r"\Narrative{" + _narr_key(key) + r"}", ""]
            else:
                out += [r"\textit{Chemin indisponible dans le manifest.}\\", ""]
        return out

    def section_models() -> List[str]:
        if not models:
            return []
        out: List[str] = []
        out += [r"\section{Modèles}", ""]
        for it in models:
            key = it.get("key") or it.get("label") or it.get("name") or ""
            src_path = resolve_path(it)
            title_txt = key or _filename_only(src_path) or "Modèle"
            out += [r"\subsection{" + _escape_tex(title_txt) + r"}"]
            if src_path:
                out += [
                    r"\texttt{" + _escape_tex(src_path) + r"}\\",
                    r"\textit{Artefact modèle non rendu (binaire / non-LaTeX).}\\",
                    "",
                ]
                if key:
                    out += [r"\Narrative{" + _narr_key(key) + r"}", ""]
            else:
                out += [r"\textit{Chemin indisponible dans le manifest.}\\", ""]
        return out

    # ---------- main.tex ----------
    preamble: List[str] = []
    preamble += [
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
        r"\usepackage{etoolbox}",
        r"\AtBeginEnvironment{figure}{\centering}",
        r"\AtBeginEnvironment{table}{\centering}",
        r"\graphicspath{{./artefacts/figures/}}",
        r"\DeclareGraphicsExtensions{.pdf,.png,.jpg,.jpeg}",
    ]
    # define \\Narrative only once (robust or strict)
    preamble += narrative_macro_lines()

    main_lines: List[str] = []
    main_lines += preamble
    main_lines += [
        r"\begin{document}",
        r"\title{" + _escape_tex(title) + r"}",
        r"\author{" + _escape_tex(author) + r"}",
        r"\date{" + _escape_tex(created) + r"}",
        r"\maketitle",
        "",
        # ============================================================
        # PAGE DE REMERCIEMENTS (après page de garde)
        # ============================================================
        r"\clearpage",
        r"\thispagestyle{empty}",
        r"",
        r"\begin{center}",
        r"{\Large \textbf{Remerciements}}",
        r"\end{center}",
        r"",
        r"\vspace{1cm}",
        r"",
        r"Nous souhaitons adresser nos sincères remerciements à Monsieur \textbf{Mestre Roman} pour son accompagnement tout au long de nos deux années de Master. "
        r"Ses enseignements en économétrie, sa rigueur méthodologique et ses exigences académiques ont joué un rôle central dans la structuration de notre raisonnement empirique et dans la conduite de ce projet.",
        r"",
        r"Nous remercions également Monsieur \textbf{Paraguette} pour ses interventions en tant que professionnel sur les thématiques liées à l'intelligence artificielle et aux modèles de langage (IA/LLM). "
        r"Ses apports ont permis de mieux comprendre les enjeux actuels de l'automatisation de l'analyse économique et l'usage de ces outils en fin de cursus universitaire.",
        r"",
        r"Enfin, nous remercions nos camarades de promotion pour les échanges et discussions qui ont accompagné ce travail et contribué à enrichir notre réflexion collective.",
        r"",
        r"\clearpage",
        "",
    ]

    # ============================================================
    # PAGE D'AVANT-PROPOS (troisième page)
    # ============================================================
    main_lines += [
        r"\clearpage",
        r"\thispagestyle{empty}",
        r"",
        r"\begin{center}",
        r"{\Large \textbf{Avant-propos}}",
        r"\end{center}",
        r"",
        r"\vspace{1cm}",
        r"",
        r"Le présent rapport a été généré automatiquement par un automate économétrique public, développé sous forme d'une application Streamlit. "
        r"Cet automate produit l'ensemble du document en format \LaTeX, permettant une compilation et une lecture optimales via la plateforme Overleaf.",
        r"",
        r"L'intégralité des analyses, tableaux, graphiques et sections textuelles est générée de manière automatisée par cet outil. "
        r"Le système repose sur un pipeline économétrique déterministe, garantissant la reproductibilité complète des résultats présentés dans ce rapport.",
        r"",
        r"L'automate a été intégralement conçu et codé par nos soins en langage Python. "
        r"Dans le cadre de ce travail, l'intelligence artificielle a été mobilisée comme outil d'assistance, notamment via l'utilisation de ChatGPT (version 5.2), "
        r"afin de faciliter certaines étapes de structuration, de rédaction et d'interprétation, sans jamais se substituer aux choix méthodologiques ni à la conception du modèle.",
        r"",
        r"Ce projet illustre ainsi une articulation volontairement maîtrisée entre automatisation, intelligence artificielle et rigueur académique, "
        r"dans une démarche conforme aux exigences scientifiques du Master.",
        r"",
        r"\clearpage",
        "",
    ]

    if run_id:
        main_lines += [r"\textbf{Run ID:} " + _escape_tex(run_id) + r"\\", ""]

    # Optional raw narrative markdown (legacy)
    if narrative_markdown:
        main_lines += [r"\section{Interprétation (legacy)}", ""]
        main_lines += [r"\begin{verbatim}", narrative_markdown, r"\end{verbatim}", ""]

    if modular:
        blocks = [
            ("sec_figures.tex", section_figures()),
            ("sec_tables.tex", section_tables()),
            ("sec_metrics.tex", section_metrics()),
            ("sec_models.tex", section_models()),
        ]
        for fname, content in blocks:
            if content:
                write_block(fname, content)
                main_lines += [r"\input{" + fname + r"}", ""]
    else:
        content: List[str] = []
        content += section_figures()
        content += section_tables()
        content += section_metrics()
        content += section_models()
        main_lines += content

    main_lines += [r"\end{document}"]

    tex_path.write_text("\n".join(main_lines), encoding="utf-8")
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
