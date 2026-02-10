# src/utils/run_writer.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import uuid

from filelock import FileLock

def sanitize_tex(text: str) -> str:
    """
    Nettoie les caractères invisibles et dangereux pour LaTeX.
    """
    if not text:
        return ""
    t = text.replace("\x07", "").replace("\x08", "")
    # kill common mojibake sequences if they appear
    t = t.replace("Ã©", "é").replace("Ã¨", "è").replace("Ãª", "ê").replace("Ã ", "à")
    t = t.replace("Ã´", "ô").replace("Ã¹", "ù").replace("Ã¢", "â").replace("Ã®", "î")
    t = t.replace("Ã§", "ç").replace("Â", "")
    return t


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k not in out:
            out[k] = v
            continue
        if isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        elif isinstance(out[k], list) and isinstance(v, list):
            out[k] = out[k] + v
        else:
            out[k] = v
    return out


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    artefacts_dir: Path
    figures_dir: Path
    tables_dir: Path
    metrics_dir: Path
    models_dir: Path
    logs_dir: Path
    manifest_path: Path
    latex_dir: Path
    latex_master_path: Path


class RunWriter:
    """
    app/outputs/runs/<run_id>/
      manifest.json
      artefacts/{figures,tables,metrics,models}/...
      latex/{blocks/*.tex, master.tex}
      logs/...

    NOTE: narrative.json legacy supprimé. Source of truth narrative = artefacts/narrative/narrative.json (géré ailleurs).
    """

    def __init__(self, base_runs_dir: Path, run_id: str) -> None:
        self.base_runs_dir = base_runs_dir
        self.run_id = run_id
        self.paths = self._init_dirs()
        self._manifest_lock = FileLock(str(self.paths.manifest_path.with_suffix(".lock")))

    @classmethod
    def create_new(cls, base_runs_dir: Path) -> "RunWriter":
        run_id = f"{_utc_ts()}_{uuid.uuid4().hex[:10]}"
        return cls(base_runs_dir, run_id)

    def _init_dirs(self) -> RunPaths:
        run_dir = self.base_runs_dir / self.run_id
        artefacts_dir = run_dir / "artefacts"
        figures_dir = artefacts_dir / "figures"
        tables_dir = artefacts_dir / "tables"
        metrics_dir = artefacts_dir / "metrics"
        models_dir = artefacts_dir / "models"
        logs_dir = run_dir / "logs"
        latex_dir = run_dir / "latex"
        blocks_dir = latex_dir / "blocks"

        run_dir.mkdir(parents=True, exist_ok=True)
        for d in [artefacts_dir, figures_dir, tables_dir, metrics_dir, models_dir, logs_dir, latex_dir, blocks_dir]:
            d.mkdir(parents=True, exist_ok=True)

        manifest_path = run_dir / "manifest.json"
        latex_master_path = latex_dir / "master.tex"

        # --- Init manifest (obligatoire) ---
        if not manifest_path.exists():
            self._write_json(
                manifest_path,
                {
                    "run_id": self.run_id,
                    "created_at": _utc_ts(),
                    "updated_at": _utc_ts(),
                    "lookup": {"figures": {}, "tables": {}, "metrics": {}, "models": {}},
                    "artefacts": {"figures": [], "tables": [], "metrics": [], "models": []},
                    "steps": {},
                },
            )

        # --- Init master.tex ---
        if not latex_master_path.exists():
            latex_master_path.write_text(
                sanitize_tex(self._latex_master_skeleton()),
                encoding="utf-8"
            )

        return RunPaths(
            run_dir=run_dir,
            artefacts_dir=artefacts_dir,
            figures_dir=figures_dir,
            tables_dir=tables_dir,
            metrics_dir=metrics_dir,
            models_dir=models_dir,
            logs_dir=logs_dir,
            manifest_path=manifest_path,
            latex_dir=latex_dir,
            latex_master_path=latex_master_path,
        )

    def _latex_master_skeleton(self) -> str:
        return r"""\documentclass[11pt,a4paper]{report}

\usepackage[utf8]{inputenc}
\usepackage{newunicodechar}
\newunicodechar{−}{-}

% --- Unicode math fallbacks ---
\DeclareUnicodeCharacter{2248}{\ensuremath{\approx}}
\DeclareUnicodeCharacter{21D2}{\ensuremath{\Rightarrow}}
\DeclareUnicodeCharacter{2265}{\ensuremath{\geq}}
\DeclareUnicodeCharacter{03B1}{\ensuremath{\alpha}}
\DeclareUnicodeCharacter{03B2}{\ensuremath{\beta}}

\usepackage[T1]{fontenc}
\usepackage[french]{babel}

\usepackage{geometry}
\geometry{margin=2.5cm}
\usepackage{setspace}
\onehalfspacing

\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{adjustbox}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{pdflscape}
\usepackage{etoolbox}
\AtBeginEnvironment{figure}{\centering}
\AtBeginEnvironment{table}{\centering}

% IMPORTANT: compilation depuis run_root/latex => artefacts est au niveau parent
\graphicspath{{../artefacts/figures/}{./}}

% --- IA narrative snippets (robust) ---
\newcommand{\Narrative}[1]{%
  \IfFileExists{../artefacts/text/#1.tex}{\input{../artefacts/text/#1.tex}}{}%
}

% =================================================
% PAGE DE GARDE
% =================================================


\begin{document}

\begin{titlepage}
\thispagestyle{empty}

% -------------------------------------------------
% LOGOS (haut de page)
% -------------------------------------------------
\begin{minipage}{0.48\textwidth}
    \raggedright
    \includegraphics[width=0.70\textwidth]{image1.png}
\end{minipage}
\hfill
\begin{minipage}{0.48\textwidth}
    \raggedleft
    \includegraphics[width=0.40\textwidth]{image2.png}
\end{minipage}

\vspace*{3cm}

% -------------------------------------------------
% BLOC CENTRAL (centrage visuel)
% -------------------------------------------------
\vfill

\begin{center}
    {\Large Analyse économétrique automatisée}\\[0.3cm]
    {\Large par pipeline déterministe}\\[0.8cm]

    {\Huge\bfseries Automate économétrique}\\[0.4cm]
    {\LARGE\bfseries Croissance naturelle en France (1975--2025)}\\[1.5cm]

    {\large Aude \textsc{Bernier}}\\
    {\large Clara \textsc{Pierreuse}}\\
    {\large Justine \textsc{Reiter--Guerville}}\\[1.5cm]

    {\large Master Monnaie Banque Finance Assurance}\\
    {\itshape Parcours Systèmes d'Information Économiques et Financiers (SIEF)}\\[0.3cm]
    Université de Montpellier\\[0.6cm]
    Année universitaire 2025--2026
\end{center}


\vfill

% -------------------------------------------------
% ENCADREMENT (bas à gauche)
% -------------------------------------------------
\begin{flushleft}
{\small Sous la supervision de :\\
Monsieur Mestre Roman\\
Monsieur Paraguette Maurice}
\end{flushleft}

\end{titlepage}

% =================================================
% REMERCIEMENTS
% =================================================
\clearpage
\thispagestyle{empty}

\begin{center}
{\Large \textbf{Remerciements}}
\end{center}

\vspace{1cm}

Nous souhaitons adresser nos sincères remerciements à Monsieur \textbf{Mestre Roman}
pour son accompagnement tout au long de nos deux années de Master.
Ses enseignements en économétrie, sa rigueur méthodologique et ses exigences académiques
ont joué un rôle central dans la structuration de notre raisonnement empirique
et dans la conduite de ce projet.

Nous remercions également Monsieur \textbf{Paraguette Maurice}
pour ses interventions en tant que professionnel sur les thématiques liées à
l'intelligence artificielle et aux modèles de langage (IA/LLM).
Ses apports ont permis de mieux comprendre les enjeux actuels de l'automatisation
de l'analyse économique en fin de cursus universitaire.

Enfin, nous remercions nos camarades de promotion pour les échanges et discussions
qui ont accompagné ce travail et contribué à enrichir notre réflexion collective.

\clearpage

% =================================================
% AVANT-PROPOS
% =================================================
\thispagestyle{empty}

\begin{center}
{\Large \textbf{Avant-propos}}
\end{center}

\vspace{1cm}

Le présent rapport a été généré automatiquement par un automate économétrique public,
développé sous forme d'une application Streamlit.
Cet automate produit l'ensemble du document en format \LaTeX,
permettant une compilation et une lecture optimales via la plateforme Overleaf.

L'intégralité des analyses, tableaux, graphiques et sections textuelles
est générée de manière automatisée par cet outil,
reposant sur un pipeline économétrique déterministe
garantissant la reproductibilité complète des résultats.

L'automate a été intégralement conçu et codé par nos soins en langage Python.
L'intelligence artificielle a été mobilisée comme outil d'assistance,
notamment via l'utilisation de ChatGPT (version 5.2),
afin de faciliter certaines étapes de structuration,
de rédaction et d'interprétation,
sans jamais se substituer aux choix méthodologiques et scientifiques.

Ce travail illustre ainsi une articulation maîtrisée entre automatisation,
intelligence artificielle et rigueur académique,
conforme aux exigences du Master.

\clearpage

% =================================================
% ABSTRACT
% =================================================
\clearpage
\thispagestyle{empty}

\begin{center}
{\Large \textbf{Automate économétrique appliqué à la croissance naturelle en France (1975--2025)}}\\[0.4cm]
{\large \textit{Analyse automatisée par pipeline économétrique déterministe}}\\[1cm]
\end{center}

{\footnotesize
Aude Bernier, Clara Pierreuse, Justine Reiter--Guerville\\
Master Monnaie Banque Finance Assurance -- Parcours Systèmes d’Information Économiques et Financiers (SIEF)\\
Université de Montpellier
}

\vspace{1cm}

\noindent\textbf{Abstract}

\vspace{0.3cm}

\noindent
Ce rapport présente une analyse économétrique automatisée de la croissance naturelle de la population française sur la période 1975--2025, définie comme la différence entre les taux de natalité et de mortalité. L’étude repose sur des données mensuelles officielles issues de l’INSEE et s’inscrit dans une démarche de reproductibilité complète grâce à l’utilisation d’un pipeline économétrique entièrement automatisé.

Le cadre méthodologique combine une analyse descriptive approfondie, des décompositions saisonnières, des tests de stationnarité, une modélisation univariée des séries temporelles ainsi qu’une analyse dynamique multivariée. Les choix de modèles et les diagnostics sont réalisés de manière systématique et traçable, garantissant une cohérence méthodologique sur l’ensemble du processus d’analyse.

Les résultats mettent en évidence des évolutions structurelles de long terme de la dynamique démographique française, caractérisées par des tendances persistantes, des composantes saisonnières marquées et des changements de régime potentiels. Ces éléments soulignent l’intérêt des outils économétriques automatisés pour l’analyse démographique et le suivi des politiques publiques dans des contextes temporels complexes.

Ce travail illustre la manière dont l’automatisation et l’assistance par l’intelligence artificielle peuvent renforcer la pratique économétrique, tout en préservant la rigueur scientifique et l’interprétabilité des résultats.

\vspace{0.5cm}

\noindent\textbf{Keywords:} Démographie ; Croissance naturelle ; Économétrie des séries temporelles ; Automatisation économétrique ; France

\vspace{0.2cm}

\noindent\textbf{JEL codes:}  
C22 (Time-Series Models),  
C51 (Model Construction and Estimation),  
J11 (Demographic Trends),  
J13 (Fertility, Mortality, Family Structure),  
O38 (Technological Change – Automation and AI)

\clearpage

% =================================================
% DÉBUT DU DOCUMENT PRINCIPAL
% =================================================


\tableofcontents
\listoffigures
\listoftables
\clearpage

% -----------------------------
% INTRODUCTION (non numérotée)
% -----------------------------
\chapter*{Introduction}
\addcontentsline{toc}{chapter}{Introduction}

% AUTO
\input{blocks/sec_data.tex}

\chapter{Analyse Descriptive et Décomposition}
% AUTO
\input{blocks/sec_descriptive.tex}

\chapter{Diagnostics Statistiques}
% AUTO
\input{blocks/sec_stationarity.tex}

\chapter{Analyse Univariée}
% AUTO
\input{blocks/sec_univariate.tex}

\chapter{Analyse Multivariée et Long Terme}
% AUTO
\input{blocks/sec_multivariate.tex}
\input{blocks/sec_cointegration.tex}

\chapter{Synthèse Historique et Anthropologique}
% AUTO
\input{blocks/sec_anthropology.tex}

% -----------------------------
% CONCLUSION (non numérotée)
% -----------------------------
\clearpage
\chapter*{Conclusion}
\addcontentsline{toc}{chapter}{Conclusion}

\input{blocks/sec_conclusion.tex}

\end{document}
"""

    def read_manifest(self) -> Dict[str, Any]:
        with self._manifest_lock:
            return self._read_json(self.paths.manifest_path)

    def update_manifest(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        with self._manifest_lock:
            current = self._read_json(self.paths.manifest_path)
            merged = deep_merge(current, patch)
            merged["updated_at"] = _utc_ts()
            self._write_json_atomic(self.paths.manifest_path, merged)
            return merged

    def register_step_status(self, step_name: str, *, status: str, summary: Optional[str] = None) -> Dict[str, Any]:
        patch = {"steps": {step_name: {"status": status, "summary": summary, "ts": _utc_ts()}}}
        return self.update_manifest(patch)

    def register_artefact(
        self,
        kind: str,
        lookup_key: str,
        rel_path: str,
        *,
        page: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta = meta or {}
        artefact_obj = {"key": lookup_key, "path": rel_path, "page": page, "meta": meta, "ts": _utc_ts()}
        patch = {
            "lookup": {kind: {lookup_key: rel_path}},
            "artefacts": {kind: [artefact_obj]},
        }
        return self.update_manifest(patch)

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, obj: Dict[str, Any]) -> None:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

        last_err: Optional[Exception] = None
        for _ in range(20):
            try:
                os.replace(tmp, path)
                return
            except PermissionError as e:
                last_err = e
                time.sleep(0.05)

        raise last_err if last_err else PermissionError(f"Cannot replace {tmp} -> {path}")
