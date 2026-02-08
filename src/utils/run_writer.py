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

        # --- Init master.tex (plan stable) ---
        if not latex_master_path.exists():
            latex_master_path.write_text(self._latex_master_skeleton(), encoding="utf-8")

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
\usepackage{chngcntr}
\counterwithout{section}{chapter}
\renewcommand{\thesection}{\Roman{section}} % I, II, III...
\setcounter{secnumdepth}{1}
\setcounter{tocdepth}{1}

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
\begin{titlepage}
\thispagestyle{empty}

% -------------------------------------------------
% Titre
% -------------------------------------------------
\begin{center}
    {\Large Analyse économétrique automatisée}\\[0.3cm]
    {\Large par pipeline déterministe}\\[0.6cm]

    {\Huge\bfseries Automate économétrique}\\[0.4cm]
    {\LARGE\bfseries Croissance naturelle en France (1975--2025)}
\end{center}

\vspace{2.5cm}

% -------------------------------------------------
% Auteurs
% -------------------------------------------------
\begin{center}
    {\large Aude \textsc{Bernier}}\\
    {\large Clara \textsc{Pierreuse}}\\
    {\large Justine \textsc{Reiter--Guerville}}
\end{center}

\vfill

% -------------------------------------------------
% Formation
% -------------------------------------------------
\begin{center}
    {\large Master Monnaie Banque Finance Assurance}\\
    {\itshape Parcours Systèmes d’Information Économiques et Financiers (SIEF)}\\[0.3cm]
    Université de Montpellier\\[0.6cm]
    Année universitaire 2025--2026
\end{center}

\vspace{1.2cm}

% -------------------------------------------------
% Encadrement (NON centré)
% -------------------------------------------------
\vspace{1.2cm}
\begin{flushleft}
{\small Sous la supervision de : Monsieur Mestre Roman}
\end{flushleft}

\end{titlepage}

\begin{document}

\maketitle
\tableofcontents
\listoffigures
\listoftables
\clearpage

\chapter{Introduction et Préparation}

% AUTO
\input{blocks/sec_data.tex}

\chapter{Analyse Descriptive et Décomposition}
% AUTO
\input{blocks/sec_descriptive.tex}

\chapter{Diagnostics Statistiques}
% AUTO
\input{blocks/sec_stationarity.tex}

\chapter{Modélisation Univariée}
% AUTO
\input{blocks/sec_univariate.tex}

\chapter{Analyse Multivariée et Long Terme}
% AUTO
\input{blocks/sec_multivariate.tex}
\input{blocks/sec_cointegration.tex}

\chapter{Synthèse Historique et Anthropologique}
% AUTO
\input{blocks/sec_anthropology.tex}

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
