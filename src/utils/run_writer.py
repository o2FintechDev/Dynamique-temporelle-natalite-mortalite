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
    """
    Deep merge: b enrichit a.
    - dict -> merge récursif
    - list -> concat (sans dédup automatique pour rester déterministe)
    - scalaires -> overwrite par b
    """
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
    narrative_path: Path
    latex_dir: Path
    latex_master_path: Path


class RunWriter:
    """
    app/outputs/runs/<run_id>/
      manifest.json
      narrative.json
      artefacts/{figures,tables,metrics,models}/...
      latex/{blocks/*.tex, master.tex}
      logs/...
    """

    def __init__(self, base_runs_dir: Path, run_id: str) -> None:
        self.base_runs_dir = base_runs_dir
        self.run_id = run_id
        self.paths = self._init_dirs()
        # Lock unique par run pour éviter collisions Streamlit rerun + Windows handle locks
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
        narrative_path = run_dir / "narrative.json"
        latex_master_path = latex_dir / "master.tex"

        if not manifest_path.exists():
            self._write_json(
                manifest_path,
                {
                    "run_id": self.run_id,
                    "created_at": _utc_ts(),
                    "lookup": {"figures": {}, "tables": {}, "metrics": {}, "models": {}, "latex_blocks": {}},
                    "artefacts": {"figures": [], "tables": [], "metrics": [], "models": [], "latex_blocks": []},
                    "steps": {},
                },
            )

        if not narrative_path.exists():
            self._write_json(narrative_path, {"run_id": self.run_id, "blocks": {}, "updated_at": _utc_ts()})

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
            narrative_path=narrative_path,
            latex_dir=latex_dir,
            latex_master_path=latex_master_path,
        )

    def _latex_master_skeleton(self) -> str:
        """
        Template "rapport de recherche" (structure méthodologique imposée).

        IMPORTANT:
        - master.tex est dans latex/ -> les artefacts sont dans ../artefacts/...
        - les blocs auto sont dans latex/blocks/*.tex -> \input{blocks/...} (chemin relatif au master)
        - les figures doivent être résolues depuis ../artefacts/figures via \graphicspath
        - centrage auto des floats sans centrer tout le texte
        """
        return r"""\documentclass[11pt,a4paper]{report}

% ----------------------------
% Encodage / Langue
% ----------------------------
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[french]{babel}

% ----------------------------
% Mise en page
% ----------------------------
\usepackage{geometry}
\geometry{margin=2.5cm}
\usepackage{setspace}
\onehalfspacing

% ----------------------------
% Maths / tableaux / figures
% ----------------------------
\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{adjustbox}
\usepackage{xcolor}
\usepackage{hyperref}

% Centrage automatique figures/tables (sans centrer tout le texte)
\usepackage{etoolbox}
\AtBeginEnvironment{figure}{\centering}
\AtBeginEnvironment{table}{\centering}

% Chemins artefacts (master.tex est dans latex/)
\graphicspath{{./../artefacts/figures/}{./}}

\title{Dynamique temporelle de la natalité et de la mortalité en France (1975--2025)}
\author{AnthroDem Lab}
\date{\today}

\begin{document}

\maketitle
\tableofcontents
\listoffigures
\listoftables
\clearpage

% ============================================================
\chapter{Introduction et Préparation}
% ============================================================

\section{Mise en contexte : l'exception démographique française et l'inversion de 2023}
La France a longtemps été décrite comme une \emph{exception démographique} en Europe, caractérisée par une fécondité relativement plus élevée et un solde naturel plus résilient. L'inversion observée en 2023, marquée par une dégradation du solde naturel, impose une lecture dynamique : l'évolution conjointe de la natalité et de la mortalité doit être interprétée comme le produit d'un système socio-économique soumis à chocs, transitions et ruptures.

\section{Construction de la variable : passage en taux pour corriger l'hétéroscédasticité}
Le passage des niveaux (naissances et décès en effectifs) à des taux pour $1000$ habitants vise à stabiliser la variance et à rendre la série plus comparable dans le temps malgré les changements de taille de population. Cette transformation est économétriquement motivée par la réduction de l'hétéroscédasticité structurelle et l'amélioration de la stationnarité conditionnelle.

\section{Définition formelle : équation du solde naturel}
On note $b_t$ le taux de natalité (pour $1000$ habitants) et $d_t$ le taux de mortalité (pour $1000$ habitants). La variable d'intérêt $Y_t$ (croissance naturelle / solde naturel en taux) est définie par :
\begin{equation}
Y_t = b_t - d_t.
\end{equation}

\begin{figure}[H]
\caption{Placeholder — Série $b_t$, $d_t$ et $Y_t$ (1975--2025)}
\label{fig:intro-series}
\includegraphics[width=0.95\linewidth]{fig.placeholder.series.png}
\end{figure}

% --- BLOCK AUTO (step1) ---
\input{blocks/step1_load_and_profile.tex}

% ============================================================
\chapter{Analyse Descriptive et Décomposition}
% ============================================================

\section{Analyse qualitative : tendance et cycles}
L'inspection visuelle de $Y_t$ permet d'identifier (i) une tendance de fond, (ii) des cycles et (iii) des régimes de volatilité.

\section{Décomposition additive}
Une décomposition additive standard est posée :
\begin{equation}
Y_t = T_t + S_t + \epsilon_t.
\end{equation}

\begin{figure}[H]
\caption{Placeholder — Décomposition additive : $T_t$, $S_t$, $\epsilon_t$}
\label{fig:decomp-additive}
\includegraphics[width=0.95\linewidth]{fig.placeholder.decomp.png}
\end{figure}

\section{Diagnostic de saisonnalité : déterministe vs stochastique}
Avant toute modélisation, il est nécessaire de qualifier la saisonnalité (déterministe vs stochastique).

% --- BLOCK AUTO (step2) ---
\input{blocks/step2_descriptive.tex}

% ============================================================
\chapter{Diagnostics Statistiques : la phase critique}
% ============================================================

\section{Tests de racine unitaire : ADF, Phillips-Perron, bande de Dickey-Fuller}
La stationnarité conditionne la spécification. Trois diagnostics complémentaires sont mobilisés.

\subsection{Régressions de test (forme générique)}
\begin{equation}
\Delta Y_t = \mu + \beta t + \gamma Y_{t-1} + \sum_{i=1}^{p} \phi_i \Delta Y_{t-i} + u_t.
\end{equation}

\section{Identification du processus : TS vs DS}
Distinction entre Trend Stationary (TS) et Difference Stationary (DS).

\section{Analyse corrélographique : ACF/PACF et persistance}
\begin{figure}[H]
\caption{Placeholder — ACF/PACF de $Y_t$ (niveau)}
\label{fig:acf-pacf-level}
\includegraphics[width=0.95\linewidth]{fig.placeholder.acf_pacf_level.png}
\end{figure}

% --- BLOCK AUTO (step3) ---
\input{blocks/step3_stationarity.tex}

% ============================================================
\chapter{Modélisation Univariée et Mémoire Longue}
% ============================================================

\section{Approche Box-Jenkins : ARMA/ARIMA}
\subsection{Spécification ARIMA}
\begin{equation}
\phi(L)(1-L)^d Y_t = \theta(L)\varepsilon_t.
\end{equation}

\section{Analyse de Hurst et mémoire longue}
\subsection{Lien Hurst et différenciation fractionnaire}
\begin{equation}
H = d + 0.5.
\end{equation}

\subsection{Modèle ARFIMA}
\begin{equation}
\phi(L)(1-L)^d Y_t = \theta(L)\varepsilon_t, \quad d \in (-0.5, 0.5).
\end{equation}

% --- BLOCK AUTO (step4) ---
\input{blocks/step4_univariate.tex}

% ============================================================
\chapter{Analyse Multivariée et Long Terme}
% ============================================================

\section{Modèle VAR : dépendances dynamiques}
\begin{equation}
X_t = c + \sum_{i=1}^{p} A_i X_{t-i} + \varepsilon_t.
\end{equation}

\section{Cointégration (VECM) : test de Johansen}
\begin{equation}
\Delta X_t = \Pi X_{t-1} + \sum_{i=1}^{p-1}\Gamma_i \Delta X_{t-i} + \mu + \varepsilon_t,
\end{equation}

\subsection{Interprétation : vitesse d'ajustement}
\begin{equation}
\Pi = \alpha \beta'.
\end{equation}

% --- BLOCKS AUTO (step5 & step6) ---
\input{blocks/step5_var.tex}
\input{blocks/step6_cointegration.tex}

% ============================================================
\chapter{Synthèse Anthropologique Augmentée}
% ============================================================

\section{Lecture Toddienne : ruptures détectées et fragilisation du contrat social}
Lecture qualitative des ruptures/régimes comme signaux de recomposition du contrat social.

\section{Conclusion : pérennité du modèle français à l'horizon 2030}
Synthèse économétrique + perspective anthropologique.

% --- BLOCK AUTO (step7) ---
\input{blocks/step7_anthropology.tex}

\end{document}
"""

    def read_manifest(self) -> Dict[str, Any]:
        # Windows: protéger lecture pendant un replace
        with self._manifest_lock:
            return self._read_json(self.paths.manifest_path)

    def update_manifest(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        # Une seule écriture du manifest à la fois
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
        for _ in range(20):  # ~1s max (20 * 50ms)
            try:
                os.replace(tmp, path)  # plus direct et fiable que Path.replace
                return
            except PermissionError as e:
                last_err = e
                time.sleep(0.05)

        raise last_err if last_err else PermissionError(f"Cannot replace {tmp} -> {path}")
