# utils/run_reader.py
# utils/run_reader.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import runs_dir, run_dir


@dataclass(frozen=True)
class RunFiles:
    run_id: str
    root: Path
    manifest: Path
    artefacts: Path
    figures: Path
    tables: Path
    metrics: Path
    models: Path
    logs: Path


def list_runs() -> list[str]:
    rd = runs_dir()
    if not rd.exists():
        return []
    return sorted([p.name for p in rd.iterdir() if p.is_dir()], reverse=True)


def get_run_files(run_id: str) -> RunFiles:
    root = run_dir(run_id)
    artefacts = root / "artefacts"
    return RunFiles(
        run_id=run_id,
        root=root,
        manifest=root / "manifest.json",
        artefacts=artefacts,
        figures=artefacts / "figures",
        tables=artefacts / "tables",
        metrics=artefacts / "metrics",
        models=artefacts / "models",
        logs=root / "logs",
    )


def read_manifest(run_id: str) -> dict[str, Any]:
    p = get_run_files(run_id).manifest
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def read_table_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def read_metric_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_table(run_id: str, slug_contains: str) -> Path | None:
    files = get_run_files(run_id).tables
    if not files.exists():
        return None
    matches = [p for p in files.glob("table_*.csv") if slug_contains in p.name]
    return sorted(matches)[-1] if matches else None


def latest_metric(run_id: str, slug_contains: str) -> Path | None:
    files = get_run_files(run_id).metrics
    if not files.exists():
        return None
    matches = [p for p in files.glob("metric_*.json") if slug_contains in p.name]
    return sorted(matches)[-1] if matches else None


def latest_figure(run_id: str, slug_contains: str) -> Path | None:
    files = get_run_files(run_id).figures
    if not files.exists():
        return None
    matches = [p for p in files.glob("fig_*.png") if slug_contains in p.name]
    return sorted(matches)[-1] if matches else None


def list_figures(run_id: str) -> list[Path]:
    p = get_run_files(run_id).figures
    return sorted(p.glob("fig_*.png")) if p.exists() else []


class RunManager:
    """
    Accès centralisé aux runs + lookup manifest.
    Utilisation attendue côté pages:
      run_id = RunManager.get_latest_run_id()
      path = RunManager.get_artefact_path("acf_taux_naissances", run_id=run_id)
    """

    @staticmethod
    def get_latest_run_id() -> str | None:
        runs = list_runs()
        return runs[0] if runs else None

    @staticmethod
    def load_manifest(run_id: str | None = None) -> dict[str, Any]:
        rid = run_id or RunManager.get_latest_run_id()
        if not rid:
            return {}
        return read_manifest(rid)

    @staticmethod
    def get_artefact_path(label: str, *, run_id: str | None = None, absolute: bool = True) -> Path | None:
        """
        Retourne le chemin d'un artefact via manifest.lookup[label].
        - label: clé stable (slug) enregistrée par l'Executor lors de la persistance.
        - absolute=True: renvoie un Path absolu (recommandé pour Streamlit).
        """
        rid = run_id or RunManager.get_latest_run_id()
        if not rid:
            return None

        manifest = read_manifest(rid)
        lookup = manifest.get("lookup") or {}
        if not isinstance(lookup, dict):
            return None

        p = lookup.get(label)
        if not p:
            return None

        path = Path(p)
        if absolute and not path.is_absolute():
            # Défensif: si jamais un jour tu stockes en relatif
            path = (get_run_files(rid).root / path).resolve()
        return path

    @staticmethod
    def has(label: str, *, run_id: str | None = None) -> bool:
        return RunManager.get_artefact_path(label, run_id=run_id) is not None
