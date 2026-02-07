# src/utils/run_reader.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Iterable

import pandas as pd

from src.utils.paths import runs_dir, run_dir


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

def _find_artefact_item(manifest: dict[str, Any], *, kind: str, label: str) -> dict[str, Any] | None:
    items = ((manifest.get("artefacts") or {}).get(kind) or [])
    if not isinstance(items, list):
        return None
    return next((it for it in items if isinstance(it, dict) and it.get("key") == label), None)


def get_table_dataframe(label: str, *, run_id: str | None = None) -> pd.DataFrame | None:
    rid = run_id or RunManager.get_latest_run_id()
    if not rid:
        return None

    rf = get_run_files(rid)
    manifest = read_manifest(rid)

    it = _find_artefact_item(manifest, kind="tables", label=label)
    if it:
        meta = it.get("meta") or {}
        csv_rel = meta.get("csv_path")
        if isinstance(csv_rel, str) and csv_rel:
            p = Path(csv_rel)
            if not p.is_absolute():
                p = (rf.root / p).resolve()
            if p.exists():
                return pd.read_csv(p)

    # fallback si jamais certaines tables sont encore des CSV dans lookup
    p_lookup = RunManager.get_artefact_path(label, run_id=rid, kind="tables", absolute=True)
    if p_lookup and p_lookup.exists() and p_lookup.suffix.lower() == ".csv":
        return pd.read_csv(p_lookup)

    return None


def read_metric_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _lookup_get_path(lookup: Any, label: str, *, kind: Optional[str] = None) -> Optional[str]:
    """
    Supporte:
    - lookup "plat" (ancien): { "<label>": "<path>", ... }
    - lookup "typé" (nouveau): { "metrics": { "<label>": "<path>" }, ... }
    """
    if not isinstance(lookup, dict):
        return None

    # 1) ancien format (plat)
    if label in lookup and isinstance(lookup[label], str):
        return lookup[label]

    # 2) nouveau format (par kind)
    if kind:
        bucket = lookup.get(kind)
        if isinstance(bucket, dict):
            p = bucket.get(label)
            if isinstance(p, str):
                return p

    # 3) fallback: scan de tous les buckets
    for _, bucket in lookup.items():
        if isinstance(bucket, dict) and label in bucket and isinstance(bucket[label], str):
            return bucket[label]

    return None


def get_artefacts_for_page(manifest: dict[str, Any], *, page: str, kind: Optional[str] = None) -> list[dict[str, Any]]:
    """
    Retourne les artefacts du manifest filtrés par page (et éventuellement par kind).
    """
    arte = (manifest.get("artefacts") or {})
    if not isinstance(arte, dict):
        return []

    kinds: Iterable[str]
    if kind:
        kinds = [kind]
    else:
        kinds = ["figures", "tables", "metrics", "models", "latex_blocks"]

    out: list[dict[str, Any]] = []
    for k in kinds:
        items = arte.get(k) or []
        if not isinstance(items, list):
            continue
        out.extend([it for it in items if isinstance(it, dict) and it.get("page") == page])

    return out


class RunManager:
    @staticmethod
    def get_latest_run_id() -> str | None:
        runs = list_runs()
        return runs[0] if runs else None

    @staticmethod
    def get_artefact_path(
        label: str,
        *,
        run_id: str | None = None,
        kind: str | None = None,     # "metrics" | "tables" | "figures" | "models" | "latex_blocks"
        absolute: bool = True,
    ) -> Path | None:
        """
        Résout un label -> path via manifest.lookup.
        Compatible lookup plat (ancien) ou lookup typé (nouveau).
        """
        rid = run_id or RunManager.get_latest_run_id()
        if not rid:
            return None

        manifest = read_manifest(rid)
        lookup = manifest.get("lookup") or {}

        p = _lookup_get_path(lookup, label, kind=kind)
        if not p:
            return None

        path = Path(p)
        if absolute and not path.is_absolute():
            path = (get_run_files(rid).root / path).resolve()
        return path

def read_table_from_artefact(run_id: str, artefact: dict[str, Any]) -> pd.DataFrame:
    """
    Lit une table pour l'UI Streamlit.
    - Si manifest path pointe vers .csv : lit directement.
    - Si manifest path pointe vers .tex : lit le CSV associé (meta.csv_path ou même nom .csv).
    """
    rf = get_run_files(run_id)
    rel = str(artefact.get("path") or "")
    meta = artefact.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}

    def _read_csv(p: Path) -> pd.DataFrame:
        # robuste: utf-8-sig + détection séparateur basique
        try:
            return pd.read_csv(p, encoding="utf-8")
        except Exception:
            try:
                return pd.read_csv(p, encoding="utf-8-sig")
            except Exception:
                # fallback séparateur ';'
                return pd.read_csv(p, sep=";", encoding="utf-8-sig")

    def _abs(p: Path) -> Path:
        return p if p.is_absolute() else (rf.root / p).resolve()

    # 1) chemin CSV explicite dans le meta
    csv_rel = meta.get("csv_path")
    if isinstance(csv_rel, str) and csv_rel:
        p = _abs(Path(csv_rel))
        if not p.exists():
            raise FileNotFoundError(f"CSV meta.csv_path introuvable: {p}")
        return _read_csv(p)

    # 2) si path déjà csv
    if rel.lower().endswith(".csv"):
        p = _abs(Path(rel))
        if not p.exists():
            raise FileNotFoundError(f"CSV introuvable: {p}")
        return _read_csv(p)

    # 3) si path .tex -> tente même nom .csv
    if rel.lower().endswith(".tex"):
        csv_guess = Path(rel).with_suffix(".csv")
        p = _abs(csv_guess)
        if not p.exists():
            raise FileNotFoundError(
                f"CSV associé introuvable (meta.csv_path absent). Attendu: {p} (à partir de {rel})"
            )
        return _read_csv(p)

    raise FileNotFoundError(f"Table non lisible pour UI: path={rel} meta={meta}")
