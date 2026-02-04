# uils/run_writer.py
from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .run_context import get_current_run

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

class RunWriter:
    def __init__(self) -> None:
        self.ctx = get_current_run()
        self._counters: dict[str, int] = {"fig": 0, "table": 0, "metric": 0, "model": 0}

    def _next(self, kind: str) -> int:
        self._counters[kind] += 1
        return self._counters[kind]

    @staticmethod
    def _slug(s: str) -> str:
        return s.strip().lower().replace(" ", "_").replace("-", "_").replace("__", "_")

    def save_table(self, df: pd.DataFrame, slug: str) -> Path:
        i = self._next("table")
        name = f"table_{i:03d}_{self._slug(slug)}.csv"
        path = self.ctx.tables / name
        df.to_csv(path, index=True)
        return path

    def save_metric(self, payload: dict[str, Any], slug: str) -> Path:
        i = self._next("metric")
        name = f"metric_{i:03d}_{self._slug(slug)}.json"
        path = self.ctx.metrics / name
        _write_json(path, payload)
        return path

    def save_model_pickle(self, obj: Any, slug: str) -> Path:
        i = self._next("model")
        name = f"model_{i:03d}_{self._slug(slug)}.pkl"
        path = self.ctx.models / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return path

    def save_figure(self, fig: Any, slug: str) -> Path:
        """
        fig: matplotlib.figure.Figure
        """
        i = self._next("fig")
        name = f"fig_{i:03d}_{self._slug(slug)}.png"
        path = self.ctx.figures / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        return path

    def save_manifest(self, manifest: dict[str, Any]) -> Path:
        manifest = dict(manifest)
        manifest.setdefault("created_at_utc", utc_now_iso())
        path = self.ctx.root / "manifest.json"
        _write_json(path, manifest)
        return path
