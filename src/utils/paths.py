from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import hashlib
import json

@dataclass(frozen=True)
class RunPaths:
    run_id: str
    run_dir: Path
    artefacts_dir: Path
    figures_dir: Path
    tables_dir: Path
    metrics_dir: Path
    models_dir: Path
    logs_dir: Path

def make_run_id(payload: dict) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    h = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:10]
    return f"{ts}_{h}"

def ensure_dirs(base_outputs_dir: Path, run_id: str) -> RunPaths:
    run_dir = base_outputs_dir / "runs" / run_id
    artefacts_dir = run_dir / "artefacts"
    figures_dir = artefacts_dir / "figures"
    tables_dir = artefacts_dir / "tables"
    metrics_dir = artefacts_dir / "metrics"
    models_dir = artefacts_dir / "models"
    logs_dir = run_dir / "logs"

    for d in [figures_dir, tables_dir, metrics_dir, models_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        artefacts_dir=artefacts_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        metrics_dir=metrics_dir,
        models_dir=models_dir,
        logs_dir=logs_dir,
    )
