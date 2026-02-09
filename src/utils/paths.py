# src/utils/paths.py
from __future__ import annotations

from pathlib import Path

def repo_root() -> Path:
    # repo_root = parent of "src"
    return Path(__file__).resolve().parents[2]

def app_dir() -> Path:
    return repo_root() / "app"

def data_dir() -> Path:
    return repo_root() / "data"

def outputs_dir() -> Path:
    return app_dir() / "outputs"

def runs_dir() -> Path:
    return outputs_dir() / "runs"

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def run_dir(run_id: str) -> Path:
    return runs_dir() / run_id

def run_logs_dir(run_id: str) -> Path:
    return run_dir(run_id) / "logs"

def run_artefacts_dir(run_id: str) -> Path:
    return run_dir(run_id) / "artefacts"

def run_artefacts_subdirs(run_id: str) -> dict[str, Path]:
    base = run_artefacts_dir(run_id)
    return {
        "figures": base / "figures",
        "tables": base / "tables",
        "metrics": base / "metrics",
        "models": base / "models",
    }

def ensure_run_tree(run_id: str) -> dict[str, Path]:
    ensure_dirs(runs_dir())
    rd = run_dir(run_id)
    sub = run_artefacts_subdirs(run_id)
    ensure_dirs(rd, run_logs_dir(run_id), run_artefacts_dir(run_id), *sub.values())
    return {"run": rd, "logs": run_logs_dir(run_id), **sub}
