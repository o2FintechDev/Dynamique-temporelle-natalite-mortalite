from __future__ import annotations

import os
from dataclasses import dataclass
from contextvars import ContextVar
from pathlib import Path

from .paths import ensure_run_tree

_CURRENT_RUN: ContextVar["RunContext | None"] = ContextVar("_CURRENT_RUN", default=None)

@dataclass(frozen=True)
class RunContext:
    run_id: str
    root: Path
    logs: Path
    figures: Path
    tables: Path
    metrics: Path
    models: Path

def set_current_run(ctx: RunContext) -> None:
    _CURRENT_RUN.set(ctx)

def get_current_run() -> RunContext:
    ctx = _CURRENT_RUN.get()
    if ctx is None:
        raise RuntimeError("RunContext non initialisé. Appelle init_run_context(run_id).")
    return ctx

def init_run_context(run_id: str) -> RunContext:
    tree = ensure_run_tree(run_id)
    ctx = RunContext(
        run_id=run_id,
        root=tree["run"],
        logs=tree["logs"],
        figures=tree["figures"],
        tables=tree["tables"],
        metrics=tree["metrics"],
        models=tree["models"],
    )
    set_current_run(ctx)
    os.environ["ANTHRODEM_RUN_ID"] = run_id
    return ctx
