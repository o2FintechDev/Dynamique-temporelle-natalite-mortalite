from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .run_context import get_current_run

_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log

    log.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(_FMT))
    log.addHandler(ch)

    # File handler bound to current run (if available)
    try:
        ctx = get_current_run()
        logfile = ctx.logs / "run.log"
        fh = RotatingFileHandler(logfile, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(_FMT))
        log.addHandler(fh)
    except Exception:
        # run_context not initialised yet: ignore file logging
        pass

    log.propagate = False
    return log
