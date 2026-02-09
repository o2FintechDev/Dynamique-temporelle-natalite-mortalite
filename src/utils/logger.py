# src/utils/logger.py
from __future__ import annotations
import logging

def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        h.setFormatter(fmt)
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log
