from __future__ import annotations
import logging

def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(name)s - %(message)s")
        handler.setFormatter(fmt)
        log.addHandler(handler)
        log.setLevel(logging.INFO)
    return log
