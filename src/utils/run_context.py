# src/utils/run_context.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import secrets

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def new_run_id() -> str:
    return f"{utc_now_iso()}_{secrets.token_hex(5)}"

@dataclass
class RunContext:
    run_id: str
    seq_table: int = 0
    seq_fig: int = 0
    seq_metric: int = 0
    seq_model: int = 0
