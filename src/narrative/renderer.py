# src/narrative/renderer.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
from datetime import datetime, timezone

from src.utils.run_writer import RunWriter


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def append_to_narrative(runs_dir: str, run_id: str, step_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrichit narrative.json sous blocks[step_name] (append-only logique).
    """
    rw = RunWriter(Path(runs_dir), run_id)
    p = rw.paths.narrative_path
    obj = json.loads(p.read_text(encoding="utf-8"))

    blocks = obj.get("blocks", {})
    step_block = blocks.get(step_name, {})
    # merge simple (step-level). Si tu veux list-append, fais-le côté data.
    step_block.update(data)
    step_block["updated_at"] = _utc_ts()
    blocks[step_name] = step_block

    obj["blocks"] = blocks
    obj["updated_at"] = _utc_ts()

    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return obj
