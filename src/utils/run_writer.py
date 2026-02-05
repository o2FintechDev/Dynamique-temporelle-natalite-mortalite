# src/utils/run_writer.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import uuid


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge: b enrichit a.
    - dict -> merge récursif
    - list -> concat (sans dédup automatique pour rester déterministe)
    - scalaires -> overwrite par b
    """
    out = dict(a)
    for k, v in b.items():
        if k not in out:
            out[k] = v
            continue
        if isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        elif isinstance(out[k], list) and isinstance(v, list):
            out[k] = out[k] + v
        else:
            out[k] = v
    return out


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    artefacts_dir: Path
    figures_dir: Path
    tables_dir: Path
    metrics_dir: Path
    models_dir: Path
    logs_dir: Path
    manifest_path: Path
    narrative_path: Path
    latex_dir: Path
    latex_master_path: Path


class RunWriter:
    """
    Structure attendue :
    app/outputs/runs/<run_id>/
      manifest.json
      narrative.json
      artefacts/{figures,tables,metrics,models}/...
      latex/{blocks/*.tex, master.tex}
      logs/...
    """

    def __init__(self, base_runs_dir: Path, run_id: str) -> None:
        self.base_runs_dir = base_runs_dir
        self.run_id = run_id
        self.paths = self._init_dirs()

    @classmethod
    def create_new(cls, base_runs_dir: Path) -> "RunWriter":
        run_id = f"{_utc_ts()}_{uuid.uuid4().hex[:10]}"
        return cls(base_runs_dir, run_id)

    def _init_dirs(self) -> RunPaths:
        run_dir = self.base_runs_dir / self.run_id
        artefacts_dir = run_dir / "artefacts"
        figures_dir = artefacts_dir / "figures"
        tables_dir = artefacts_dir / "tables"
        metrics_dir = artefacts_dir / "metrics"
        models_dir = artefacts_dir / "models"
        logs_dir = run_dir / "logs"
        latex_dir = run_dir / "latex"
        blocks_dir = latex_dir / "blocks"

        run_dir.mkdir(parents=True, exist_ok=True)
        for d in [artefacts_dir, figures_dir, tables_dir, metrics_dir, models_dir, logs_dir, latex_dir, blocks_dir]:
            d.mkdir(parents=True, exist_ok=True)

        manifest_path = run_dir / "manifest.json"
        narrative_path = run_dir / "narrative.json"
        latex_master_path = latex_dir / "master.tex"

        if not manifest_path.exists():
            self._write_json(
                manifest_path,
                {
                    "run_id": self.run_id,
                    "created_at": _utc_ts(),
                    "lookup": {"figures": {}, "tables": {}, "metrics": {}, "models": {}, "latex_blocks": {}},
                    # NOTE: chaque item artefact embarque désormais "page"
                    "artefacts": {"figures": [], "tables": [], "metrics": [], "models": [], "latex_blocks": []},
                    "steps": {},
                },
            )
        if not narrative_path.exists():
            self._write_json(narrative_path, {"run_id": self.run_id, "blocks": {}, "updated_at": _utc_ts()})
        if not latex_master_path.exists():
            latex_master_path.write_text(self._latex_master_skeleton(), encoding="utf-8")

        return RunPaths(
            run_dir=run_dir,
            artefacts_dir=artefacts_dir,
            figures_dir=figures_dir,
            tables_dir=tables_dir,
            metrics_dir=metrics_dir,
            models_dir=models_dir,
            logs_dir=logs_dir,
            manifest_path=manifest_path,
            narrative_path=narrative_path,
            latex_dir=latex_dir,
            latex_master_path=latex_master_path,
        )

    def _latex_master_skeleton(self) -> str:
        return r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[french]{babel}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{geometry}
\geometry{margin=2.2cm}

\title{Croissance naturelle en France (1975--2025) : Analyse économétrique \& lecture anthropologique}
\author{AnthroDem Lab}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\newpage

% --- BLOCKS (auto) ---
% \input{blocks/<step_name>.tex}

\end{document}
"""

    def read_manifest(self) -> Dict[str, Any]:
        return self._read_json(self.paths.manifest_path)

    def update_manifest(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        current = self.read_manifest()
        merged = deep_merge(current, patch)
        merged["updated_at"] = _utc_ts()
        self._write_json_atomic(self.paths.manifest_path, merged)
        return merged

    def register_step_status(self, step_name: str, *, status: str, summary: Optional[str] = None) -> Dict[str, Any]:
        patch = {"steps": {step_name: {"status": status, "summary": summary, "ts": _utc_ts()}}}
        return self.update_manifest(patch)

    def register_artefact(
        self,
        kind: str,  # figures|tables|metrics|models|latex_blocks
        lookup_key: str,
        rel_path: str,
        *,
        page: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        IMPORTANT: 'page' est le routage UI. Chaque page Streamlit filtre uniquement ses artefacts.
        """
        meta = meta or {}
        artefact_obj = {"key": lookup_key, "path": rel_path, "page": page, "meta": meta, "ts": _utc_ts()}
        patch = {
            "lookup": {kind: {lookup_key: rel_path}},
            "artefacts": {kind: [artefact_obj]},
        }
        return self.update_manifest(patch)

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, obj: Dict[str, Any]) -> None:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
