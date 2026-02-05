# src/agent/executor.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Callable

import streamlit as st

from src.utils.run_writer import RunWriter
from src.utils.logger import get_logger

log = get_logger("agent.executor")


@dataclass
class StepSpec:
    name: str
    required_prev_artefacts: Dict[str, list[str]]  # ex: {"tables": ["ts_clean"], "metrics": ["adf_births"]}
    tool: Callable[..., Dict[str, Any]]            # tool(step_ctx) -> outputs
    produces: Dict[str, list[str]]                 # ex: {"figures": ["fig_ts"], "tables": ["summary_tbl"]}


class AgentExecutor:
    def __init__(self, runs_dir: str = "app/outputs/runs") -> None:
        self.base_runs_dir = Path(runs_dir)

    def get_or_create_run(self) -> RunWriter:
        if "run_id" in st.session_state and st.session_state.run_id:
            run_id = st.session_state.run_id
            return RunWriter(self.base_runs_dir, run_id)

        rw = RunWriter.create_new(self.base_runs_dir)
        st.session_state.run_id = rw.run_id
        return rw

    def _has_required(self, manifest: Dict[str, Any], req: Dict[str, list[str]]) -> bool:
        lookup = manifest.get("lookup", {})
        for kind, keys in (req or {}).items():
            kind_lookup = lookup.get(kind, {}) or {}
            for k in keys:
                if k not in kind_lookup:
                    return False
        return True

    def execute_step(self, step: StepSpec, *, step_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        rw = self.get_or_create_run()
        manifest = rw.read_manifest()

        # Statefulness: si step déjà DONE => ne pas recalculer
        step_status = (manifest.get("steps", {}).get(step.name) or {}).get("status")
        if step_status == "DONE":
            log.info("Step %s déjà calculée (DONE). Skip.", step.name)
            return {"run_id": rw.run_id, "skipped": True, "manifest": manifest}

        # Gating: vérifier prérequis
        if not self._has_required(manifest, step.required_prev_artefacts):
            rw.register_step_status(step.name, status="BLOCKED", summary="Prerequisites missing in manifest.lookup")
            return {
                "run_id": rw.run_id,
                "skipped": True,
                "blocked": True,
                "missing_prerequisites": step.required_prev_artefacts,
                "manifest": rw.read_manifest(),
            }

        rw.register_step_status(step.name, status="RUNNING")

        step_params = step_params or {}
        step_ctx = {"run_id": rw.run_id, "run_paths": rw.paths, "manifest": manifest, "params": step_params}

        outputs = step.tool(step_ctx)  # tool responsable d’écrire fichiers + register_artefact
        rw.register_step_status(step.name, status="DONE", summary=outputs.get("summary"))

        return {"run_id": rw.run_id, "outputs": outputs, "manifest": rw.read_manifest()}

    def recommend_next(self, current_step: str, ordered_steps: list[StepSpec]) -> Optional[str]:
        rw = self.get_or_create_run()
        manifest = rw.read_manifest()

        idx = next((i for i, s in enumerate(ordered_steps) if s.name == current_step), None)
        if idx is None:
            return None

        # règle stricte: ne recommander que si prerequis du candidat sont satisfaits
        for cand in ordered_steps[idx + 1 :]:
            if self._has_required(manifest, cand.required_prev_artefacts):
                # et on évite de proposer une step déjà DONE
                if (manifest.get("steps", {}).get(cand.name) or {}).get("status") != "DONE":
                    return cand.name
        return None
