# src/agent/executor.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Callable

import streamlit as st

from src.utils.run_writer import RunWriter
from src.utils.logger import get_logger
from src.utils.persist import persist_outputs
from src.agent.schemas import Plan

log = get_logger("agent.executor")


@dataclass
class StepSpec:
    name: str
    required_prev_artefacts: Dict[str, list[str]]
    tool: Callable[..., Dict[str, Any]]
    produces: Dict[str, list[str]]


class AgentExecutor:
    def __init__(self, runs_dir: str = "app/outputs/runs", run_id: Optional[str] = None) -> None:
        self.base_runs_dir = Path(runs_dir)
        self._forced_run_id = run_id

    def get_or_create_run(self) -> RunWriter:
        if self._forced_run_id:
            st.session_state.run_id = self._forced_run_id
            return RunWriter(self.base_runs_dir, self._forced_run_id)

        if "run_id" in st.session_state and st.session_state.run_id:
            return RunWriter(self.base_runs_dir, st.session_state.run_id)

        rw = RunWriter.create_new(self.base_runs_dir)
        st.session_state.run_id = rw.run_id
        return rw

    def run(self, plan: Plan, user_query: Optional[str] = None) -> Dict[str, Any]:
        """
        IMPORTANT:
        - Routing page uniquement via params["_page"] (injecté par streamlit_app.py).
        - AUCUNE dépendance à plan.page_targets (non présent dans ton Plan dataclass).
        """
        from src.agent.registry import STEP_REGISTRY

        step = STEP_REGISTRY.get(plan.intent)
        if step is None:
            raise ValueError(f"Intent/step inconnu: {plan.intent}")

        if user_query:
            log.info("chat intent=%s | %s", plan.intent, user_query)

        params: Dict[str, Any] = {}
        if plan.tool_calls:
            params = plan.tool_calls[0].params or {}

        return self.execute_step(step, step_params=params)

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

        # skip si DONE
        step_status = (manifest.get("steps", {}).get(step.name) or {}).get("status")
        if step_status == "DONE":
            return {"run_id": rw.run_id, "skipped": True, "manifest": manifest}

        # gating minimal: step1 obligatoire avant autres steps
        if step.name != "step1_load_and_profile":
            s1 = (manifest.get("steps", {}).get("step1_load_and_profile") or {}).get("status")
            if s1 != "DONE":
                rw.register_step_status(step.name, status="BLOCKED", summary="Requires step1_load_and_profile DONE")
                return {"run_id": rw.run_id, "blocked": True, "manifest": rw.read_manifest()}

        # gating lookup optionnel
        if step.required_prev_artefacts and not self._has_required(manifest, step.required_prev_artefacts):
            rw.register_step_status(step.name, status="BLOCKED", summary="Prerequisites missing in manifest.lookup")
            return {"run_id": rw.run_id, "blocked": True, "manifest": rw.read_manifest()}

        rw.register_step_status(step.name, status="RUNNING")

        step_params = step_params or {}
        page = step_params.get("_page")  # routing UI injecté par streamlit_app.py

        step_ctx = {"run_id": rw.run_id, "run_paths": rw.paths, "manifest": manifest, "params": step_params}

        outputs = step.tool(step_ctx)

        # Persist avec routage page
        persist_info = persist_outputs(rw, step.name, outputs, page=page)

        rw.register_step_status(step.name, status="DONE", summary=(outputs or {}).get("summary"))

        if step.name == "export_latex_pdf":
            rw.update_manifest({"run_status": "FINALIZED"})

        return {"run_id": rw.run_id, "outputs": outputs, "persist": persist_info, "manifest": rw.read_manifest()}

    def recommend_next(self, current_step: str, ordered_steps: list[StepSpec]) -> Optional[str]:
        rw = self.get_or_create_run()
        manifest = rw.read_manifest()

        idx = next((i for i, s in enumerate(ordered_steps) if s.name == current_step), None)
        if idx is None:
            return None

        for cand in ordered_steps[idx + 1 :]:
            if self._has_required(manifest, cand.required_prev_artefacts):
                if (manifest.get("steps", {}).get(cand.name) or {}).get("status") != "DONE":
                    return cand.name
        return None
