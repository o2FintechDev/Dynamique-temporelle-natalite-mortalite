from __future__ import annotations

import platform
from typing import Any

import pandas as pd

from src.agent.schemas import Plan, ExecutionResult, ArtefactRef, Manifest
from src.agent.tools import get_tool
from src.utils.run_writer import RunWriter, utc_now_iso
from src.utils import get_logger

log = get_logger("agent.executor")

class AgentExecutor:
    """
    Exécute un Plan:
      - appelle tools.py (mapping strict)
      - persiste systématiquement les sorties via RunWriter
      - produit un manifest.json canonique
    """
    def __init__(self, user_query: str) -> None:
        self.user_query = user_query
        self.writer = RunWriter()

    def run(self, plan: Plan) -> ExecutionResult:
        tools_called: list[str] = []
        variables: list[str] = []
        artefacts: list[ArtefactRef] = []

        for call in plan.tool_calls:
            tools_called.append(call.tool_name)
            variables.extend(call.variables)

            fn = get_tool(call.tool_name)
            out = fn(variables=call.variables, **call.params)  # dict sérialisable

            # Convention de sorties:
            # out may contain: {"tables": {slug: df}, "metrics": {slug: dict}, "models": {slug: obj}}
            if "tables" in out:
                for slug, df in out["tables"].items():
                    if not isinstance(df, pd.DataFrame):
                        raise TypeError(f"Table '{slug}' doit être un DataFrame.")
                    p = self.writer.save_table(df, slug)
                    artefacts.append(ArtefactRef(kind="table", path=str(p), label=slug))

            if "metrics" in out:
                for slug, payload in out["metrics"].items():
                    if not isinstance(payload, dict):
                        raise TypeError(f"Metric '{slug}' doit être un dict.")
                    p = self.writer.save_metric(payload, slug)
                    artefacts.append(ArtefactRef(kind="metric", path=str(p), label=slug))

            if "models" in out:
                for slug, obj in out["models"].items():
                    p = self.writer.save_model_pickle(obj, slug)
                    artefacts.append(ArtefactRef(kind="model", path=str(p), label=slug))

            # Figures: volontairement gérées plus tard (charts.py) mais même logique attendue:
            # out["figures"] = {slug: matplotlib_figure} (optionnel, J2)
            # -> on implémentera l’export figure J2 pour éviter les divergences.

        result = ExecutionResult(
            run_id=self.writer.ctx.run_id,
            intent=plan.intent,
            variables=sorted(set(variables)),
            tools_called=tools_called,
            artefacts=artefacts,
        )

        manifest = Manifest(
            run_id=self.writer.ctx.run_id,
            created_at_utc=utc_now_iso(),
            user_query=self.user_query,
            intent=plan.intent,
            tools_called=tools_called,
            variables=sorted(set(variables)),
            artefacts=[a.model_dump() for a in artefacts],
            versions={
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
        ).model_dump()

        mp = self.writer.save_manifest(manifest)
        artefacts.append(ArtefactRef(kind="manifest", path=str(mp), label="manifest"))

        return result
