from __future__ import annotations
from typing import Any
import pandas as pd

from src.utils import SessionStore, get_logger
from .schema import Plan, NarrativePacket, EvidenceItem
from .tools import ToolContext, build_wide_monthly, make_coverage_report, make_describe, make_timeseries_plot

log = get_logger("agent.executor")

def execute_plan(plan: Plan, ctx: ToolContext, store: SessionStore) -> dict[str, Any]:
    """
    Exécute le plan et stocke des artefacts.
    Retourne un dict de state utile (wide, artefact ids...).
    """
    state: dict[str, Any] = {}
    wide: pd.DataFrame | None = None

    for call in plan.tool_calls:
        if call.tool_name == "build_wide_monthly":
            wide = build_wide_monthly(ctx, call.params["variable_ids"])
            art = store.add("dataframe", "Table harmonisée (mensuelle)", wide, meta={"kind": "wide"})
            state["wide"] = wide
            state["wide_artefact_id"] = art.artefact_id

        elif call.tool_name == "make_coverage_report":
            if wide is None:
                continue
            rep = make_coverage_report(wide)
            art = store.add("dataframe", "Data coverage report", rep, meta={"kind": "coverage"})
            state["coverage_artefact_id"] = art.artefact_id

        elif call.tool_name == "make_describe":
            if wide is None:
                continue
            desc = make_describe(wide)
            art = store.add("dataframe", "Statistiques descriptives", desc, meta={"kind": "describe"})
            state["describe_artefact_id"] = art.artefact_id

        elif call.tool_name == "make_timeseries_plot":
            if wide is None:
                continue
            fig = make_timeseries_plot(wide, call.params.get("title", "Séries temporelles"))
            art = store.add("figure", call.params.get("title", "Séries temporelles"), fig, meta={"kind": "plot"})
            state["plot_artefact_id"] = art.artefact_id

        else:
            log.info(f"Outil ignoré: {call.tool_name}")

    return state

def build_narrative_packet(plan: Plan, store: SessionStore) -> NarrativePacket:
    evidence = []
    for a in store.list():
        evidence.append(EvidenceItem(artefact_id=a.artefact_id, type=a.type, title=a.title))
    return NarrativePacket(
        intent=plan.intent,
        evidence=evidence,
        constraints={
            "no_external_facts": True,
            "only_from_artefacts": True,
            "no_sentence_without_artefact_ref": True,
        },
    )
