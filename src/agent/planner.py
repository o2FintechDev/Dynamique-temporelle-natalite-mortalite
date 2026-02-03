from __future__ import annotations
import re
from src.data_api import VARIABLES
from .schema import Plan, ToolCall, Intent

def _match_variables(user_text: str) -> list[str]:
    """
    Matching simple:
    - si l’utilisateur cite un variable_id exact: on le prend
    - sinon matching par mots-clés présents dans label
    """
    txt = user_text.lower()
    hits = []

    for vid, spec in VARIABLES.items():
        if re.search(rf"\b{re.escape(vid.lower())}\b", txt):
            hits.append(vid)

    if hits:
        return hits

    for vid, spec in VARIABLES.items():
        lab = spec.label.lower()
        # match grossier par tokens significatifs
        tokens = [t for t in re.split(r"[^a-z0-9]+", lab) if len(t) >= 4]
        score = sum(1 for t in tokens if t in txt)
        if score >= 2:
            hits.append(vid)

    # fallback: si rien, renvoyer un set “démo”
    return hits or ["unrate_us", "cpi_us"]

def infer_intent(user_text: str) -> Intent:
    t = user_text.lower()
    if any(k in t for k in ["compare", "compar", "vs", "diff", "écart", "corré"]):
        return "compare"
    if any(k in t for k in ["synth", "résume", "resume", "conclusion", "insight"]):
        return "summarize"
    return "explore"

def make_plan(user_text: str) -> Plan:
    intent = infer_intent(user_text)
    vars_ = _match_variables(user_text)

    tool_calls = [
        ToolCall(tool_name="build_wide_monthly", params={"variable_ids": vars_}),
        ToolCall(tool_name="make_coverage_report", params={}),
        ToolCall(tool_name="make_describe", params={}),
        ToolCall(tool_name="make_timeseries_plot", params={"title": f"Séries mensuelles: {', '.join(vars_)}"}),
    ]

    rationale = f"Intention={intent}. Variables={vars_}. Production: table harmonisée + couverture + descriptif + graphique."
    return Plan(intent=intent, rationale=rationale, tool_calls=tool_calls)
