from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from src.agent.schemas import Plan, Artefact
from src.agent.tools import (
    ToolContext,
    tool_load_data,
    tool_coverage_report,
    tool_plot_timeseries,
    tool_describe_stats,
)
from src.utils.logger import get_logger
from src.utils.settings import settings

log = get_logger("agent.executor", settings.log_level)

TOOL_REGISTRY = {
    "load_data": lambda ctx, **p: tool_load_data(ctx),
    "coverage_report": lambda ctx, **p: tool_coverage_report(ctx),
    "plot_timeseries": lambda ctx, **p: tool_plot_timeseries(ctx, **p),
    "describe_stats": lambda ctx, **p: tool_describe_stats(ctx, **p),

    # econometrics
    "acf_pacf": lambda ctx, **p: __import__("src.agent.tools", fromlist=["tool_acf_pacf"]).tool_acf_pacf(ctx, **p),
    "stationarity_tests": lambda ctx, **p: __import__("src.agent.tools", fromlist=["tool_stationarity_tests"]).tool_stationarity_tests(ctx, **p),
    "ts_ds_decision": lambda ctx, **p: __import__("src.agent.tools", fromlist=["tool_ts_ds_decision"]).tool_ts_ds_decision(ctx, **p),
    "fit_univariate_models": lambda ctx, **p: __import__("src.agent.tools", fromlist=["tool_fit_univariate_models"]).tool_fit_univariate_models(ctx, **p),
    "fit_var_model": lambda ctx, **p: __import__("src.agent.tools", fromlist=["tool_fit_var_model"]).tool_fit_var_model(ctx, **p),
}


def execute_plan(plan: Plan, ctx: ToolContext) -> list[Artefact]:
    artefacts: list[Artefact] = []
    for call in plan.calls:
        tool = call.tool
        params = call.params or {}
        try:
            if tool not in TOOL_REGISTRY:
                log.warning(json.dumps({"event": "tool_missing", "tool": tool}, ensure_ascii=False))
                continue
            out = TOOL_REGISTRY[tool](ctx, **params)
            artefacts.extend(out)
        except Exception as e:
            # fallback offline: on log et on continue
            log.error(json.dumps({"event": "tool_error", "tool": tool, "error": str(e)}, ensure_ascii=False))
            continue
    return artefacts
