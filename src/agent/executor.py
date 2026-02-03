from __future__ import annotations
from typing import Any
from src.agent.schemas import Plan, Artifact
from src.agent import tools as T

TOOL_REGISTRY = {
    "load_dataset": lambda state, **kw: T.tool_load_dataset(state),
    "describe_variable": lambda state, **kw: T.tool_describe_variable(state, kw["var"]),
    "plot_series": lambda state, **kw: T.tool_plot_series(state, kw["var"]),
    "plot_compare": lambda state, **kw: T.tool_plot_compare(state, kw["var1"], kw["var2"]),
    "compute_correlation": lambda state, **kw: T.tool_compute_correlation(state, kw["var1"], kw["var2"]),
    "coverage_report": lambda state, **kw: T.tool_coverage_report(state),
    "missingness_table": lambda state, **kw: T.tool_missingness_table(state),
    "key_metrics_pack": lambda state, **kw: T.tool_key_metrics_pack(state, kw["vars"]),
}

def run_plan(plan: Plan,_
