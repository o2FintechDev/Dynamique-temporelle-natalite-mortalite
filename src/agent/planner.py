from __future__ import annotations
from src.agent.schemas import Plan, ToolCall, Intent

DEFAULT_VARS = ["taux_naissances", "taux_décès", "Croissance Naturelle"]

def make_plan(intent: Intent, user_text: str) -> Plan:
    if intent in ("unknown", "explore"):
        return Plan(intent="explore", calls=[
            ToolCall(tool="load_data"),
            ToolCall(tool="coverage_report"),
            ToolCall(tool="plot_timeseries", params={"vars": DEFAULT_VARS}),
            ToolCall(tool="describe_stats", params={"vars": DEFAULT_VARS}),
        ])

    if intent == "diagnose":
        return Plan(intent="diagnose", calls=[
            ToolCall(tool="load_data"),
            ToolCall(tool="coverage_report"),
            ToolCall(tool="acf_pacf", params={"var": DEFAULT_VARS[0]}),
            ToolCall(tool="stationarity_tests", params={"var": DEFAULT_VARS[0]}),
            ToolCall(tool="ts_ds_decision", params={"var": DEFAULT_VARS[0]}),
        ])

    if intent == "model":
        return Plan(intent="model", calls=[
            ToolCall(tool="load_data"),
            ToolCall(tool="fit_univariate_models", params={"var": DEFAULT_VARS[0], "max_p": 3, "max_q": 3, "max_d": 1}),
            ToolCall(tool="fit_var_model", params={"vars": DEFAULT_VARS, "max_lag": 6}),
        ])

    if intent == "summarize":
        return Plan(intent="summarize", calls=[
            ToolCall(tool="load_data"),
            ToolCall(tool="coverage_report"),
        ])

    if intent == "export":
        return Plan(intent="export", calls=[
            ToolCall(tool="export_latex"),
            ToolCall(tool="build_pdf"),
        ])

    return Plan(intent="unknown", calls=[ToolCall(tool="load_data"), ToolCall(tool="coverage_report")])
