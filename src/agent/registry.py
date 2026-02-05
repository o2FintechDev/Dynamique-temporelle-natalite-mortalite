# src/agent/registry.py
from __future__ import annotations

from typing import Any, Dict, Callable

from src.agent.executor import StepSpec
from src.agent.tools import get_tool, Y_CANON

ToolWrapper = Callable[[Dict[str, Any]], Dict[str, Any]]

def _wrap_tool(tool_name: str) -> ToolWrapper:
    """
    Wrap un tool TOOL_REGISTRY (signature: fn(*, variables, y, **params))
    -> tool(step_ctx) pour AgentExecutor.
    """
    fn = get_tool(tool_name)

    def _wrapped(step_ctx: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(step_ctx.get("params") or {})
        y = params.pop("y", Y_CANON)
        variables = params.pop("variables", [y])

        # export: run_id obligatoire, y/variables ignorés
        return fn(variables=variables, y=y, **params)

    return _wrapped

STEP_REGISTRY: dict[str, StepSpec] = {
    "step1_load_and_profile": StepSpec("step1_load_and_profile", {}, _wrap_tool("step1_load_and_profile"), {}),
    "step2_descriptive":      StepSpec("step2_descriptive",      {}, _wrap_tool("step2_descriptive"),      {}),
    "step3_stationarity":     StepSpec("step3_stationarity",     {}, _wrap_tool("step3_stationarity"),     {}),
    "step4_univariate":       StepSpec("step4_univariate",       {}, _wrap_tool("step4_univariate"),       {}),
    "step5_var":              StepSpec("step5_var",              {}, _wrap_tool("step5_var"),              {}),
    "step6_cointegration":    StepSpec("step6_cointegration",    {}, _wrap_tool("step6_cointegration"),    {}),
    "step7_anthropology":     StepSpec("step7_anthropology",     {}, _wrap_tool("step7_anthropology"),     {}),
    "export_latex_pdf":       StepSpec("export_latex_pdf",       {}, _wrap_tool("export_latex_pdf"),       {}),
}