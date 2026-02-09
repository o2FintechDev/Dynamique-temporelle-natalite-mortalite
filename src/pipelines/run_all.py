# src/pipelines/run_all.py
from __future__ import annotations

from src.agent.schemas import Plan, ToolCall
from src.agent.executor import AgentExecutor
from src.utils.session_state import get_state

Y = "Croissance_Naturelle"

def build_plan() -> Plan:
    variables = [Y]
    calls = [
        ToolCall("step1_load_and_profile", variables, {"y": Y}),
        ToolCall("step2_descriptive",      variables, {"y": Y}),
        ToolCall("step3_stationarity",     variables, {"y": Y, "lags": 24}),
        ToolCall("step4_univariate",       variables, {"y": Y}),
        ToolCall("step5_var",              variables, {"y": Y, "vars_mode": "decomp"}),
        ToolCall("step6_cointegration",    variables, {"y": Y, "vars_mode": "decomp"}),
        ToolCall("step7_anthropology",     variables, {"y": Y}),
    ]
    return Plan(intent="run_all", tool_calls=calls)

def main() -> None:
    plan = build_plan()

    ex = AgentExecutor(user_query="RUN_ALL AnthroDem Lab — Croissance_Naturelle France 1975–2025")
    res = ex.run(plan)

    # Set selected run for narrative tool consistency
    st = get_state()
    st.selected_run_id = res.run_id

    # Export report
    ex2 = AgentExecutor(user_query="EXPORT LaTeX/PDF")
    plan2 = Plan(intent="export", tool_calls=[ToolCall("export_latex_pdf", [Y], {"run_id": res.run_id})])
    ex2.run(plan2)

    print(f"RUN OK: {res.run_id}")

if __name__ == "__main__":
    main()
