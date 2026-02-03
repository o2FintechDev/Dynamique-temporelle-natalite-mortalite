from __future__ import annotations
from src.agent.schemas import Plan, ToolCall
from src.agent.intent import detect_intent

KNOWN_VARS = [
    "Nb_naissances","Nb_décès","solde_naturel","taux_naissances","taux_décès",
    "Croissance_Naturelle","Nb_mariages","IPC","Taux_chômage","Masse_Monétaire","Population"
]

def _pick_vars(text: str) -> list[str]:
    found = []
    for v in KNOWN_VARS:
        if v.lower() in text.lower():
            found.append(v)
    return found

def make_plan(user_text: str) -> Plan:
    intent = detect_intent(user_text)
    vars_ = _pick_vars(user_text)

    calls: list[ToolCall] = [ToolCall(name="load_dataset", args={})]

    if intent == "explore":
        y = vars_[0] if vars_ else "Nb_naissances"
        calls += [
            ToolCall(name="describe_variable", args={"var": y}),
            ToolCall(name="plot_series", args={"var": y}),
        ]
        note = f"Exploration centrée sur {y}."
        return Plan(intent="explore", tool_calls=calls, notes=note)

    if intent == "compare":
        y1 = vars_[0] if len(vars_) >= 1 else "Nb_naissances"
        y2 = vars_[1] if len(vars_) >= 2 else "Nb_décès"
        calls += [
            ToolCall(name="describe_variable", args={"var": y1}),
            ToolCall(name="describe_variable", args={"var": y2}),
            ToolCall(name="plot_compare", args={"var1": y1, "var2": y2}),
            ToolCall(name="compute_correlation", args={"var1": y1, "var2": y2}),
        ]
        note = f"Comparaison {y1} vs {y2}."
        return Plan(intent="compare", tool_calls=calls, notes=note)

    if intent == "quality":
        calls += [
            ToolCall(name="coverage_report", args={}),
            ToolCall(name="missingness_table", args={}),
        ]
        return Plan(intent="quality", tool_calls=calls, notes="Contrôle couverture + valeurs manquantes.")

    # synthesize
    calls += [
        ToolCall(name="coverage_report", args={}),
        ToolCall(name="key_metrics_pack", args={"vars": vars_ or ["Nb_naissances","Nb_décès","IPC","Taux_chômage"]}),
    ]
    return Plan(intent="synthesize", tool_calls=calls, notes="Synthèse basée sur métriques + couverture.")
