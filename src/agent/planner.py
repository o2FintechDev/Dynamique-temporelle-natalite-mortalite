# src/agent/planner.py
from __future__ import annotations

import re
from typing import Iterable

from src.agent.schemas import Plan, ToolCall, Intent

# Variables autorisées (contrat data)
ALL_COLS = [
    "taux_naissances",
    "taux_décès",
    "Croissance_Naturelle",
    "Nb_mariages",
    "IPC",
    "Masse_Monétaire",
]

DEFAULT_Y = "taux_naissances"
DEFAULT_X = ["IPC", "Masse_Monétaire"]


def _extract_vars_from_text(text: str) -> list[str]:
    """
    Extraction déterministe: repère les noms de colonnes exacts dans le texte.
    """
    t = (text or "").lower()
    found = []
    for v in ALL_COLS:
        if v.lower() in t:
            found.append(v)
    return found


def _choose_y_x(found: list[str]) -> tuple[str, list[str]]:
    """
    Règles auditables:
    - si une variable explicite est mentionnée, la 1ère = y
    - x = autres variables mentionnées, sinon DEFAULT_X
    - y ne doit pas être dans x
    """
    if found:
        y = found[0]
        x = [v for v in found[1:] if v != y]
        if not x:
            x = [v for v in DEFAULT_X if v != y]
        return y, x
    return DEFAULT_Y, DEFAULT_X.copy()


def make_plan(intent: Intent, user_text: str) -> Plan:
    found = _extract_vars_from_text(user_text)
    y, x = _choose_y_x(found)

    if intent == "exploration":
        return Plan(
            intent="exploration",
            page_targets=["1_Exploration"],
            tool_calls=[
                ToolCall(
                    tool_name="pipeline_load_and_profile",
                    variables=ALL_COLS,
                    params={},
                )
            ],
            notes=f"Planner v2: exploration | y={y} x={x}",
        )

    if intent == "methodologie":
        return Plan(
            intent="methodologie",
            page_targets=["2_Methodologie"],
            tool_calls=[
                ToolCall(
                    tool_name="eco_diagnostics",
                    variables=[y],
                    params={"y": y, "lags": 24},
                )
            ],
            notes=f"Planner v2: methodologie | y={y}",
        )

    if intent == "modelisation":
        return Plan(
            intent="modelisation",
            page_targets=["3_Modeles"],
            tool_calls=[
                ToolCall(
                    tool_name="eco_modelisation",
                    variables=[y] + x,
                    params={"y": y, "x": x},
                )
            ],
            notes=f"Planner v2: modelisation | y={y} x={x}",
        )

    if intent == "resultats":
        vars_ = [y] + x if x else [y]
        return Plan(
            intent="resultats",
            page_targets=["4_Resultats"],
            tool_calls=[
                ToolCall(
                    tool_name="eco_resultats",
                    variables=vars_,
                    params={"vars": vars_},
                )
            ],
            notes=f"Planner v2: resultats | vars={vars_}",
        )

    if intent == "anthropologie":
        return Plan(
            intent="anthropologie",
            page_targets=["5_Analyse_Anthropologique"],
            tool_calls=[
                ToolCall(
                    tool_name="narrative_anthropology",
                    variables=[y],
                    params={"y": y},
                )
            ],
            notes=f"Planner v2: anthropologie | y={y}",
        )

    if intent == "export":
        # export vise la run sélectionnée dans l'UI; on peut aussi le passer explicitement plus tard
        return Plan(
            intent="export",
            page_targets=["6_Historique_Artefacts"],
            tool_calls=[],
            notes="Planner v2: export (déclenchement via page 6 / tool export_latex_pdf)",
        )

    # fallback safe
    return Plan(
        intent="exploration",
        page_targets=["1_Exploration"],
        tool_calls=[ToolCall(tool_name="pipeline_load_and_profile", variables=ALL_COLS, params={})],
        notes="Planner v2: fallback exploration",
    )
