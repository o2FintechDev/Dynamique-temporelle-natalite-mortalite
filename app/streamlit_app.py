# app/streamlit_app.py
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import streamlit as st

from src.utils.run_context import init_run_context
from src.utils.session_state import get_state, set_current_run_id, set_last_query
from src.utils.run_reader import list_runs, read_manifest
from src.utils.logger import get_logger

from src.agent.schemas import Plan, ToolCall
from src.agent.executor import AgentExecutor

# NEW (J7)
from src.agent.intent import classify_intent
from src.agent.planner import make_plan

log = get_logger("app.streamlit_app")


ALL_COLS = [
    "taux_naissances",
    "taux_décès",
    "Croissance_Naturelle",
    "Nb_mariages",
    "IPC",
    "Masse_Monétaire",
]


def make_run_id(user_query: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    h = hashlib.sha1(user_query.encode("utf-8")).hexdigest()[:10]
    return f"{ts}_{h}"


def build_plan(run_type: str, target_y: str, predictors_x: list[str]) -> Plan:
    """
    Construit un Plan "manuel" conforme au contrat.
    (Le mode Auto utilise make_plan() directement)
    """
    y = target_y
    x = [v for v in predictors_x if v != y]

    if run_type == "Exploration complète":
        return Plan(
            intent="exploration",
            page_targets=["1_Exploration"],
            tool_calls=[
                ToolCall(
                    tool_name="pipeline_load_and_profile",
                    variables=ALL_COLS,
                    params={},
                ),
            ],
            notes="Launcher: exploration complète",
        )

    if run_type == "Méthodologie seule (ACF/PACF/ADF)":
        return Plan(
            intent="methodologie",
            page_targets=["2_Methodologie"],
            tool_calls=[
                ToolCall(
                    tool_name="eco_diagnostics",
                    variables=[y],
                    params={"y": y, "lags": 24},
                ),
            ],
            notes="Launcher: diagnostics méthodologie",
        )

    if run_type == "Modélisation (ARIMA + VAR + Granger)":
        return Plan(
            intent="modelisation",
            page_targets=["3_Modeles"],
            tool_calls=[
                ToolCall(
                    tool_name="eco_modelisation",
                    variables=[y] + x,
                    params={"y": y, "x": x},
                ),
            ],
            notes="Launcher: modélisation complète",
        )

    if run_type == "Résultats (Cointegration + IRF/FEVD)":
        vars_ = [y] + x if x else [y]
        return Plan(
            intent="resultats",
            page_targets=["4_Resultats"],
            tool_calls=[
                ToolCall(
                    tool_name="eco_resultats",
                    variables=vars_,
                    params={"vars": vars_},
                ),
            ],
            notes="Launcher: résultats",
        )

    if run_type == "Anthropologie (Todd)":
        return Plan(
            intent="anthropologie",
            page_targets=["5_Analyse_Anthropologique"],
            tool_calls=[
                ToolCall(
                    tool_name="narrative_anthropology",
                    variables=[y],
                    params={"y": y},
                ),
            ],
            notes="Launcher: anthropologie",
        )

    # fallback safe
    return Plan(
        intent="exploration",
        page_targets=["1_Exploration"],
        tool_calls=[ToolCall(tool_name="pipeline_load_and_profile", variables=ALL_COLS, params={})],
        notes="Launcher fallback",
    )


st.set_page_config(page_title="AnthroDem Lab", layout="wide")
st.title("AnthroDem Lab — Agent IA interprétatif (offline)")

state = get_state()

with st.sidebar:
    st.header("Runs")
    runs = list_runs()

    default_idx = 0
    if state.selected_run_id and state.selected_run_id in runs:
        default_idx = runs.index(state.selected_run_id)

    selected = st.selectbox("Run sélectionnée", options=runs, index=default_idx if runs else 0)
    if runs:
        state.selected_run_id = selected

    st.divider()
    st.header("Launcher")

    run_type = st.radio(
        "Type d'analyse",
        options=[
            "Auto (intent + planner)",
            "Exploration complète",
            "Méthodologie seule (ACF/PACF/ADF)",
            "Modélisation (ARIMA + VAR + Granger)",
            "Résultats (Cointegration + IRF/FEVD)",
            "Anthropologie (Todd)",
        ],
        index=0,
    )

    user_query = st.text_area("Requête (audit)", value=state.last_user_query or "", height=110)

    # Champs manuels (désactivables si Auto)
    manual_enabled = run_type != "Auto (intent + planner)"
    st.caption("Paramètres manuels (ignorés en mode Auto)")

    target_y = st.selectbox("Variable cible (Y)", options=ALL_COLS, index=0, disabled=not manual_enabled)

    predictors_x = st.multiselect(
        "Prédicteurs (X) — utilisés si requis",
        options=[c for c in ALL_COLS if c != target_y],
        default=["IPC", "Masse_Monétaire"] if target_y not in ["IPC", "Masse_Monétaire"] else ["Nb_mariages"],
        disabled=not manual_enabled,
    )

    launch = st.button("Lancer l'analyse", type="primary")


if launch:
    # user_query obligatoire en Auto (sinon pas d’intent)
    uq = user_query.strip()
    if run_type == "Auto (intent + planner)" and not uq:
        st.error("En mode Auto, la requête ne peut pas être vide.")
        st.stop()

    # fallback texte si manuel sans requête
    if not uq:
        uq = f"{run_type} | y={target_y} | x={predictors_x}"

    run_id = make_run_id(uq)

    init_run_context(run_id)
    set_current_run_id(run_id)
    set_last_query(uq)

    # Plan
    if run_type == "Auto (intent + planner)":
        intent = classify_intent(uq)
        plan = make_plan(intent, uq)
        # audit: injecte la requête dans les notes
        plan.notes = (plan.notes or "") + f" | auto_intent={intent}"
    else:
        plan = build_plan(run_type, target_y, predictors_x)

    exe = AgentExecutor(user_query=uq)
    try:
        res = exe.run(plan)
        st.success(f"Run créée: {res.run_id} | intent={res.intent}")
        state.selected_run_id = res.run_id
    except Exception as e:
        st.error(str(e))

st.divider()
if state.selected_run_id:
    st.subheader("Manifest (run sélectionnée)")
    st.json(read_manifest(state.selected_run_id))
else:
    st.info("Aucune run disponible. Lance une analyse via le Launcher.")
