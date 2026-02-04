from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import streamlit as st

from src.utils.run_context import init_run_context
from src.utils.session_state import get_state, set_current_run_id, set_last_query
from src.utils import get_logger

from src.agent.schemas import Plan, ToolCall
from src.agent.executor import AgentExecutor

log = get_logger("app.streamlit_app")

def make_run_id(user_query: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    h = hashlib.sha1(user_query.encode("utf-8")).hexdigest()[:10]
    return f"{ts}_{h}"

st.set_page_config(page_title="AnthroDem Lab", layout="wide")

st.title("AnthroDem Lab — Agent IA interprétatif (offline)")

state = get_state()

with st.sidebar:
    st.header("Run")
    user_query = st.text_area("Requête", value=state.last_user_query or "", height=120)
    launch = st.button("Lancer une run", type="primary")

if launch:
    run_id = make_run_id(user_query.strip() or "empty_query")
    init_run_context(run_id)
    set_current_run_id(run_id)
    set_last_query(user_query)

    # JOUR 1: plan minimal de smoke-test (pas de logique d’intent avancée ici)
    # -> l’intent classifier et planner seront consolidés J2/J3
    plan = Plan(
        intent="exploration",
        page_targets=["1_Exploration"],
        tool_calls=[
            ToolCall(tool_name="pipeline_load_and_profile", variables=[
                "taux_naissances", "taux_décès", "Croissance_Naturelle", "Nb_mariages", "IPC", "Masse_Monétaire"
            ]),
        ],
        notes="Smoke-test run standardisée J1.",
    )

    exe = AgentExecutor(user_query=user_query)
    try:
        res = exe.run(plan)
        st.success(f"Run créée: {res.run_id}")
        st.json(res.model_dump())
    except Exception as e:
        st.error(str(e))

st.divider()
st.caption("Les pages Streamlit lisent les artefacts dans app/outputs/runs/ (pas de recalcul côté pages).")
