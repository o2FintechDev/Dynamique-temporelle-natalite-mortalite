from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import streamlit as st

from src.utils.run_context import init_run_context
from src.utils.session_state import get_state, set_current_run_id, set_last_query
from src.utils.run_reader import list_runs, read_manifest
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
    st.header("Runs")
    runs = list_runs()
    default_idx = 0
    if state.selected_run_id and state.selected_run_id in runs:
        default_idx = runs.index(state.selected_run_id)

    selected = st.selectbox("Run sélectionnée", options=runs, index=default_idx if runs else 0)
    if runs:
        state.selected_run_id = selected

    st.divider()
    st.subheader("Nouvelle run")
    user_query = st.text_area("Requête", value=state.last_user_query or "", height=120)
    launch = st.button("Lancer une run", type="primary")

if launch:
    run_id = make_run_id(user_query.strip() or "empty_query")
    init_run_context(run_id)
    set_current_run_id(run_id)
    set_last_query(user_query)

    plan = Plan(
        intent="exploration",
        page_targets=["1_Exploration"],
        tool_calls=[
            ToolCall(
                tool_name="pipeline_load_and_profile",
                variables=["taux_naissances", "taux_décès", "Croissance_Naturelle", "Nb_mariages", "IPC", "Masse_Monétaire"],
                params={},
            ),
        ],
        notes="Run J2: pipeline réel + lecture artefacts.",
    )

    exe = AgentExecutor(user_query=user_query)
    try:
        res = exe.run(plan)
        st.success(f"Run créée: {res.run_id}")
        state.selected_run_id = res.run_id
    except Exception as e:
        st.error(str(e))

st.divider()
if state.selected_run_id:
    st.subheader("Manifest (run sélectionnée)")
    st.json(read_manifest(state.selected_run_id))
else:
    st.info("Aucune run disponible. Lance une run.")
