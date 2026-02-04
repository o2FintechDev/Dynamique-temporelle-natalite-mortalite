import streamlit as st
from src.utils.session_state import get_state
from src.utils.run_reader import RunManager, read_metric_json

st.title("Analyse anthropologique augmentée")

run_id = get_state().selected_run_id or RunManager.get_latest_run_id()
if not run_id:
    st.warning("Aucun run.")
    st.stop()

p = RunManager.get_artefact_path("m.anthro.todd_analysis", run_id=run_id)
if not p:
    st.warning("Artefact absent: m.anthro.todd_analysis")
    st.stop()

payload = read_metric_json(p)
st.markdown(payload.get("markdown", ""))

p = RunManager.get_artefact_path("m.note.stepX", run_id=run_id)
if p:
    st.markdown(read_metric_json(p).get("markdown",""))
