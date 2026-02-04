import streamlit as st
from src.utils.session_state import get_state
from src.utils.run_reader import read_manifest, RunManager

st.title("Historique des artefacts")

run_id = get_state().selected_run_id or RunManager.get_latest_run_id()
if not run_id:
    st.warning("Aucun run.")
    st.stop()

m = read_manifest(run_id)
st.json(m)


