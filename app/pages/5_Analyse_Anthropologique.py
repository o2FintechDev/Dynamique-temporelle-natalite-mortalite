from __future__ import annotations

import streamlit as st
from src.utils.session_state import get_state
from src.utils.run_reader import read_manifest, latest_metric, read_metric_json

st.title("5 — Analyse anthropologique (cadrée, offline)")

state = get_state()
run_id = state.selected_run_id
if not run_id:
    st.warning("Aucune run sélectionnée.")
    st.stop()

st.caption(f"Run: {run_id}")
st.json(read_manifest(run_id))

st.divider()
st.subheader("Narration tracée (metric JSON)")

p = latest_metric(run_id, "anthropology_markdown")
if p:
    payload = read_metric_json(p)
    st.markdown(payload.get("markdown", ""))
    st.caption(str(p))
else:
    st.info("Narration absente (lancer une run avec narrative_anthropology).")
