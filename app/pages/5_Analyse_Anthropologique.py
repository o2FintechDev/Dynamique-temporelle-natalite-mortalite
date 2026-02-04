# app/pages/5_Analyse_Anthropologique.py
from __future__ import annotations

import streamlit as st

from src.utils.session_state import get_state
from src.utils.run_reader import read_manifest, RunManager, read_metric_json

st.title("5 — Analyse anthropologique (cadrée, offline)")

state = get_state()
run_id = state.selected_run_id
if not run_id:
    st.warning("Aucune run sélectionnée.")
    st.stop()

manifest = read_manifest(run_id)
st.caption(f"Run: {run_id}")
st.json(manifest)

st.divider()
st.subheader("Narration tracée (metric JSON via lookup)")

p = RunManager.get_artefact_path("todd_analysis", run_id=run_id)
if p:
    payload = read_metric_json(p)
    st.markdown(payload.get("markdown", ""))
    st.caption(str(p))
else:
    st.info("Donnée non disponible pour ce type de run (todd_analysis). Lance une run 'Anthropologie (Todd)'.")
