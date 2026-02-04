# app/pages/4_Resultats.py
from __future__ import annotations

import streamlit as st

from src.utils.session_state import get_state
from src.utils.run_reader import RunManager, read_manifest, read_table_csv


st.title("4 — Résultats (cointegration / IRF / FEVD)")

state = get_state()
run_id = state.selected_run_id
if not run_id:
    st.warning("Aucune run sélectionnée.")
    st.stop()

manifest = read_manifest(run_id)
st.caption(f"Run: {run_id}")
st.json(manifest)

st.divider()
st.subheader("Cointegration — Engle-Granger (pairwise)")

p_eg = RunManager.get_artefact_path("engle_granger", run_id=run_id)
if p_eg:
    st.dataframe(read_table_csv(p_eg), use_container_width=True)
    st.caption(str(p_eg))
else:
    st.info("Donnée non disponible pour ce type de run (Engle-Granger). Lance une run 'Résultats'.")

st.subheader("Cointegration — Johansen (trace)")

p_j = RunManager.get_artefact_path("johansen_trace", run_id=run_id)
if p_j:
    st.dataframe(read_table_csv(p_j), use_container_width=True)
    st.caption(str(p_j))
else:
    st.info("Donnée non disponible pour ce type de run (Johansen).")

st.divider()
st.subheader("IRF")

p_irf = RunManager.get_artefact_path("irf", run_id=run_id)
if p_irf:
    st.image(str(p_irf), use_container_width=True)
    st.caption(str(p_irf))
else:
    st.info("Donnée non disponible pour ce type de run (IRF).")

st.subheader("FEVD (résumé)")

p_fevd = RunManager.get_artefact_path("fevd", run_id=run_id)
if p_fevd:
    st.dataframe(read_table_csv(p_fevd), use_container_width=True)
    st.caption(str(p_fevd))
else:
    st.info("Donnée non disponible pour ce type de run (FEVD).")