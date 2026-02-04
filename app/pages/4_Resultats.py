from __future__ import annotations

import streamlit as st
from src.utils.session_state import get_state
from src.utils.run_reader import read_manifest, latest_table, latest_figure, read_table_csv

st.title("4 — Résultats (cointegration / IRF / FEVD)")

state = get_state()
run_id = state.selected_run_id
if not run_id:
    st.warning("Aucune run sélectionnée.")
    st.stop()

st.caption(f"Run: {run_id}")
st.json(read_manifest(run_id))

st.divider()
st.subheader("Cointegration — Engle-Granger (pairwise)")
p_eg = latest_table(run_id, "engle_granger")
if p_eg:
    st.dataframe(read_table_csv(p_eg), use_container_width=True)
    st.caption(str(p_eg))
else:
    st.info("Engle-Granger absent (lancer eco_resultats).")

st.subheader("Cointegration — Johansen (trace)")
p_j = latest_table(run_id, "johansen_trace")
if p_j:
    st.dataframe(read_table_csv(p_j), use_container_width=True)
    st.caption(str(p_j))
else:
    st.info("Johansen absent.")

st.divider()
st.subheader("IRF")
p_irf = latest_figure(run_id, "irf")
if p_irf:
    st.image(str(p_irf), use_container_width=True)
    st.caption(str(p_irf))
else:
    st.info("IRF absent.")

st.subheader("FEVD (résumé)")
p_fevd = latest_table(run_id, "fevd")
if p_fevd:
    st.dataframe(read_table_csv(p_fevd), use_container_width=True)
    st.caption(str(p_fevd))
else:
    st.info("FEVD absent.")
