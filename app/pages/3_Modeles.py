from __future__ import annotations

import streamlit as st
from src.utils.session_state import get_state
from src.utils.run_reader import read_manifest, latest_table, read_table_csv

st.title("3 — Modèles (univarié / multivarié)")

state = get_state()
run_id = state.selected_run_id
if not run_id:
    st.warning("Aucune run sélectionnée.")
    st.stop()

st.caption(f"Run: {run_id}")
st.json(read_manifest(run_id))

st.divider()
st.subheader("Grid ARIMA (AIC/BIC)")
p_grid = latest_table(run_id, "univariate_grid_")
if p_grid:
    st.dataframe(read_table_csv(p_grid), use_container_width=True)
    st.caption(str(p_grid))
else:
    st.info("Grid ARIMA absent (lancer une run avec eco_modelisation).")

st.divider()
st.subheader("VAR — sélection de lag")
p_var = latest_table(run_id, "var_selection")
if p_var:
    st.dataframe(read_table_csv(p_var), use_container_width=True)
    st.caption(str(p_var))
else:
    st.info("VAR selection absente (si multivarié non lancé).")
