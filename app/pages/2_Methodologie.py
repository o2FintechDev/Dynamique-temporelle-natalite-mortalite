from __future__ import annotations

import streamlit as st
from src.utils.session_state import get_state
from src.utils.run_reader import read_manifest, latest_table, latest_figure, read_table_csv

st.title("2 — Méthodologie (diagnostics TS)")

state = get_state()
run_id = state.selected_run_id
if not run_id:
    st.warning("Aucune run sélectionnée.")
    st.stop()

st.caption(f"Run: {run_id}")
st.json(read_manifest(run_id))

st.divider()
st.subheader("ACF / PACF")

p_acf = latest_figure(run_id, "acf_")
p_pacf = latest_figure(run_id, "pacf_")
c1, c2 = st.columns(2)

with c1:
    if p_acf:
        st.image(str(p_acf), use_container_width=True)
        st.caption(str(p_acf))
    else:
        st.info("Figure ACF absente (lancer une run avec eco_diagnostics).")

with c2:
    if p_pacf:
        st.image(str(p_pacf), use_container_width=True)
        st.caption(str(p_pacf))
    else:
        st.info("Figure PACF absente.")

st.divider()
st.subheader("Table ACF/PACF")
p_tab = latest_table(run_id, "acf_pacf_")
if p_tab:
    st.dataframe(read_table_csv(p_tab), use_container_width=True)
    st.caption(str(p_tab))
else:
    st.info("Table ACF/PACF absente.")

st.divider()
st.subheader("Tests de stationnarité (ADF + KPSS si dispo)")
p_adf = latest_table(run_id, "adf_")
if p_adf:
    st.dataframe(read_table_csv(p_adf), use_container_width=True)
    st.caption(str(p_adf))
else:
    st.info("ADF absent.")

p_kpss = latest_table(run_id, "kpss_")
if p_kpss:
    st.dataframe(read_table_csv(p_kpss), use_container_width=True)
    st.caption(str(p_kpss))
