from __future__ import annotations

from pathlib import Path
import streamlit as st

from src.utils.session_state import get_state
from src.utils.run_reader import (
    get_run_files,
    read_manifest,
    read_table_csv,
    latest_table,
    latest_metric,
    read_metric_json,
)

st.title("1 — Exploration (lecture artefacts)")

state = get_state()
run_id = state.selected_run_id

if not run_id:
    st.warning("Aucune run sélectionnée. Lance une run dans la page principale.")
    st.stop()

st.caption(f"Run: {run_id}")
st.json(read_manifest(run_id))

st.divider()
st.subheader("Tables — stats descriptives / manquants / couverture")

p_desc = latest_table(run_id, "desc_stats")
p_miss = latest_table(run_id, "missing_report")
p_cov = latest_table(run_id, "coverage_report")

cols = st.columns(3)

with cols[0]:
    st.markdown("**Desc stats**")
    if p_desc:
        st.dataframe(read_table_csv(p_desc), use_container_width=True)
        st.caption(str(p_desc))
    else:
        st.info("Artefact absent: desc_stats")

with cols[1]:
    st.markdown("**Missing report**")
    if p_miss:
        st.dataframe(read_table_csv(p_miss), use_container_width=True)
        st.caption(str(p_miss))
    else:
        st.info("Artefact absent: missing_report")

with cols[2]:
    st.markdown("**Coverage report**")
    if p_cov:
        st.dataframe(read_table_csv(p_cov), use_container_width=True)
        st.caption(str(p_cov))
    else:
        st.info("Artefact absent: coverage_report")

st.divider()
st.subheader("Metrics — meta dataset")

p_meta = latest_metric(run_id, "dataset_meta")
if p_meta:
    st.json(read_metric_json(p_meta))
    st.caption(str(p_meta))
else:
    st.info("Artefact absent: dataset_meta")
