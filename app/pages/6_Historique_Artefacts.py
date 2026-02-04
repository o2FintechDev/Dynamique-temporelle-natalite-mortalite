from __future__ import annotations

from pathlib import Path
import streamlit as st

from src.utils.run_reader import list_runs, get_run_files, read_manifest

st.title("6 — Historique Artefacts")

runs = list_runs()
if not runs:
    st.info("Aucune run disponible.")
    st.stop()

run_id = st.selectbox("Choisir une run", options=runs, index=0)
rf = get_run_files(run_id)

st.subheader("Manifest")
st.json(read_manifest(run_id))

st.divider()
st.subheader("Artefacts")

def list_paths(p: Path, pattern: str) -> list[Path]:
    if not p.exists():
        return []
    return sorted(p.glob(pattern))

tabs = st.tabs(["Tables", "Metrics", "Figures", "Models", "Logs"])

with tabs[0]:
    files = list_paths(rf.tables, "table_*.csv")
    st.write(f"{len(files)} fichiers")
    for f in files:
        st.code(str(f), language="text")

with tabs[1]:
    files = list_paths(rf.metrics, "metric_*.json")
    st.write(f"{len(files)} fichiers")
    for f in files:
        st.code(str(f), language="text")

with tabs[2]:
    files = list_paths(rf.figures, "fig_*.png")
    st.write(f"{len(files)} fichiers")
    for f in files:
        st.code(str(f), language="text")

with tabs[3]:
    files = list_paths(rf.models, "model_*.pkl")
    st.write(f"{len(files)} fichiers")
    for f in files:
        st.code(str(f), language="text")

with tabs[4]:
    files = list_paths(rf.logs, "*.log")
    st.write(f"{len(files)} fichiers")
    for f in files:
        st.code(str(f), language="text")
