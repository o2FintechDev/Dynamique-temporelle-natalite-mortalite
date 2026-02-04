import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st
from src.utils.session_state import get_state
from src.utils.run_reader import RunManager, read_table_csv, read_metric_json

st.title("Analyse descriptive + décomposition (Étape 2)")

run_id = get_state().selected_run_id or RunManager.get_latest_run_id()
if not run_id:
    st.warning("Aucun run sélectionné.")
    st.stop()

def show_fig(label: str, caption: str):
    p = RunManager.get_artefact_path(label, run_id=run_id)
    if p:
        st.image(str(p), caption=caption, width='content')
    else:
        st.warning(f"Artefact absent: {label}")

def show_tbl(label: str, title: str):
    p = RunManager.get_artefact_path(label, run_id=run_id)
    if p:
        st.subheader(title)
        st.dataframe(read_table_csv(p), width='content')
    else:
        st.warning(f"Artefact absent: {label}")

# Figures
show_fig("fig.desc.level", "Série en niveau")
show_fig("fig.desc.decomp", "Décomposition STL (trend/seasonal/resid)")

# Tables
show_tbl("tbl.desc.summary", "Statistiques descriptives")
show_tbl("tbl.desc.seasonality", "Qualification saisonnalité")

# Analyse persistée step2
p_note = RunManager.get_artefact_path("m.note.step2", run_id=run_id)
if p_note:
    st.markdown(read_metric_json(p_note).get("markdown", ""))
else:
    st.info("Note step2 non disponible (m.note.step2).")

p = RunManager.get_artefact_path("m.note.stepX", run_id=run_id)
if p:
    st.markdown(read_metric_json(p).get("markdown",""))
