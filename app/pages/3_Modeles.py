#  app/pages/3_Modeles.py
from __future__ import annotations

import streamlit as st

from src.utils.session_state import get_state
from src.utils.run_reader import RunManager, read_table_csv, read_manifest, read_metric_json


st.title("3 — Modèles (univarié / multivarié)")

state = get_state()
run_id = state.selected_run_id
if not run_id:
    st.warning("Aucune run sélectionnée.")
    st.stop()

manifest = read_manifest(run_id)
st.caption(f"Run: {run_id}")
st.json(manifest)

# Déduction simple de y (cible) : 1ère variable du manifest si présent
y = None
vars_ = manifest.get("variables") or []
if vars_:
    y = vars_[0]

if not y:
    st.info("Variable cible inconnue dans le manifest. Lance une run de modélisation avec variables explicites.")
    st.stop()

st.divider()
st.subheader("ARIMA — Grid (AIC/BIC)")

label_grid = f"univariate_grid_{y}"
p_grid = RunManager.get_artefact_path(label_grid, run_id=run_id)
if p_grid:
    st.dataframe(read_table_csv(p_grid), width='stretch')
    st.caption(str(p_grid))
else:
    st.info("Donnée non disponible pour ce type de run (grid ARIMA). Lance une run avec 'Modélisation'.")

st.subheader("ARIMA — Meilleur modèle (metric)")

label_best = f"univariate_best_{y}"
p_best = RunManager.get_artefact_path(label_best, run_id=run_id)
if p_best:
    metric = read_metric_json(p_best)
    st.json(metric)
    st.caption(str(p_best))
else:
    st.info("Donnée non disponible pour ce type de run (meilleur ARIMA).")

st.divider()
st.subheader("VAR — sélection de lag")

label_var = "var_selection"
p_var = RunManager.get_artefact_path(label_var, run_id=run_id)
if p_var:
    st.dataframe(read_table_csv(p_var), width='stretch')
    st.caption(str(p_var))
else:
    st.info("Donnée non disponible pour ce type de run (VAR).")

st.subheader("Granger — pairwise (optionnel)")

label_granger = "granger_pairwise"
p_g = RunManager.get_artefact_path(label_granger, run_id=run_id)
if p_g:
    st.dataframe(read_table_csv(p_g), width='stretch')
    st.caption(str(p_g))
else:
    st.info("Donnée non disponible pour ce type de run (Granger).")