from __future__ import annotations
import streamlit as st
import pandas as pd

st.title("Qualité & couverture des données")

store = st.session_state.get("store", None)
if not store or not store.list():
    st.info("Aucun artefact en session. Utiliser d’abord la page Accueil / Agent.")
    st.stop()

cov_art = next((a for a in reversed(store.list()) if a.meta.get("kind") == "coverage"), None)
desc_art = next((a for a in reversed(store.list()) if a.meta.get("kind") == "describe"), None)

if cov_art:
    st.subheader(f"Data coverage report [{cov_art.artefact_id}]")
    st.dataframe(cov_art.payload, use_container_width=True)
else:
    st.info("Pas de coverage report en session.")

if desc_art:
    st.subheader(f"Statistiques descriptives [{desc_art.artefact_id}]")
    st.dataframe(desc_art.payload, use_container_width=True)
else:
    st.info("Pas de statistiques descriptives en session.")
