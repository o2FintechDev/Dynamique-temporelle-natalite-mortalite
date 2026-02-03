from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.settings import settings
from src.utils.session_state import get_session

st.set_page_config(page_title="Exploration", layout="wide")
st.title("Exploration — Données & Graphiques")

sess = get_session(st)

def load_df_from_run(run_id: str) -> pd.DataFrame | None:
    # On recharge le DF via Excel (source unique) puis harmonize: reproductible
    from src.data_pipeline.loader import load_local_excel
    from src.data_pipeline.harmonize import harmonize_monthly_index
    try:
        raw = load_local_excel()
        df_ms, _ = harmonize_monthly_index(raw, "Date")
        return df_ms
    except Exception as e:
        st.error(f"Chargement échoué: {e}")
        return None

if not sess.current_run_id:
    st.info("Aucun run actif. Lance un run depuis l’Accueil/Agent.")
    st.stop()

df = load_df_from_run(sess.current_run_id)
if df is None:
    st.stop()

st.subheader("Aperçu (index mensuel MS)")
st.dataframe(df.head(12))

vars_default = ["taux_naissances", "taux_décès", "Croissance_Naturelle"]
st.subheader("Séries (niveau)")
cols = st.columns(3)
for i, v in enumerate(vars_default):
    with cols[i]:
        st.line_chart(df[v])

st.subheader("Décomposition (placeholder MVP)")
st.info("La décomposition STL/seasonal_decompose + artefacts systématiques est intégrée Jour 2 (sans casser l’historique).")
