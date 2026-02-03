from __future__ import annotations
import streamlit as st
import pandas as pd

from src.visualization.charts import ts_line_chart

st.title("Exploration des données")

store = st.session_state.get("store", None)
if not store or not store.list():
    st.info("Aucun artefact en session. Utiliser d’abord la page Accueil / Agent.")
    st.stop()

# récupérer la dernière table wide
wide_art = next((a for a in reversed(store.list()) if a.meta.get("kind") == "wide"), None)
if not wide_art:
    st.info("Aucune table harmonisée trouvée. Lance une demande via l’agent.")
    st.stop()

wide = wide_art.payload
st.subheader(f"Table harmonisée [{wide_art.artefact_id}]")
st.dataframe(wide.tail(24), use_container_width=True)

st.subheader("Visualisation")
fig = ts_line_chart(wide, title="Séries mensuelles (session)")
st.plotly_chart(fig, use_container_width=True)
