import streamlit as st

from src.utils import ensure_session_state

st.set_page_config(page_title="AnthroDem Lab", layout="wide")
ensure_session_state(st)

st.title("AnthroDem Lab")
st.caption("Exploration & interprétation factuelle (offline par défaut).")

st.markdown(
    """
**Statut (Étape 0)**  
- Repo initialisé  
- Pages Streamlit en place  
- Agent et ingestion : implémentés à l’étape 1
"""
)

st.info("Rendez-vous dans les pages via le menu Streamlit à gauche.")
