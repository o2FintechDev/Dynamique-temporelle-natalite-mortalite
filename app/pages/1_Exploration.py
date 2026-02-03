import streamlit as st
from src.utils import ensure_session_state

ensure_session_state(st)
st.title("Exploration des données")
st.warning("Étape 0 : exploration inactive (chargement Excel à l’étape 1).")
