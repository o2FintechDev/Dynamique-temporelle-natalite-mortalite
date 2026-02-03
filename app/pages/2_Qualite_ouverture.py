import streamlit as st
from src.utils import ensure_session_state

ensure_session_state(st)
st.title("Qualité & couverture des données")
st.warning("Étape 0 : rapport de couverture à l’étape 1.")
