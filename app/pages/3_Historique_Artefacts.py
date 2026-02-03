import streamlit as st
from src.utils import ensure_session_state

ensure_session_state(st)
st.title("Historique & artefacts")

st.subheader("Historique chat")
if st.session_state.history:
    for msg in st.session_state.history:
        st.write(f"**{msg['role']}**: {msg['content']}")
else:
    st.caption("Aucun message.")

st.subheader("Artefacts")
if st.session_state.artifacts:
    for a in st.session_state.artifacts:
        st.write(f"- [{a.kind}] {a.title} ({a.artifact_id})")
else:
    st.caption("Aucun artefact.")
