from __future__ import annotations
import streamlit as st
import pandas as pd
from datetime import datetime

st.title("Historique & artefacts de session")

store = st.session_state.get("store", None)
if not store or not store.list():
    st.info("Aucun artefact en session.")
    st.stop()

rows = []
for a in store.list():
    rows.append({
        "artefact_id": a.artefact_id,
        "type": a.type,
        "title": a.title,
        "created_at": datetime.fromtimestamp(a.created_at).isoformat(timespec="seconds"),
        "meta": a.meta,
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.subheader("Détail artefact")
aid = st.text_input("artefact_id", value=store.list()[-1].artefact_id)
if aid:
    try:
        art = store.get(aid)
        st.write(f"**{art.title}** — `{art.artefact_id}` [{art.type}]")
        if art.type == "dataframe":
            st.dataframe(art.payload, use_container_width=True)
        elif art.type == "figure":
            st.plotly_chart(art.payload, use_container_width=True)
        else:
            st.write(art.payload)
    except Exception as e:
        st.error(str(e))
