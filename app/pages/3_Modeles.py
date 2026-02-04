import streamlit as st
from src.utils.session_state import get_state
from src.utils.run_reader import RunManager, read_table_csv, read_metric_json

st.title("Modèles")

run_id = get_state().selected_run_id or RunManager.get_latest_run_id()
if not run_id:
    st.warning("Aucun run.")
    st.stop()

def show_tbl(label: str, title: str):
    p = RunManager.get_artefact_path(label, run_id=run_id)
    if p:
        st.subheader(title)
        st.dataframe(read_table_csv(p), use_container_width=True)
    else:
        st.warning(f"Artefact absent: {label}")

def show_fig(label: str, caption: str):
    p = RunManager.get_artefact_path(label, run_id=run_id)
    if p:
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.warning(f"Artefact absent: {label}")

show_tbl("tbl.uni.selection", "Sélection ARIMA (AIC/BIC)")
show_tbl("tbl.uni.resid_diag", "Diagnostics résidus (Ljung-Box / Normalité / ARCH)")
show_tbl("tbl.uni.memory", "Analyse de mémoire (R/S, Hurst)")

for lab, cap in [("fig.uni.fit","Fit ARIMA"),("fig.uni.resid_acf","ACF résidus"),("fig.uni.qq","QQ-plot")]:
    show_fig(lab, cap)

m_best = RunManager.get_artefact_path("m.uni.best", run_id=run_id)
if m_best:
    best = read_metric_json(m_best).get("best", {})
    st.markdown(
        f"**Interprétation (courte)** : Le modèle retenu minimise l’AIC (ordre={best.get('order')}). "
        "Les diagnostics résidus valident (ou invalident) l’adéquation via Ljung-Box/ARCH/normalité."
    )

p = RunManager.get_artefact_path("m.note.stepX", run_id=run_id)
if p:
    st.markdown(read_metric_json(p).get("markdown",""))