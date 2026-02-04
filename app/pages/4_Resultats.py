import streamlit as st
from src.utils.session_state import get_state
from src.utils.run_reader import RunManager, read_table_csv, read_metric_json

st.title("Résultats")

run_id = get_state().selected_run_id or RunManager.get_latest_run_id()
if not run_id:
    st.warning("Aucun run.")
    st.stop()

def show_tbl(label: str, title: str):
    p = RunManager.get_artefact_path(label, run_id=run_id)
    if p:
        st.subheader(title)
        st.dataframe(read_table_csv(p), width='content')
    else:
        st.warning(f"Artefact absent: {label}")

def show_fig(label: str, caption: str):
    p = RunManager.get_artefact_path(label, run_id=run_id)
    if p:
        st.image(str(p), caption=caption, width='content')
    else:
        st.warning(f"Artefact absent: {label}")

show_tbl("tbl.var.lag_selection", "Sélection VAR(p)")
show_tbl("tbl.var.granger", "Tests de Granger (composantes)")
show_tbl("tbl.var.fevd", "FEVD")
show_fig("fig.var.irf", "IRF (VAR)")

show_tbl("tbl.coint.eg", "Engle–Granger (indicatif)")
show_tbl("tbl.coint.johansen", "Johansen (rang)")
show_tbl("tbl.coint.var_vs_vecm_choice", "Choix VAR diff vs VECM")
show_tbl("tbl.vecm.params", "Paramètres VECM (si estimé)")

m_tsds = RunManager.get_artefact_path("m.diag.ts_vs_ds", run_id=run_id)
if m_tsds:
    verdict = read_metric_json(m_tsds).get("verdict")
    st.markdown(
        f"**Interprétation (courte)** : La décision stationnarité conclut **{verdict}** (TS vs DS) "
        "sur la base ADF/PP/lecture bande. Les résultats VAR/VECM décrivent la dynamique interne des composantes."
    )
p = RunManager.get_artefact_path("m.note.stepX", run_id=run_id)
if p:
    st.markdown(read_metric_json(p).get("markdown",""))