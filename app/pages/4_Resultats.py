from __future__ import annotations
import pandas as pd
import streamlit as st

from src.utils.settings import settings
from src.utils.session_state import get_session
from src.utils.paths import ensure_dirs
from src.agent.tools import ToolContext
from src.econometrics.cointegration import engle_granger_artefacts, johansen_artefacts
from src.econometrics.impulse import irf_fevd_artefacts

st.set_page_config(page_title="Résultats", layout="wide")
st.title("Résultats — Cointégration, VAR/VECM, IRF/FEVD")

sess = get_session(st)

def load_df_ms() -> pd.DataFrame:
    from src.data_pipeline.loader import load_local_excel
    from src.data_pipeline.harmonize import harmonize_monthly_index
    raw = load_local_excel()
    df_ms, _ = harmonize_monthly_index(raw, "Date")
    return df_ms

if not sess.current_run_id:
    st.info("Aucun run actif. Lance un run depuis Accueil/Agent.")
    st.stop()

df = load_df_ms()
run_dirs = ensure_dirs(settings.outputs_dir, sess.current_run_id)
ctx = ToolContext(run_id=sess.current_run_id, run_dirs=run_dirs, memory={"df_ms": df})

vars_sel = st.multiselect("Variables (système)", options=list(df.columns), default=list(df.columns)[:3])
lags = st.slider("Lags (Johansen / IRF)", 1, 12, 4)

col1, col2, col3 = st.columns(3)
if col1.button("Engle-Granger (pairwise vs 1ère variable)"):
    artefacts = engle_granger_artefacts(ctx, vars=vars_sel)
    st.success(f"Artefacts Engle-Granger: {len(artefacts)}")
    for a in artefacts:
        st.write(f"- {a.artefact_id} | {a.name} | {a.path}")
        if a.kind == "table":
            st.dataframe(pd.read_csv(a.path))

if col2.button("Johansen"):
    artefacts = johansen_artefacts(ctx, vars=vars_sel, det_order=0, k_ar_diff=lags)
    st.success(f"Artefacts Johansen: {len(artefacts)}")
    for a in artefacts:
        st.write(f"- {a.artefact_id} | {a.name} | {a.path}")
        if a.kind == "table":
            st.dataframe(pd.read_csv(a.path))

if col3.button("IRF + FEVD (VAR)"):
    artefacts = irf_fevd_artefacts(ctx, vars=vars_sel, max_lag=min(lags, 8), horizon=24)
    st.success(f"Artefacts IRF/FEVD: {len(artefacts)}")
    for a in artefacts:
        st.write(f"- {a.artefact_id} | {a.name} | {a.path}")
        if a.kind == "figure":
            st.image(a.path)
        if a.kind == "table":
            st.dataframe(pd.read_csv(a.path))
