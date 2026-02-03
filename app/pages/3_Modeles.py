from __future__ import annotations
import pandas as pd
import streamlit as st

from src.utils.settings import settings
from src.utils.session_state import get_session
from src.utils.paths import ensure_dirs
from src.agent.tools import ToolContext
from src.econometrics.univariate import fit_univariate_grid_artefacts
from src.econometrics.multivariate import fit_var_artefacts, granger_artefacts

st.set_page_config(page_title="Modèles", layout="wide")
st.title("Modèles — Estimation & comparaison")

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

st.header("Univarié")
var = st.selectbox("Variable univariée", options=list(df.columns), index=0)
max_p = st.slider("max p (AR)", 0, 6, 3)
max_q = st.slider("max q (MA)", 0, 6, 3)
max_d = st.slider("max d (diff)", 0, 2, 1)

if st.button("Fit grid ARIMA (fallback ARMA/AR/MA)"):
    artefacts = fit_univariate_grid_artefacts(ctx, var=var, max_p=max_p, max_q=max_q, max_d=max_d)
    st.success(f"Artefacts modèles univariés: {len(artefacts)}")
    for a in artefacts:
        st.write(f"- {a.artefact_id} | {a.name} | {a.path}")
        if a.kind == "table":
            st.dataframe(pd.read_csv(a.path))

st.header("Multivarié (VAR)")
target = st.selectbox("Cible VAR", options=list(df.columns), index=0)
exogs = st.multiselect("Variables incluses (endog VAR)", options=list(df.columns), default=[target])
max_lag = st.slider("Lag max", 1, 12, 6)

col1, col2 = st.columns(2)
if col1.button("Fit VAR (sélection AIC/BIC)"):
    artefacts = fit_var_artefacts(ctx, vars=list(dict.fromkeys(exogs)), max_lag=max_lag)
    st.success(f"Artefacts VAR: {len(artefacts)}")
    for a in artefacts:
        st.write(f"- {a.artefact_id} | {a.name} | {a.path}")
        if a.kind == "table":
            st.dataframe(pd.read_csv(a.path))

if col2.button("Causalité de Granger (pairwise)"):
    artefacts = granger_artefacts(ctx, vars=list(dict.fromkeys(exogs)), max_lag=min(max_lag, 6))
    st.success(f"Artefacts Granger: {len(artefacts)}")
    for a in artefacts:
        st.write(f"- {a.artefact_id} | {a.name} | {a.path}")
        if a.kind == "table":
            st.dataframe(pd.read_csv(a.path))
