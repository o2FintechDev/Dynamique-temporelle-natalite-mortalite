from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.settings import settings
from src.utils.session_state import get_session
from src.utils.paths import ensure_dirs
from src.agent.tools import ToolContext
from src.econometrics.diagnostics import acf_pacf_artefacts, stationarity_tests_artefacts, decide_ts_ds

st.set_page_config(page_title="Méthodologie", layout="wide")
st.title("Méthodologie — Diagnostics économétriques")

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

var = st.selectbox(
    "Variable cible",
    options=list(df.columns),
    index=0,
)

st.subheader("Série (niveau)")
st.line_chart(df[var])

run_dirs = ensure_dirs(settings.outputs_dir, sess.current_run_id)
ctx = ToolContext(run_id=sess.current_run_id, run_dirs=run_dirs, memory={"df_ms": df})

col1, col2, col3 = st.columns(3)
do_acf = col1.button("ACF/PACF")
do_tests = col2.button("Tests stationnarité (ADF/PP)")
do_decide = col3.button("Décision TS vs DS")

if do_acf:
    artefacts = acf_pacf_artefacts(ctx, var=var, lags=48)
    st.success(f"Artefacts ACF/PACF générés: {len(artefacts)}")
    for a in artefacts:
        st.write(f"- {a.artifact_id} | {a.name} | {a.path}")
        if a.kind == "figure":
            st.image(a.path)

if do_tests:
    artefacts = stationarity_tests_artefacts(ctx, var=var)
    st.success(f"Artefacts tests générés: {len(artefacts)}")
    for a in artefacts:
        st.write(f"- {a.artifact_id} | {a.name} | {a.path}")
        if a.kind == "table":
            st.dataframe(pd.read_csv(a.path))

if do_decide:
    decision, artefact = decide_ts_ds(ctx, var=var)
    st.success(f"Décision: {decision}")
    st.write(f"- {artefact.artifact_id} | {artefact.name} | {artefact.path}")
    st.json(json.loads(Path(artefact.path).read_text(encoding="utf-8")))

st.caption("Tous les résultats sont stockés en artefacts dans app/outputs/runs/<run_id>/artefacts/.")
