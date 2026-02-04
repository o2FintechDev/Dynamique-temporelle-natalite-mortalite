from __future__ import annotations

import streamlit as st

from src.utils.run_reader import RunManager, read_table_csv, read_metric_json


st.title("2 — Méthodologie (lecture stricte des artefacts)")

run_id = RunManager.get_latest_run_id()
if not run_id:
    st.warning("Aucune donnée disponible. Lancez une analyse depuis l'accueil.")
    st.stop()

manifest = RunManager.load_manifest(run_id)
st.caption(f"Run: {run_id}")
st.json(manifest)

st.divider()
st.subheader("ACF / PACF")

# Labels attendus (issus des slugs utilisés dans executor: label = slug)
# Exemple: y='taux_naissances' => 'acf_taux_naissances', 'pacf_taux_naissances'
# Si ton y est dynamique, tu peux le lire depuis manifest['variables'] ou faire un select.
y = None
vars_ = manifest.get("variables") or []
# heuristique: prend la première variable si présente
if vars_:
    y = vars_[0]

if not y:
    st.info("Variable cible inconnue dans le manifest. Lance une run avec variables explicites.")
    st.stop()

label_acf = f"acf_{y}"
label_pacf = f"pacf_{y}"
label_tab = f"acf_pacf_{y}"
label_adf = f"adf_{y}"
label_lb = f"ljungbox_diff_{y}"

c1, c2 = st.columns(2)

with c1:
    p = RunManager.get_artefact_path(label_acf, run_id=run_id)
    if p:
        st.image(str(p), use_container_width=True)
        st.caption(str(p))
    else:
        st.info("Donnée non disponible pour ce type de run (ACF).")

with c2:
    p = RunManager.get_artefact_path(label_pacf, run_id=run_id)
    if p:
        st.image(str(p), use_container_width=True)
        st.caption(str(p))
    else:
        st.info("Donnée non disponible pour ce type de run (PACF).")

st.divider()
st.subheader("Table ACF/PACF")
p = RunManager.get_artefact_path(label_tab, run_id=run_id)
if p:
    st.dataframe(read_table_csv(p), use_container_width=True)
    st.caption(str(p))
else:
    st.info("Donnée non disponible pour ce type de run (table ACF/PACF).")

st.subheader("ADF (3 spécifications)")
p = RunManager.get_artefact_path(label_adf, run_id=run_id)
if p:
    st.dataframe(read_table_csv(p), use_container_width=True)
    st.caption(str(p))
else:
    st.info("Donnée non disponible pour ce type de run (ADF).")

st.subheader("Ljung-Box sur diff (proxy)")
p = RunManager.get_artefact_path(label_lb, run_id=run_id)
if p:
    st.dataframe(read_table_csv(p), use_container_width=True)
    st.caption(str(p))
else:
    st.info("Donnée non disponible pour ce type de run (Ljung-Box).")
