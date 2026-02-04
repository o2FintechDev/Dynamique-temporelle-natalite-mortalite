# app/pages/6_Historique_Artefacts.py
from __future__ import annotations

from pathlib import Path
import streamlit as st

from src.utils.run_reader import list_runs, get_run_files, read_manifest, RunManager, read_metric_json
from src.agent.tools import get_tool


st.title("6 — Historique Artefacts")

runs = list_runs()
if not runs:
    st.info("Aucune run disponible.")
    st.stop()

run_id = st.selectbox("Choisir une run", options=runs, index=0)
rf = get_run_files(run_id)

st.subheader("Manifest")
manifest = read_manifest(run_id)
st.json(manifest)

st.divider()
st.subheader("Export LaTeX / PDF")

if st.button("Générer Rapport", type="primary"):
    try:
        tool = get_tool("export_latex_pdf")
        # appelle le tool directement (offline) sur le run choisi
        out = tool(variables=[], run_id=run_id)
        st.success("Export terminé (voir metric export_report).")
        st.json(out)
    except Exception as e:
        st.error(str(e))

# Affichage du résultat si présent dans lookup du run exporté (ou si tu l’as généré récemment)
p_export = RunManager.get_artefact_path("export_report", run_id=run_id)
if p_export:
    st.subheader("Résultat export (metric export_report)")
    payload = read_metric_json(p_export)
    st.json(payload)

# Liens directs vers report.tex / report.pdf si présents dans le dossier du run
tex_path = rf.root / "report.tex"
pdf_path = rf.root / "report.pdf"

c1, c2 = st.columns(2)
with c1:
    st.write("report.tex")
    if tex_path.exists():
        st.code(str(tex_path), language="text")
    else:
        st.info("report.tex non généré.")
with c2:
    st.write("report.pdf")
    if pdf_path.exists():
        st.code(str(pdf_path), language="text")
    else:
        st.info("report.pdf non généré (pdflatex absent ou échec).")

st.divider()
st.subheader("Artefacts")

def list_paths(p: Path, pattern: str) -> list[Path]:
    if not p.exists():
        return []
    return sorted(p.glob(pattern))

tabs = st.tabs(["Tables", "Metrics", "Figures", "Models", "Logs"])

with tabs[0]:
    files = list_paths(rf.tables, "table_*.csv")
    st.write(f"{len(files)} fichiers")
    for f in files:
        st.code(str(f), language="text")

with tabs[1]:
    files = list_paths(rf.metrics, "metric_*.json")
    st.write(f"{len(files)} fichiers")
    for f in files:
        st.code(str(f), language="text")

with tabs[2]:
    files = list_paths(rf.figures, "fig_*.png")
    st.write(f"{len(files)} fichiers")
    for f in files:
        st.code(str(f), language="text")

with tabs[3]:
    files = list_paths(rf.models, "model_*.pkl")
    st.write(f"{len(files)} fichiers")
    for f in files:
        st.code(str(f), language="text")

with tabs[4]:
    files = list_paths(rf.logs, "*.log")
    st.write(f"{len(files)} fichiers")
    for f in files:
        st.code(str(f), language="text")
