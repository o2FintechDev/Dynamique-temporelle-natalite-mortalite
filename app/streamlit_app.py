# app/streamlit_app.py
from __future__ import annotations
import json
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]  # remonte de /app vers /
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import streamlit as st
from src.utils.settings import settings
from src.utils.paths import make_run_id, ensure_dirs
from src.utils.session_state import get_session
from src.utils.logger import get_logger
from src.utils.settings import settings
from src.utils.paths import make_run_id, ensure_dirs
from src.utils.session_state import get_session
from src.utils.logger import get_logger

from src.agent.intent import classify_intent
from src.agent.planner import make_plan
from src.agent.executor import execute_plan
from src.agent.tools import ToolContext
from src.narrative.renderer import build_mvp_narrative
from src.narrative.latex_report import export_report_tex

log = get_logger("app", settings.log_level)

st.set_page_config(page_title="AnthroDem Lab", layout="wide")

st.title("AnthroDem Lab — Accueil / Agent")

sess = get_session(st)

with st.sidebar:
    st.subheader("Configuration")
    st.write(f"LLM assisté: **{settings.llm_assisted}** (désactivé par défaut)")
    st.write(f"Data: `{settings.data_path}`")
    st.write(f"Outputs: `{settings.outputs_dir}`")

st.markdown("### Chat utilisateur")
user_text = st.text_area("Demande", value="Explore et génère le coverage report + graphiques de base.", height=90)

colA, colB, colC = st.columns([1, 1, 2])
run_clicked = colA.button("Exécuter")
export_tex_clicked = colB.button("Export LaTeX")
build_pdf_clicked = colB.button("Build PDF")

def persist_manifest(run_dir: Path, plan: dict, artefacts: list[dict], narrative: str, audit: dict) -> Path:
    manifest = {
        "plan": plan,
        "artefacts": artefacts,
        "narrative": narrative,
        "audit": audit,
    }
    p = run_dir / "manifest.json"
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return p

if run_clicked:
    intent = classify_intent(user_text)
    plan = make_plan(intent, user_text).dict()  # FIXED: Changed .model_dump() to .dict()

    payload_for_id = {"user_text": user_text, "intent": plan["intent"], "calls": plan["calls"]}
    run_id = make_run_id(payload_for_id)
    run_dirs = ensure_dirs(settings.outputs_dir, run_id)

    ctx = ToolContext(run_id=run_id, run_dirs=run_dirs, memory={})
    artefacts = execute_plan(make_plan(intent, user_text), ctx)

    narrative, audit = build_mvp_narrative(artefacts)

    # Persist run bundle
    artefacts_dump = [a.dict() for a in artefacts]  # FIXED: Changed .model_dump() to .dict()
    persist_manifest(run_dirs.run_dir, plan, artefacts_dump, narrative, audit)

    sess.current_run_id = run_id
    sess.last_plan = plan
    sess.last_artefacts = artefacts_dump
    sess.last_narrative = narrative

    st.success(f"Run terminé: {run_id}")

st.markdown("### Plan JSON")
st.code(json.dumps(sess.last_plan or {}, ensure_ascii=False, indent=2), language="json")

st.markdown("### Artefacts produits")
if sess.last_artefacts:
    for a in sess.last_artefacts:
        st.write(f"- **{a['artefact_id']}** | {a['kind']} | {a['name']} | `{a['path']}`")
else:
    st.info("Aucun artefact (exécute un run).")

st.markdown("### Synthèse narrative auditée")
if sess.last_narrative:
    st.text(sess.last_narrative)
else:
    st.info("Aucune synthèse (exécute un run).")

def load_current_run_bundle() -> tuple[str, dict] | None:
    if not sess.current_run_id:
        return None
    run_dir = settings.outputs_dir / "runs" / sess.current_run_id
    p = run_dir / "manifest.json"
    if not p.exists():
        return None
    return sess.current_run_id, json.loads(p.read_text(encoding="utf-8"))

def try_build_pdf(run_dir: Path) -> tuple[bool, str]:
    tex = run_dir / "report.tex"
    if not tex.exists():
        return False, "report.tex introuvable."
    try:
        # pdflatex compile in run_dir
        cmd = ["pdflatex", "-interaction=nonstopmode", tex.name]
        proc = subprocess.run(cmd, cwd=str(run_dir), capture_output=True, text=True)
        ok = (proc.returncode == 0) and (run_dir / "report.pdf").exists()
        if ok:
            return True, "PDF généré."
        return False, "pdflatex a échoué (fallback LaTeX conservé)."
    except Exception:
        return False, "pdflatex indisponible (fallback LaTeX conservé)."

bundle = load_current_run_bundle()

if export_tex_clicked:
    if not bundle:
        st.error("Aucun run actif.")
    else:
        run_id, man = bundle
        run_dir = settings.outputs_dir / "runs" / run_id
        # Rehydrate artefacts (dict -> minimal)
        from src.agent.schemas import Artefact as ArtefactModel
        artefacts = [ArtefactModel(**a) for a in man.get("artefacts", [])]
        narrative = man.get("narrative", "")
        audit = man.get("audit", {})

        report_path = export_report_tex(run_dir, run_id, artefacts, narrative, audit)

        st.success(f"Export LaTeX OK: {report_path}")
        st.download_button("Télécharger report.tex", data=report_path.read_bytes(), file_name="report.tex")

if build_pdf_clicked:
    if not bundle:
        st.error("Aucun run actif.")
    else:
        run_id, _ = bundle
        run_dir = settings.outputs_dir / "runs" / run_id
        ok, msg = try_build_pdf(run_dir)
        if ok:
            pdf = run_dir / "report.pdf"
            st.success(msg)
            st.download_button("Télécharger report.pdf", data=pdf.read_bytes(), file_name="report.pdf")
        else:
            st.warning(msg)
            st.info("Action: installer TeX Live/MiKTeX puis relancer 'Build PDF'.")