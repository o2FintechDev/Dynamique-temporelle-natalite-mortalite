from __future__ import annotations
import json
from pathlib import Path
import streamlit as st

from src.utils.settings import settings
from src.utils.session_state import get_session

st.set_page_config(page_title="Historique Artefacts", layout="wide")
st.title("Historique — Runs & Artefacts")

sess = get_session(st)
runs_root = settings.outputs_dir / "runs"
runs_root.mkdir(parents=True, exist_ok=True)

run_ids = sorted([p.name for p in runs_root.iterdir() if p.is_dir()], reverse=True)
if not run_ids:
    st.info("Aucun run disponible.")
    st.stop()

selected = st.selectbox("Run", options=run_ids, index=0)
sess.current_run_id = selected

run_dir = runs_root / selected
manifest_path = run_dir / "manifest.json"

if not manifest_path.exists():
    st.warning("manifest.json absent (run incomplet).")
    st.stop()

man = json.loads(manifest_path.read_text(encoding="utf-8"))
st.subheader("Plan JSON")
st.code(json.dumps(man.get("plan", {}), ensure_ascii=False, indent=2), language="json")

st.subheader("Artefacts")
artefacts = man.get("artefacts", [])
for a in artefacts:
    st.write(f"- **{a['artefact_id']}** | {a['kind']} | {a['name']}")
    p = Path(a["path"])
    if p.exists() and a["kind"] == "figure":
        st.image(str(p))
    if p.exists() and a["kind"] in ("table", "metric", "file"):
        st.download_button(f"Télécharger {p.name}", data=p.read_bytes(), file_name=p.name, key=f"dl_{selected}_{a['artefact_id']}")

st.subheader("Narration auditée")
st.text(man.get("narrative", ""))

st.subheader("Audit")
st.json(man.get("audit", {}))
