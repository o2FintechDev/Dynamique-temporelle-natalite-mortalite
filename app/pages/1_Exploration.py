# pages/1_Exploration.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.utils.session_state import get_state
from src.utils.run_reader import get_run_files, read_manifest, RunManager, read_metric_json


PAGE_ID = "1_Exploration"
PAGE_TITLE = "1 — Exploration"


def _get_run_id() -> str | None:
    state = get_state()
    rid = getattr(state, "selected_run_id", None)
    if rid:
        return rid
    return st.session_state.get("run_id")


def _load_manifest(run_id: str) -> Dict[str, Any]:
    m = read_manifest(run_id)
    return m or {}


def _run_root(run_id: str) -> Path:
    rf = get_run_files(run_id)
    return Path(rf.root)


def _filter_items(m: Dict[str, Any], kind: str) -> List[Dict[str, Any]]:
    items = (m.get("artefacts", {}) or {}).get(kind, []) or []
    return [it for it in items if it.get("page") == PAGE_ID]


def _abs_path(run_id: str, rel: str) -> Path:
    return _run_root(run_id) / rel


def _render_metrics(run_id: str, items: List[Dict[str, Any]]) -> None:
    if not items:
        st.info("Aucune métrique pour cette page.")
        return

    for it in items:
        key = it.get("key", "")
        rel = it.get("path", "")
        st.subheader(key or "metric")
        p = _abs_path(run_id, rel)
        try:
            payload = read_metric_json(p)
        except Exception as e:
            st.error(f"Lecture métrique impossible: {rel} ({e})")
            continue

        if isinstance(payload, dict) and "markdown" in payload:
            st.markdown(payload["markdown"])
        else:
            st.json(payload)


def _render_tables(run_id: str, items: List[Dict[str, Any]]) -> None:
    if not items:
        st.info("Aucune table pour cette page.")
        return

    for it in items:
        key = it.get("key", "")
        rel = it.get("path", "")
        st.subheader(key or "table")
        p = _abs_path(run_id, rel)
        try:
            df = pd.read_csv(p, index_col=0)
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Lecture table impossible: {rel} ({e})")


def _render_figures(run_id: str, items: List[Dict[str, Any]]) -> None:
    if not items:
        st.info("Aucune figure pour cette page.")
        return

    for it in items:
        key = it.get("key", "")
        rel = it.get("path", "")
        st.subheader(key or "figure")
        p = _abs_path(run_id, rel)
        if not p.exists():
            st.error(f"Figure introuvable: {rel}")
            continue
        st.image(str(p), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)

    run_id = _get_run_id()
    if not run_id:
        st.warning("Aucun run sélectionné.")
        return

    m = _load_manifest(run_id)

    st.caption(f"Run: {run_id}")

    # Step meta
    steps = m.get("steps", {}) or {}
    s1 = steps.get("step1_load_and_profile", {}) or {}
    st.markdown("### Statut")
    st.write(
        {
            "step1_load_and_profile": {
                "status": s1.get("status"),
                "ts": s1.get("ts"),
                "summary": s1.get("summary"),
            }
        }
    )

    st.markdown("### Note")
    p_note = RunManager.get_artefact_path("m.note.step1", run_id=run_id)
    if p_note:
        payload = read_metric_json(p_note)
        if isinstance(payload, dict) and "markdown" in payload:
            st.markdown(payload["markdown"])
        else:
            st.json(payload)
    else:
        st.info("Note step1 non disponible.")

    st.markdown("### Tables")
    _render_tables(run_id, _filter_items(m, "tables"))

    st.markdown("### Figures")
    _render_figures(run_id, _filter_items(m, "figures"))

    st.markdown("### Métriques")
    _render_metrics(run_id, _filter_items(m, "metrics"))


if __name__ == "__main__":
    main()
