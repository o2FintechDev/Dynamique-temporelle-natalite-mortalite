# pages/4_Resultats.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.utils.session_state import get_state
from src.utils.run_reader import get_run_files, read_manifest, read_metric_json, read_table_from_artefact


PAGE_ID = "4_Resultats"
PAGE_TITLE = "4 — Résultats"


def _get_run_id() -> str | None:
    state = get_state()
    rid = getattr(state, "selected_run_id", None)
    if rid:
        return rid
    return st.session_state.get("run_id")


def _run_root(run_id: str) -> Path:
    rf = get_run_files(run_id)
    return Path(rf.root)


def _abs_path(run_id: str, rel: str) -> Path:
    return _run_root(run_id) / rel


def _filter_items(m: Dict[str, Any], kind: str) -> List[Dict[str, Any]]:
    items = (m.get("artefacts", {}) or {}).get(kind, []) or []
    return [it for it in items if it.get("page") == PAGE_ID]


def _render_figures(run_id: str, items: List[Dict[str, Any]]) -> None:
    if not items:
        st.info("Aucune figure pour cette page.")
        return
    for it in items:
        st.subheader(it.get("key", "figure"))
        p = _abs_path(run_id, it.get("path", ""))
        if p.exists():
            st.image(str(p), width='stretch')
        else:
            st.error(f"Figure introuvable: {it.get('path')}")


def _render_tables(run_id: str, items: List[Dict[str, Any]]) -> None:
    if not items:
        st.info("Aucune table pour cette page.")
        return

    for it in items:
        key = it.get("key", "")
        st.subheader(key or "table")

        try:
            df = read_table_from_artefact(run_id, it)
            st.dataframe(df, width="stretch")
        except Exception as e:
            st.error(f"Lecture table impossible: {it.get('path','')} ({e})")


def _render_metrics(run_id: str, items: List[Dict[str, Any]]) -> None:
    if not items:
        st.info("Aucune métrique pour cette page.")
        return
    for it in items:
        st.subheader(it.get("key", "metric"))
        p = _abs_path(run_id, it.get("path", ""))
        try:
            payload = read_metric_json(p)
        except Exception as e:
            st.error(f"Lecture métrique impossible: {it.get('path')} ({e})")
            continue
        if isinstance(payload, dict) and "markdown" in payload:
            st.markdown(payload["markdown"])
        else:
            st.json(payload)


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)

    run_id = _get_run_id()
    if not run_id:
        st.warning("Aucun run sélectionné.")
        return

    m = read_manifest(run_id) or {}
    st.caption(f"Run: {run_id}")

    st.markdown("### Figures")
    _render_figures(run_id, _filter_items(m, "figures"))

    st.markdown("### Tables")
    _render_tables(run_id, _filter_items(m, "tables"))

    st.markdown("### Métriques")
    _render_metrics(run_id, _filter_items(m, "metrics"))


if __name__ == "__main__":
    main()
