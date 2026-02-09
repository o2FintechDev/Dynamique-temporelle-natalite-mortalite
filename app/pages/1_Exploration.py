# app/pages/1_Exploration.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from src.visualization.page_layouts import PAGE_LAYOUTS
from src.visualization.rendering import render_ordered_page

from src.utils.session_state import get_state
from src.utils.run_reader import (
    get_run_files,
    read_manifest,
    RunManager,
    read_metric_json,
)

PAGE_ID = "1_Exploration"
PAGE_TITLE = "1 — Exploration"


def _get_run_id() -> str | None:
    state = get_state()
    rid = getattr(state, "selected_run_id", None)
    if rid:
        return rid
    return st.session_state.get("run_id")


def _run_root(run_id: str) -> Path:
    rf = get_run_files(run_id)
    return Path(rf.root)


def _filter_items(m: Dict[str, Any], kind: str) -> List[Dict[str, Any]]:
    items = (m.get("artefacts", {}) or {}).get(kind, []) or []
    return [it for it in items if it.get("page") == PAGE_ID]


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)

    run_id = _get_run_id()
    if not run_id:
        st.warning("Aucun run sélectionné.")
        return

    m = read_manifest(run_id) or {}
    st.caption(f"Run: {run_id}")

    # --- Statut Step 1 (inchangé) ---
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


    # --- Rendu ordonné (figures / tables) + métriques à la fin ---
    layout = PAGE_LAYOUTS.get(PAGE_ID, [])
    render_ordered_page(
        run_id=run_id,
        run_root=_run_root(run_id),
        figs=_filter_items(m, "figures"),
        tables=_filter_items(m, "tables"),
        metrics=_filter_items(m, "metrics"),
        layout=layout,
        show_unlisted=True,
    )


if __name__ == "__main__":
    main()
