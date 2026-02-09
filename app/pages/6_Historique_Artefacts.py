# app/pages/6_Historique_Artefacts.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from src.utils.session_state import get_state
from src.utils.run_reader import get_run_files, read_manifest, read_metric_json, read_table_from_artefact


PAGE_ID = "6_Historique_Artefacts"
PAGE_TITLE = "6 — Historique des artefacts"


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


def _group_by_page(m: Dict[str, Any]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    arte = m.get("artefacts", {}) or {}
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for kind in ["figures", "tables", "metrics", "models", "latex_blocks"]:
        for it in (arte.get(kind) or []):
            page = it.get("page") or "UNROUTED"
            out.setdefault(page, {}).setdefault(kind, []).append(it)
    return out


def _render_kind(run_id: str, kind: str, items: List[Dict[str, Any]]) -> None:
    if not items:
        st.caption("—")
        return

    if kind == "figures":
        for it in items:
            st.markdown(f"**{it.get('key','figure')}**  \n`{it.get('path','')}`")
            p = _abs_path(run_id, it.get("path", ""))
            if p.exists():
                st.image(str(p), width='stretch')
            else:
                st.error("Figure introuvable")
    elif kind == "tables":
        for it in items:
            st.markdown(f"**{it.get('key','table')}**  \n`{it.get('path','')}`")
            try:
                df = read_table_from_artefact(run_id, it)
                st.dataframe(df, width="stretch")
            except Exception as e:
                st.error(f"Lecture table impossible ({e})")
    elif kind == "metrics":
        for it in items:
            st.markdown(f"**{it.get('key','metric')}**  \n`{it.get('path','')}`")
            p = _abs_path(run_id, it.get("path", ""))
            try:
                payload = read_metric_json(p)
            except Exception as e:
                st.error(f"Lecture métrique impossible ({e})")
                continue
            if isinstance(payload, dict) and "markdown" in payload:
                st.markdown(payload["markdown"])
            else:
                st.json(payload)
    elif kind == "models":
        for it in items:
            st.markdown(f"**{it.get('key','model')}**  \n`{it.get('path','')}`")
            p = _abs_path(run_id, it.get("path", ""))
            if p.exists():
                st.code(p.read_text(encoding="utf-8", errors="replace"), language="text")
            else:
                st.error("Fichier modèle introuvable")
    else:
        for it in items:
            st.markdown(f"**{it.get('key','item')}**  \n`{it.get('path','')}`")


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)

    run_id = _get_run_id()
    if not run_id:
        st.warning("Aucun run sélectionné.")
        return

    m = read_manifest(run_id) or {}
    st.caption(f"Run: {run_id}")

    grouped = _group_by_page(m)

    # Contrôles
    pages = sorted(grouped.keys())
    sel_pages = st.multiselect("Pages à afficher", options=pages, default=pages)

    for page in sel_pages:
        block = grouped.get(page, {})
        with st.expander(f"Page: {page}", expanded=False):
            for kind in ["figures", "tables", "metrics", "models", "latex_blocks"]:
                items = block.get(kind, []) or []
                st.markdown(f"### {kind} ({len(items)})")
                _render_kind(run_id, kind, items)


if __name__ == "__main__":
    main()
