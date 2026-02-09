# src/visualization/rendering.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set

import streamlit as st

from src.visualization.ui_labels import pretty_label
from src.utils.run_reader import read_metric_json, read_table_from_artefact
from src.visualization.page_layouts import SectionSpec


def _index_by_key(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for it in items:
        k = it.get("key")
        if k:
            out[k] = it
    return out


def _render_fig(run_root: Path, it: Dict[str, Any]) -> None:
    rel = it.get("path", "")
    p = run_root / rel
    if not p.exists():
        st.error(f"Figure introuvable: {rel}")
        return
    st.image(str(p), width="stretch")


def _render_table(run_id: str, it: Dict[str, Any]) -> None:
    meta = (it.get("meta") or {})
    if meta.get("nrows") == 0 and meta.get("ncols") == 0:
        st.info("Table vide (aucun coefficient significatif selon le filtre appliqué).")
        return
    try:
        df = read_table_from_artefact(run_id, it)
        st.dataframe(df, width="stretch")
    except Exception as e:
        st.error(f"Lecture table impossible: {it.get('path','')} ({e})")

def _render_metric(run_root: Path, it: Dict[str, Any]) -> None:
    rel = it.get("path", "")
    p = run_root / rel
    try:
        payload = read_metric_json(p)
    except Exception as e:
        st.error(f"Lecture métrique impossible: {rel} ({e})")
        return

    if isinstance(payload, dict) and "markdown" in payload:
        st.markdown(payload["markdown"])
    else:
        st.json(payload)


def render_ordered_page(
    *,
    run_id: str,
    run_root: Path,
    figs: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    metrics: List[Dict[str, Any]],
    layout: List[SectionSpec],
    show_unlisted: bool = True,
) -> None:
    fig_by = _index_by_key(figs)
    tbl_by = _index_by_key(tables)

    used_fig: Set[str] = set()
    used_tbl: Set[str] = set()

    # ---- Sections (figures + tables) ----
    for sec in layout:
        st.markdown(f"## {sec.title}")
        if sec.help_md:
            st.caption(sec.help_md)

        # Figures in specified order
        for k in sec.figures:
            it = fig_by.get(k)
            if not it:
                continue
            used_fig.add(k)
            st.subheader(pretty_label(k))
            st.caption(f"`{k}`")
            _render_fig(run_root, it)

        # Tables in specified order
        for k in sec.tables:
            it = tbl_by.get(k)
            if not it:
                continue
            used_tbl.add(k)
            st.subheader(pretty_label(k))
            st.caption(f"`{k}`")
            _render_table(run_id, it)

        st.divider()

    # ---- Optional: show unlisted figs/tables (safety) ----
    if show_unlisted:
        unlisted_figs = [it for it in figs if it.get("key") and it["key"] not in used_fig]
        unlisted_tbls = [it for it in tables if it.get("key") and it["key"] not in used_tbl]

        if unlisted_figs or unlisted_tbls:
            st.markdown("## Autres artefacts")
            st.caption("Artefacts présents dans le run mais non positionnés explicitement dans le layout.")
            if unlisted_figs:
                st.markdown("### Figures")
                for it in unlisted_figs:
                    k = it.get("key", "")
                    st.subheader(pretty_label(k) if k else "Figure")
                    if k:
                        st.caption(f"`{k}`")
                    _render_fig(run_root, it)
            if unlisted_tbls:
                st.markdown("### Tables")
                for it in unlisted_tbls:
                    k = it.get("key", "")
                    st.subheader(pretty_label(k) if k else "Table")
                    if k:
                        st.caption(f"`{k}`")
                    _render_table(run_id, it)
            st.divider()

    # ---- Metrics at end of page ----
    st.markdown("## Métriques")
    if not metrics:
        st.info("Aucune métrrique pour cette page.")
        return

    for it in metrics:
        k = it.get("key", "")
        st.subheader(pretty_label(k) if k else "Métrique")
        if k:
            st.caption(f"`{k}`")
        _render_metric(run_root, it)
