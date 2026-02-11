# app/pages/5_Analyse_Anthropologique.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.utils.session_state import get_state
from src.utils.run_reader import get_run_files, read_manifest, read_metric_json, RunManager, read_table_from_artefact


PAGE_ID = "5_Analyse_Anthropologique"
PAGE_TITLE = "5 — Analyse anthropologique"

TODD_METRIC_KEY = "m.anthro.todd_analysis"

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


def _render_analysis_block(run_id: str) -> None:
    st.markdown("### Bloc d’analyse (Todd)")
    p = RunManager.get_artefact_path("m.anthro.todd_analysis", run_id=run_id)
    if not p:
        st.info("Aucune analyse anthropologique persistée pour ce run.")
        return
    payload = read_metric_json(p)
    if isinstance(payload, dict) and "markdown" in payload:
        st.markdown(payload["markdown"])
    else:
        st.json(payload)

def _render_tables(run_id: str, items: List[Dict[str, Any]]) -> None:
    if not items:
        return

    rendered_any = False

    for it in items:
        # On tente de lire la table ; si ça échoue, on skip sans afficher de bloc
        try:
            df = read_table_from_artefact(run_id, it)
        except Exception:
            continue

        # Si dataframe vide / None, on skip aussi
        if df is None or (hasattr(df, "empty") and df.empty):
            continue

        # À partir du moment où on a une table valide, on affiche
        rendered_any = True
        key = it.get("key", "")
        st.subheader(key or "table")
        st.dataframe(df, width="stretch")

    # Si aucune table n'a pu être affichée, on ne montre rien (pas de "Aucune table…")
    if not rendered_any:
        return


def _render_figures(run_id: str, items: List[Dict[str, Any]]) -> None:
    # Garde uniquement les figures dont le fichier existe vraiment
    existing: List[Dict[str, Any]] = []
    for it in items:
        rel = it.get("path", "")
        if not rel:
            continue
        p = _abs_path(run_id, rel)
        if p.exists():
            existing.append(it)

    # Si aucune figure réelle, on n'affiche rien (pas de titre)
    if not existing:
        return

    st.markdown("### Figures annexes")
    for it in existing:
        st.subheader(it.get("key", "figure"))
        p = _abs_path(run_id, it.get("path", ""))
        st.image(str(p), width="stretch")


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)

    run_id = _get_run_id()
    if not run_id:
        st.warning("Aucun run sélectionné.")
        return

    m = read_manifest(run_id) or {}
    st.caption(f"Run: {run_id}")

    _render_analysis_block(run_id)

    # Si tu veux aussi router des artefacts anthropo via page=5_Analyse_Anthropologique
    _render_figures(run_id, _filter_items(m, "figures"))
    _render_tables(run_id, _filter_items(m, "tables"))

    mets = _filter_items(m, "metrics")
    mets = [it for it in mets if it.get("key") != TODD_METRIC_KEY]
    if mets:
        st.markdown("### Métriques annexes")
        for it in mets:
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


if __name__ == "__main__":
    main()
