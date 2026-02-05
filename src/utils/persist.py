# src/utils/persist.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import json

import pandas as pd
import matplotlib.figure as mpl_fig

from src.utils.run_writer import RunWriter
from src.utils.logger import get_logger
from src.visualization.tables import save_table_csv_and_tex


log = get_logger("utils.persist")


def _safe_filename(label: str) -> str:
    return label.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _relpath(root: Path, p: Path) -> str:
    try:
        return str(p.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def _json_safe(x: Any) -> Any:
    """
    Rend un objet sérialisable JSON sans exploser.
    - dict/list/str/int/float/bool/None: OK
    - autres: str(x)
    """
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    return str(x)


def persist_outputs(
    rw: RunWriter,
    step_name: str,
    outputs: Optional[Dict[str, Any]],
    *,
    page: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Persiste figures/tables/metrics/models et enregistre dans manifest avec la clé 'page'.
    """
    outputs = outputs or {}

    # ROUTING: si rien n’est fourni, on ne casse pas le filtrage UI
    page = page or "UNROUTED"

    run_root = rw.paths.run_dir
    saved: Dict[str, Any] = {"figures": [], "tables": [], "metrics": [], "models": []}

    # -------- FIGURES (.png) --------
    figs = outputs.get("figures") or {}
    if not isinstance(figs, dict):
        log.warning("outputs['figures'] ignoré (type=%s)", type(figs))
        figs = {}

    for label, fig in figs.items():
        if not isinstance(fig, mpl_fig.Figure):
            log.warning("Figure ignorée (type=%s) label=%s", type(fig), label)
            continue

        fn = _safe_filename(label) + ".png"
        out_path = rw.paths.figures_dir / fn
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        rel = _relpath(run_root, out_path)

        rw.register_artefact(
            "figures",
            label,
            rel,
            page=page,
            meta={"step": step_name, "format": "png"},
        )
        saved["figures"].append({"key": label, "path": rel, "page": page})

    # -------- TABLES (.csv) --------
    tables = outputs.get("tables") or {}
    if not isinstance(tables, dict):
        log.warning("outputs['tables'] ignoré (type=%s)", type(tables))
        tables = {}

    for label, df in tables.items():
        if not isinstance(df, pd.DataFrame):
            log.warning("Table ignorée (type=%s) label=%s", type(df), label)
            continue

        base = _safe_filename(label)

        # 1) CSV (debug/data)
        csv_path = rw.paths.tables_dir / (base + ".csv")

        # 2) TEX (Overleaf: \input)
        csv_path, tex_path = save_table_csv_and_tex(
            df,
            csv_path,
            caption="",   # caption géré dans latex_report.py
            label="",
            float_format="{:.3f}",
        )

        rel_tex = _relpath(run_root, tex_path)
        rel_csv = _relpath(run_root, csv_path)

        # IMPORTANT: manifest pointe vers TEX
        rw.register_artefact(
            "tables",
            label,
            rel_tex,
            page=page,
            meta={
                "step": step_name,
                "format": "tex",
                "csv_path": rel_csv,
                "nrows": int(df.shape[0]),
                "ncols": int(df.shape[1]),
            },
        )
        saved["tables"].append({"key": label, "path": rel_tex, "page": page, "csv_path": rel_csv})

    # -------- METRICS (.json) --------
    metrics = outputs.get("metrics") or {}
    if not isinstance(metrics, dict):
        log.warning("outputs['metrics'] ignoré (type=%s)", type(metrics))
        metrics = {}

    for label, payload in metrics.items():
        fn = _safe_filename(label) + ".json"
        out_path = rw.paths.metrics_dir / fn

        safe_payload = _json_safe(payload)
        out_path.write_text(json.dumps(safe_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        rel = _relpath(run_root, out_path)

        rw.register_artefact(
            "metrics",
            label,
            rel,
            page=page,
            meta={"step": step_name, "format": "json"},
        )
        saved["metrics"].append({"key": label, "path": rel, "page": page})

    # -------- MODELS (.txt) --------
    models = outputs.get("models") or {}
    if not isinstance(models, dict):
        log.warning("outputs['models'] ignoré (type=%s)", type(models))
        models = {}

    for label, obj in models.items():
        fn = _safe_filename(label) + ".txt"
        out_path = rw.paths.models_dir / fn
        out_path.write_text(str(obj), encoding="utf-8")
        rel = _relpath(run_root, out_path)

        rw.register_artefact(
            "models",
            label,
            rel,
            page=page,
            meta={"step": step_name, "format": "txt"},
        )
        saved["models"].append({"key": label, "path": rel, "page": page})

    log.info(
        "persist_outputs step=%s page=%s | figs=%d tbl=%d met=%d mdl=%d",
        step_name,
        page,
        len(saved["figures"]),
        len(saved["tables"]),
        len(saved["metrics"]),
        len(saved["models"]),
    )

    return saved
