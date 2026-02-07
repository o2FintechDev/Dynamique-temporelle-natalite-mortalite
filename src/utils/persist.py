# src/utils/persist.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import json
import re

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


def _escape_tex(s: Any) -> str:
    """
    Echappement minimal sûr pour captions/labels LaTeX.
    IMPORTANT: ne pas l'utiliser pour des blocs de texte longs (utiliser verbatim).
    """
    if s is None:
        return ""
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _latex_safe_label_id(x: str, *, maxlen: int = 60) -> str:
    """
    ID sûr pour \label{...}:
    - pas d'espaces, pas de backslash, pas de \protect, pas de caractères actifs
    - uniquement [a-zA-Z0-9:_-]
    """
    x = (x or "artefact").strip().lower()
    x = x.replace(" ", "-")
    # remplace tout ce qui n'est pas alnum / : _ - par _
    x = re.sub(r"[^a-z0-9:_-]+", "_", x)
    x = x.strip("_")
    return (x[:maxlen] or "artefact")


def _extract_step_note_markdown(step_name: str, outputs: Dict[str, Any]) -> Optional[str]:
    """
    Récupère une note markdown si présente.
    Convention déjà dans ton code: m.note.step2, m.note.step3, m.note.step4...
    """
    metrics = outputs.get("metrics") or {}
    if not isinstance(metrics, dict):
        return None

    # Tentatives directes: m.note.<step_name> ou m.note.stepX (si step_name contient stepX)
    candidates: list[str] = []

    # ex: step_name="step3_stationarity" -> "m.note.step3"
    m = re.match(r"^(step\d+)", step_name)
    if m:
        candidates.append(f"m.note.{m.group(1)}")

    # ex: "m.note.step3_stationarity" (au cas où)
    candidates.append(f"m.note.{step_name}")

    for k in candidates:
        v = metrics.get(k)
        if isinstance(v, dict) and isinstance(v.get("markdown"), str):
            return v["markdown"]

    # fallback: premier m.note.* rencontré
    for k, v in metrics.items():
        if isinstance(k, str) and k.startswith("m.note.") and isinstance(v, dict) and isinstance(v.get("markdown"), str):
            return v["markdown"]

    return None


def _write_latex_block(
    rw: RunWriter,
    *,
    step_name: str,
    page: str,
    outputs: Dict[str, Any],
    saved: Dict[str, Any],
) -> Optional[Path]:
    """
    Génère un bloc LaTeX auto: latex/blocks/<step_name>.tex

    Hypothèse Overleaf (recommandée):
    - le fichier compilé est main.tex à la racine
    - main.tex fait: \input{latex/master.tex}
    - master.tex fait: \input{blocks/<step>.tex}

    Donc, depuis latex/blocks/*.tex, les artefacts sont à:
    - ../../artefacts/figures/<file>.png
    - ../../artefacts/tables/<file>.tex
    """
    blocks_dir = rw.paths.latex_dir / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)

    block_path = blocks_dir / f"{_safe_filename(step_name)}.tex"

    note_md = _extract_step_note_markdown(step_name, outputs)

    lines: list[str] = []

    # Titre du bloc (pas de \label ici)
    lines.append(r"\section*{" + _escape_tex(step_name) + r"}")
    if page:
        lines.append(r"\textit{Page: " + _escape_tex(page) + r"}\\")
    lines.append("")

    if note_md:
        lines += [r"\begin{verbatim}", note_md, r"\end{verbatim}", ""]

    # -------- FIGURES --------
    figs = saved.get("figures") or []
    if figs:
        lines.append(r"\subsection*{Figures}")
        for f in figs:
            # f["path"] relatif run_root: artefacts/figures/<file>.png
            fname = Path(f.get("path", "")).name
            caption = f.get("key") or fname
            lab_id = _latex_safe_label_id(f"fig:{caption}")

            rel_fig = f"../../artefacts/figures/{fname}"

            lines += [
                r"\begin{figure}[H]",
                r"\includegraphics[width=0.95\linewidth]{"
                + r"\detokenize{" + rel_fig + r"}"
                + r"}",
                r"\caption{" + _escape_tex(caption) + r"}",
                r"\label{" + lab_id + r"}",
                r"\end{figure}",
                "",
            ]

    # -------- TABLES --------
    tbls = saved.get("tables") or []
    if tbls:
        lines.append(r"\subsection*{Tableaux}")
        for t in tbls:
            # manifest pointe vers .tex: artefacts/tables/<file>.tex
            tname = Path(t.get("path", "")).name
            caption = t.get("key") or tname
            lab_id = _latex_safe_label_id(f"tab:{caption}")

            rel_input = f"../../artefacts/tables/{tname}"

            lines += [
                r"\begin{table}[H]",
                r"\caption{" + _escape_tex(caption) + r"}",
                r"\label{" + lab_id + r"}",
                r"\begin{adjustbox}{max width=\linewidth,center}",
                r"\input{" + r"\detokenize{" + rel_input + r"}" + r"}",
                r"\end{adjustbox}",
                r"\end{table}",
                "",
            ]

    # -------- METRICS (liste seulement) --------
    mets = saved.get("metrics") or []
    if mets:
        lines.append(r"\subsection*{Métriques (JSON, non rendues)}")
        for m in mets:
            p = m.get("path", "")
            # afficher un chemin relatif au run_root (lisible)
            lines.append(r"\texttt{" + _escape_tex(p) + r"}\\")
        lines.append("")

    block_path.write_text("\n".join(lines), encoding="utf-8")
    return block_path


def persist_outputs(
    rw: RunWriter,
    step_name: str,
    outputs: Optional[Dict[str, Any]],
    *,
    page: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Persiste figures/tables/metrics/models et enregistre dans manifest avec la clé 'page'.
    + Génère un bloc LaTeX par step dans latex/blocks/<step_name>.tex (auto).
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

    # -------- TABLES (.csv + .tex) --------
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
            caption="",   # caption géré dans latex_report.py / blocs
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

    # -------- LATEX BLOCK (auto) --------
    try:
        block_path = _write_latex_block(
            rw,
            step_name=step_name,
            page=page,
            outputs=outputs,
            saved=saved,
        )
        if block_path is not None:
            rel_block = _relpath(run_root, block_path)
            rw.register_artefact(
                "latex_blocks",
                f"block.{step_name}",
                rel_block,
                page=page,
                meta={"step": step_name, "format": "tex"},
            )
    except Exception as e:
        log.exception("LATEX block generation failed: %s", e)

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
