from __future__ import annotations
import uuid
import pandas as pd

from src.data_pipeline.loader import load_dataset
from src.data_pipeline.quality import per_column_missingness
from src.data_pipeline.coverage_report import data_coverage_report
from src.visualization.charts import line_series, compare_two

def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"

def tool_load_dataset(state: dict) -> dict:
    df, meta = load_dataset()
    state["df"] = df
    return {
        "artifact_id": _id("meta"),
        "kind": "json",
        "title": "Métadonnées dataset",
        "payload": meta.__dict__,
    }

def tool_describe_variable(state: dict, var: str) -> dict:
    df: pd.DataFrame = state["df"]
    s = df[var]
    desc = {
        "variable": var,
        "n_total": int(len(s)),
        "n_non_null": int(s.notna().sum()),
        "min": (float(s.min()) if s.notna().any() else None),
        "max": (float(s.max()) if s.notna().any() else None),
        "mean": (float(s.mean()) if s.notna().any() else None),
        "start": (s.dropna().index.min().date().isoformat() if s.notna().any() else None),
        "end": (s.dropna().index.max().date().isoformat() if s.notna().any() else None),
    }
    return {"artifact_id": _id("metrics"), "kind": "metrics", "title": f"Stats descriptives: {var}", "payload": desc}

def tool_plot_series(state: dict, var: str) -> dict:
    df: pd.DataFrame = state["df"]
    fig = line_series(df, var, title=f"{var} (mensuel)")
    return {"artifact_id": _id("fig"), "kind": "figure", "title": f"Série temporelle: {var}", "payload": fig}

def tool_plot_compare(state: dict, var1: str, var2: str) -> dict:
    df: pd.DataFrame = state["df"]
    fig = compare_two(df, var1, var2, title=f"{var1} vs {var2} (mensuel)")
    return {"artifact_id": _id("fig"), "kind": "figure", "title": f"Comparaison: {var1} vs {var2}", "payload": fig}

def tool_compute_correlation(state: dict, var1: str, var2: str) -> dict:
    df: pd.DataFrame = state["df"]
    d = df[[var1, var2]].dropna()
    corr = float(d[var1].corr(d[var2])) if len(d) >= 3 else None
    payload = {"var1": var1, "var2": var2, "n_pairs": int(len(d)), "pearson_corr": corr}
    return {"artifact_id": _id("metrics"), "kind": "metrics", "title": f"Corrélation: {var1} vs {var2}", "payload": payload}

def tool_coverage_report(state: dict) -> dict:
    df: pd.DataFrame = state["df"]
    rep = data_coverage_report(df)
    return {"artifact_id": _id("json"), "kind": "json", "title": "Data coverage report", "payload": rep}

def tool_missingness_table(state: dict) -> dict:
    df: pd.DataFrame = state["df"]
    tab = per_column_missingness(df)
    return {"artifact_id": _id("table"), "kind": "table", "title": "Valeurs manquantes par variable", "payload": tab}

def tool_key_metrics_pack(state: dict, vars: list[str]) -> dict:
    df: pd.DataFrame = state["df"]
    out = []
    for v in vars:
        if v in df.columns:
            s = df[v]
            out.append({
                "variable": v,
                "n_non_null": int(s.notna().sum()),
                "mean": float(s.mean()) if s.notna().any() else None,
                "min": float(s.min()) if s.notna().any() else None,
                "max": float(s.max()) if s.notna().any() else None,
                "start": s.dropna().index.min().date().isoformat() if s.notna().any() else None,
                "end": s.dropna().index.max().date().isoformat() if s.notna().any() else None,
            })
    return {"artifact_id": _id("table"), "kind": "table", "title": "Pack métriques clés", "payload": pd.DataFrame(out)}
