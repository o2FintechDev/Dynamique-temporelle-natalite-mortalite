from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd

from src.data_pipeline.loader import load_local_excel
from src.data_pipeline.harmonize import harmonize_monthly_index
from src.data_pipeline.coverage_report import coverage_report
from src.visualization.charts import save_timeseries_png, FigureSpec
from src.visualization.tables import save_table_csv
from src.utils.logger import get_logger
from src.utils.settings import settings
from src.agent.schemas import Artefact

log = get_logger("agent.tools", settings.log_level)

@dataclass
class ToolContext:
    run_id: str
    run_dirs: any  # RunPaths
    memory: dict

def _next_id(ctx: ToolContext, prefix: str) -> str:
    n = ctx.memory.setdefault("_counters", {}).setdefault(prefix, 0) + 1
    ctx.memory["_counters"][prefix] = n
    return f"{prefix}_{n:03d}"

def tool_load_data(ctx: ToolContext) -> list[Artefact]:
    raw = load_local_excel()
    df_ms, meta = harmonize_monthly_index(raw, "Date")
    ctx.memory["df_ms"] = df_ms
    ctx.memory["harmonize_meta"] = meta

    # artefact metric meta
    aid = _next_id(ctx, "metric")
    path = ctx.run_dirs.metrics_dir / f"{aid}_harmonize_meta.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return [
        Artefact(
            artefact_id=aid,
            kind="metric",
            name="harmonize_meta",
            path=str(path),
            meta=meta,
        )
    ]

def tool_coverage_report(ctx: ToolContext) -> list[Artefact]:
    df_ms: pd.DataFrame = ctx.memory["df_ms"]
    table, meta = coverage_report(df_ms)

    aid_t = _next_id(ctx, "table")
    p_csv = ctx.run_dirs.tables_dir / f"{aid_t}_coverage.csv"
    save_table_csv(table, p_csv)

    aid_m = _next_id(ctx, "metric")
    p_meta = ctx.run_dirs.metrics_dir / f"{aid_m}_coverage_meta.json"
    p_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return [
        Artefact(artefact_id=aid_t, kind="table", name="coverage_report", path=str(p_csv), meta={"shape": list(table.shape)}),
        Artefact(artefact_id=aid_m, kind="metric", name="coverage_meta", path=str(p_meta), meta=meta),
    ]

def tool_plot_timeseries(ctx: ToolContext, vars: list[str]) -> list[Artefact]:
    df_ms: pd.DataFrame = ctx.memory["df_ms"]
    artefacts: list[Artefact] = []
    for v in vars:
        s = df_ms[v]
        aid = _next_id(ctx, "fig")
        p = ctx.run_dirs.figures_dir / f"{aid}_{v}.png"
        save_timeseries_png(s, p, FigureSpec(title=f"Série mensuelle (niveau) — {v}", xlabel="Date", ylabel=v))
        artefacts.append(Artefact(artefact_id=aid, kind="figure", name=f"timeseries_{v}", path=str(p), meta={"var": v}))
    return artefacts

def tool_describe_stats(ctx: ToolContext, vars: list[str]) -> list[Artefact]:
    df_ms: pd.DataFrame = ctx.memory["df_ms"]
    desc = df_ms[vars].describe().T
    aid = _next_id(ctx, "table")
    p = ctx.run_dirs.tables_dir / f"{aid}_describe.csv"
    save_table_csv(desc.reset_index(names="variable"), p)
    return [Artefact(artefact_id=aid, kind="table", name="describe_stats", path=str(p), meta={"vars": vars})]

# --- econometrics wrappers (artefacts) ---
from src.econometrics.diagnostics import acf_pacf_artefacts, stationarity_tests_artefacts, decide_ts_ds
from src.econometrics.univariate import fit_univariate_grid_artefacts
from src.econometrics.multivariate import fit_var_artefacts

def tool_acf_pacf(ctx: ToolContext, var: str, lags: int = 48) -> list[Artefact]:
    return acf_pacf_artefacts(ctx, var=var, lags=lags)

def tool_stationarity_tests(ctx: ToolContext, var: str) -> list[Artefact]:
    return stationarity_tests_artefacts(ctx, var=var)

def tool_ts_ds_decision(ctx: ToolContext, var: str) -> list[Artefact]:
    _, a = decide_ts_ds(ctx, var=var)
    return [a]

def tool_fit_univariate_models(ctx: ToolContext, var: str, max_p: int = 3, max_q: int = 3, max_d: int = 1) -> list[Artefact]:
    return fit_univariate_grid_artefacts(ctx, var=var, max_p=max_p, max_q=max_q, max_d=max_d)

def tool_fit_var_model(ctx: ToolContext, vars: list[str], max_lag: int = 6) -> list[Artefact]:
    return fit_var_artefacts(ctx, vars=vars, max_lag=max_lag)
