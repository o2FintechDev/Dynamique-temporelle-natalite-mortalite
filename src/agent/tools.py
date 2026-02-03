from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from src.data_api import CachedHttpClient, FredClient, InseeClient, VARIABLES, ApiError
from src.data_pipeline import align_monthly, coverage_report, basic_describe
from src.visualization.charts import ts_line_chart
from src.utils import get_logger

log = get_logger("agent.tools")

@dataclass
class ToolContext:
    http: CachedHttpClient
    fred: FredClient
    insee: InseeClient

def load_variable(ctx: ToolContext, variable_id: str) -> pd.DataFrame:
    spec = VARIABLES.get(variable_id)
    if not spec:
        raise ValueError(f"Variable inconnue: {variable_id}")

    if spec.provider == "fred":
        return ctx.fred.get_series_observations(spec.provider_key, frequency="m")
    if spec.provider == "insee_bdm":
        return ctx.insee.get_bdm_series(spec.provider_key)
    raise ValueError(f"Provider non supporté: {spec.provider}")

def build_wide_monthly(ctx: ToolContext, variable_ids: list[str]) -> pd.DataFrame:
    dfs = {}
    for vid in variable_ids:
        dfs[vid] = load_variable(ctx, vid)
    wide = align_monthly(dfs)
    return wide

def make_coverage_report(wide: pd.DataFrame) -> pd.DataFrame:
    return coverage_report(wide)

def make_describe(wide: pd.DataFrame) -> pd.DataFrame:
    return basic_describe(wide)

def make_timeseries_plot(wide: pd.DataFrame, title: str):
    return ts_line_chart(wide, title=title)
