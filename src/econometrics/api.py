from __future__ import annotations

from typing import Any
import pandas as pd

from .diagnostics import run_diagnostics_pack
from .univariate import run_univariate_pack
from .multivariate import run_var_pack
from .cointegration import run_cointegration_pack
from .impulse import run_impulse_pack

def diagnostics_pack(df: pd.DataFrame, *, y: str) -> dict[str, Any]:
    return run_diagnostics_pack(df, y=y)

def modelisation_pack(df: pd.DataFrame, *, y: str, x: list[str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out.update(run_univariate_pack(df, y=y))
    if x:
        out.update(run_var_pack(df, vars=[y] + x))
    return out

def resultats_pack(df: pd.DataFrame, *, vars: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out.update(run_cointegration_pack(df, vars=vars))
    out.update(run_impulse_pack(df, vars=vars))
    return out
