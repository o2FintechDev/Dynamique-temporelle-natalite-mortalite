# econometrics/univariate.py
from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def _prep_series(df: pd.DataFrame, y: str) -> pd.Series:
    if "Date" in df.columns:
        d = df[["Date", y]].copy()
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d = d.dropna(subset=["Date"]).sort_values("Date")
        s = pd.to_numeric(d[y], errors="coerce")
    else:
        s = pd.to_numeric(df[y], errors="coerce")
    return s.dropna().astype(float)


def run_univariate_pack(
    df: pd.DataFrame,
    *,
    y: str,
    grid: list[tuple[int, int, int]] | None = None,
) -> dict[str, Any]:
    """
    Produit:
      - table univariate_grid_<y> (p,d,q,aic,bic,nobs ou error)
      - metric univariate_best_<y>
      - model best_arima_<y> (statsmodels ARIMAResults) picklable
    """
    s = _prep_series(df, y)
    n = int(len(s))

    if n < 40:
        tab = pd.DataFrame([{"error": f"ARIMA: n_obs trop faible (n={n})"}])
        return {
            "tables": {f"univariate_grid_{y}": tab},
            "metrics": {f"univariate_best_{y}": {"y": y, "n_obs": n, "best_order_aic": None, "best_aic": None}},
        }

    if grid is None:
        grid = [(p, d, q) for p in range(0, 4) for d in range(0, 2) for q in range(0, 4) if not (p == 0 and d == 0 and q == 0)]

    rows = []
    best_order = None
    best_aic = float("inf")
    best_bic = float("inf")
    best_model = None

    yv = s.values

    for (p, d, q) in grid:
        try:
            m = ARIMA(yv, order=(p, d, q), trend="c").fit()
            aic = float(m.aic)
            bic = float(m.bic)
            rows.append({"p": p, "d": d, "q": q, "aic": aic, "bic": bic, "nobs": int(m.nobs)})
            if aic < best_aic:
                best_aic = aic
                best_bic = bic
                best_order = (p, d, q)
                best_model = m
        except Exception as e:
            rows.append({"p": p, "d": d, "q": q, "error": str(e)})

    grid_df = pd.DataFrame(rows)
    if "aic" in grid_df.columns:
        grid_df = grid_df.sort_values("aic", na_position="last").reset_index(drop=True)

    metrics = {
        f"univariate_best_{y}": {
            "y": y,
            "n_obs": n,
            "best_order_aic": best_order,
            "best_aic": None if best_order is None else float(best_aic),
            "best_bic": None if best_order is None else float(best_bic),
        }
    }

    models: dict[str, Any] = {}
    if best_model is not None:
        models[f"best_arima_{y}"] = best_model

    return {"tables": {f"univariate_grid_{y}": grid_df}, "metrics": metrics, "models": models}
