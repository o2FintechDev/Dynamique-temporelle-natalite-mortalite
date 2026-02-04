# econometrics/univariate.py
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA

from src.agent.schemas import Artefact
from src.agent.tools import ToolContext, _next_id
from src.visualization.tables import save_table_csv


def _get_series(ctx: ToolContext, var: str) -> pd.Series:
    df: pd.DataFrame = ctx.memory["df_ms"]
    if "Date" in df.columns:
        tmp = df[["Date", var]].copy()
        tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
        tmp = tmp.dropna(subset=["Date"]).sort_values("Date")
        s = tmp[var].astype(float)
    else:
        s = df[var].astype(float)
    return s.dropna()


def fit_univariate_grid_artefacts(
    ctx: ToolContext,
    var: str,
    max_p: int = 3,
    max_q: int = 3,
    max_d: int = 1,
) -> list[Artefact]:
    y = _get_series(ctx, var)

    rows = []
    best = {"aic": float("inf"), "bic": float("inf"), "order": None}

    if len(y) < 40:
        tab = pd.DataFrame([{"error": f"ARIMA grid: série trop courte (n={len(y)})"}])
        aid = _next_id(ctx, "table")
        p = ctx.run_dirs.tables_dir / f"{aid}_univariate_grid_{var}.csv"
        save_table_csv(tab, p)
        aid2 = _next_id(ctx, "metric")
        pm = ctx.run_dirs.metrics_dir / f"{aid2}_univariate_best_{var}.json"
        pm.write_text(json.dumps({"best": best, "error": "series_too_short"}, ensure_ascii=False, indent=2), encoding="utf-8")
        return [
            Artefact(artefact_id=aid, kind="table", name=f"univariate_grid_{var}", path=str(p), meta={"max_p": max_p, "max_q": max_q, "max_d": max_d}),
            Artefact(artefact_id=aid2, kind="metric", name=f"univariate_best_{var}", path=str(pm), meta={"best": best}),
        ]

    for d in range(max_d + 1):
        for p_ in range(max_p + 1):
            for q_ in range(max_q + 1):
                if p_ == 0 and d == 0 and q_ == 0:
                    continue
                order = (p_, d, q_)
                try:
                    m = ARIMA(y.values, order=order, trend="c").fit()
                    aic = float(m.aic)
                    bic = float(m.bic)
                    rows.append({"p": p_, "d": d, "q": q_, "aic": aic, "bic": bic, "nobs": int(m.nobs)})

                    if aic < best["aic"]:
                        best = {"aic": aic, "bic": bic, "order": order}
                except Exception as e:
                    rows.append({"p": p_, "d": d, "q": q_, "error": str(e)})

    tab = pd.DataFrame(rows)
    if "aic" in tab.columns:
        tab = tab.sort_values(["aic"], na_position="last").reset_index(drop=True)

    aid = _next_id(ctx, "table")
    p = ctx.run_dirs.tables_dir / f"{aid}_univariate_grid_{var}.csv"
    save_table_csv(tab, p)

    aid2 = _next_id(ctx, "metric")
    pm = ctx.run_dirs.metrics_dir / f"{aid2}_univariate_best_{var}.json"
    pm.write_text(json.dumps({"best": best, "var": var, "nobs": int(len(y))}, ensure_ascii=False, indent=2), encoding="utf-8")

    return [
        Artefact(artefact_id=aid, kind="table", name=f"univariate_grid_{var}", path=str(p), meta={"max_p": max_p, "max_q": max_q, "max_d": max_d}),
        Artefact(artefact_id=aid2, kind="metric", name=f"univariate_best_{var}", path=str(pm), meta={"best": best}),
    ]
