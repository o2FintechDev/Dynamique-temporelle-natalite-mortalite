# src/econometrics/impulse.py
from __future__ import annotations

from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR


def _prep_df(df: pd.DataFrame, vars: list[str]) -> pd.DataFrame:
    cols = [v for v in vars if v in df.columns]
    if "Date" in df.columns:
        d = df[["Date"] + cols].copy()
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d = d.dropna(subset=["Date"]).sort_values("Date")
        X = d[cols].apply(pd.to_numeric, errors="coerce")
    else:
        X = df[cols].apply(pd.to_numeric, errors="coerce")
    return X.dropna()


def run_impulse_pack(df: pd.DataFrame, *, vars: list[str], maxlags: int = 12, horizon: int = 24) -> dict[str, Any]:
    X = _prep_df(df, vars)
    nobs = int(X.shape[0])
    k = int(X.shape[1])

    if k < 2 or nobs < 50:
        tab = pd.DataFrame([{"error": f"IRF/FEVD: nécessite >=2 variables et nobs>=50 (k={k}, nobs={nobs})"}])
        return {"tables": {"fevd": tab}, "metrics": {"impulse_meta": {"vars": vars, "nobs": nobs}}}

    model = VAR(X)
    sel = model.select_order(maxlags=maxlags)
    selected = int(sel.aic) if sel.aic is not None else 1
    res = model.fit(selected)

    # IRF figure (plot statsmodels)
    irf = res.irf(horizon)
    fig_irf = plt.figure()
    irf.plot(impulse=X.columns[0])
    plt.suptitle(f"IRF — choc: {X.columns[0]} (lag={selected})")

    # FEVD table : decomposition de la variance de la 1ère variable à l'horizon final
    fevd = res.fevd(horizon)
    last = fevd.decomp[-1, 0, :]  # (shocks,)
    fevd_tab = pd.DataFrame([{
        "target": X.columns[0],
        "horizon": horizon,
        **{f"shock_{col}": float(last[i]) for i, col in enumerate(X.columns)}
    }])

    return {
        "tables": {"fevd": fevd_tab},
        "figures": {"irf": fig_irf},
        "metrics": {"impulse_meta": {"vars": list(X.columns), "nobs": nobs, "selected_lag_aic": selected, "horizon": horizon}},
        "models": {"var_for_irf": res},
    }
