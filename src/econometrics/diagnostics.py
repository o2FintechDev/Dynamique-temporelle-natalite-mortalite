# econometrics/diagnostics.py

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox


def _prep_series(df: pd.DataFrame, y: str) -> pd.Series:
    if "Date" in df.columns:
        d = df[["Date", y]].copy()
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d = d.dropna(subset=["Date"]).sort_values("Date")
        s = pd.to_numeric(d[y], errors="coerce")
    else:
        s = pd.to_numeric(df[y], errors="coerce")
    return s.dropna().astype(float)


def run_diagnostics_pack(df: pd.DataFrame, *, y: str, lags: int = 24) -> dict[str, Any]:
    """
    Sorties:
      tables: acf/pacf, adf(3 specs), ljungbox(diff)
      figures: acf, pacf
      metrics: meta diagnostics
    """
    s = _prep_series(df, y)
    n = int(len(s))

    metrics: dict[str, Any] = {
        "diagnostics_meta": {"y": y, "n_obs": n, "lags": int(lags)}
    }

    if n < 30:
        tab = pd.DataFrame([{"error": f"n_obs trop faible pour diagnostics (n={n})"}])
        return {
            "tables": {f"diagnostics_error_{y}": tab},
            "metrics": metrics,
        }

    # ACF/PACF
    a_acf = acf(s.values, nlags=lags, fft=True)
    a_pacf = pacf(s.values, nlags=lags, method="ywm")
    tab_acf = pd.DataFrame({"lag": np.arange(0, lags + 1), "acf": a_acf, "pacf": a_pacf})

    fig_acf = plt.figure()
    plt.stem(tab_acf["lag"], tab_acf["acf"])
    plt.title(f"ACF — {y}")
    plt.xlabel("lag")
    plt.ylabel("acf")

    fig_pacf = plt.figure()
    plt.stem(tab_acf["lag"], tab_acf["pacf"])
    plt.title(f"PACF — {y}")
    plt.xlabel("lag")
    plt.ylabel("pacf")

    # ADF 3 specs
    def _adf(reg: str) -> dict[str, Any]:
        stat, pval, usedlag, nobs, crit, _ = adfuller(s.values, regression=reg, autolag="AIC")
        return {
            "test": "ADF",
            "spec": reg,
            "stat": float(stat),
            "pvalue": float(pval),
            "usedlag": int(usedlag),
            "nobs": int(nobs),
            "crit_1": float(crit["1%"]),
            "crit_5": float(crit["5%"]),
            "crit_10": float(crit["10%"]),
        }

    rows_adf = []
    for reg in ["n", "c", "ct"]:
        try:
            rows_adf.append(_adf(reg))
        except Exception as e:
            rows_adf.append({"test": "ADF", "spec": reg, "error": str(e)})
    tab_adf = pd.DataFrame(rows_adf)

    # Ljung-Box sur diff (proxy autocorr résiduelle en “bande”)
    rows_lb = []
    try:
        lb = acorr_ljungbox(s.diff().dropna().values, lags=[12], return_df=True)
        rows_lb.append({
            "test": "LjungBox",
            "spec": "diff",
            "lags": 12,
            "lb_stat": float(lb["lb_stat"].iloc[0]),
            "lb_pvalue": float(lb["lb_pvalue"].iloc[0]),
        })
    except Exception as e:
        rows_lb.append({"test": "LjungBox", "spec": "diff", "error": str(e)})
    tab_lb = pd.DataFrame(rows_lb)

    return {
        "tables": {
            f"acf_pacf_{y}": tab_acf,
            f"adf_{y}": tab_adf,
            f"ljungbox_diff_{y}": tab_lb,
        },
        "figures": {
            f"acf_{y}": fig_acf,
            f"pacf_{y}": fig_pacf,
        },
        "metrics": metrics,
    }
