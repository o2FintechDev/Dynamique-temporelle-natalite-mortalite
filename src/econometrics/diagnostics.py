from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

def acf_pacf_figs(series: pd.Series, lags: int = 24) -> tuple[plt.Figure, plt.Figure, pd.DataFrame]:
    x = series.dropna().astype(float)
    acf_vals, acf_conf = acf(x, nlags=lags, alpha=0.05)
    pacf_vals, pacf_conf = pacf(x, nlags=lags, alpha=0.05, method="ywm")

    df = pd.DataFrame({
        "lag": range(len(acf_vals)),
        "acf": acf_vals,
        "acf_low": acf_conf[:, 0],
        "acf_high": acf_conf[:, 1],
        "pacf": pacf_vals,
        "pacf_low": pacf_conf[:, 0],
        "pacf_high": pacf_conf[:, 1],
    }).set_index("lag")

    fig1 = plt.figure()
    plt.stem(df.index, df["acf"], basefmt=" ")
    plt.title("ACF (niveau)")
    plt.axhline(0)

    fig2 = plt.figure()
    plt.stem(df.index, df["pacf"], basefmt=" ")
    plt.title("PACF (niveau)")
    plt.axhline(0)

    return fig1, fig2, df

def adf_table(series: pd.Series) -> pd.DataFrame:
    x = series.dropna().astype(float)
    rows = []
    for reg in ["n", "c", "ct"]:
        stat, pval, usedlag, nobs, crit, _ = adfuller(x, regression=reg, autolag="AIC")
        rows.append({
            "spec": reg,
            "adf_stat": float(stat),
            "pvalue": float(pval),
            "usedlag": int(usedlag),
            "nobs": int(nobs),
            "crit_1": float(crit["1%"]),
            "crit_5": float(crit["5%"]),
            "crit_10": float(crit["10%"]),
        })
    return pd.DataFrame(rows).set_index("spec")


def dickey_fuller_band_metrics(acf_df: pd.DataFrame) -> pd.DataFrame:
    # lecture “bande”: proportion de lags hors CI 95% et run length
    x = acf_df.loc[1:, ["acf", "acf_low", "acf_high"]].copy()
    outside = (x["acf"] < x["acf_low"]) | (x["acf"] > x["acf_high"])
    run = 0
    max_run = 0
    for v in outside.values:
        run = run + 1 if v else 0
        max_run = max(max_run, run)
    return pd.DataFrame([{
        "acf_outside_ratio": float(outside.mean()),
        "acf_max_consecutive_outside": int(max_run),
        "acf_abs_area_1_24": float(np.abs(x["acf"]).sum()),
    }]).set_index(pd.Index(["band"]))

def ts_vs_ds_decision(adf: pd.DataFrame,  band: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    # règle simple, traçable:
    # - si ADF (c ou ct) rejette (p<0.05) => TS
    # - sinon DS
    p_c = float(adf.loc["c", "pvalue"])
    p_ct = float(adf.loc["ct", "pvalue"])

    vote_ts = (p_c < 0.05) or (p_ct < 0.05) 
    verdict = "TS" if vote_ts else "DS"

    tbl = pd.DataFrame([{
        "adf_p_c": p_c,
        "adf_p_ct": p_ct,
        "band_outside_ratio": float(band.loc["band", "acf_outside_ratio"]),
        "verdict": verdict,
        "rule": "TS si (ADF c/ct p<0.05), sinon DS",
    }]).set_index(pd.Index(["decision"]))

    metric = {"verdict": verdict, "adf_p_c": p_c, "adf_p_ct": p_ct}
    return tbl, metric

def ljungbox_diff(series: pd.Series, lags: int = 24) -> pd.DataFrame:
    x = series.dropna().astype(float)
    dx = x.diff().dropna()
    lb = acorr_ljungbox(dx, lags=[lags], return_df=True)
    lb.index = ["diff"]
    return lb
