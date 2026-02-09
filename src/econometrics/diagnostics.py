# src/econometrics/diagnostics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox


# ============================================================
# ACF / PACF
# ============================================================
def acf_pacf_figs(series: pd.Series, lags: int = 24) -> tuple[plt.Figure, plt.Figure, pd.DataFrame]:
    x = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    acf_vals, acf_conf = acf(x, nlags=lags, alpha=0.05)
    pacf_vals, pacf_conf = pacf(x, nlags=lags, alpha=0.05, method="ywm")

    df = (
        pd.DataFrame(
            {
                "lag": range(len(acf_vals)),
                "acf": acf_vals,
                "acf_low": acf_conf[:, 0],
                "acf_high": acf_conf[:, 1],
                "pacf": pacf_vals,
                "pacf_low": pacf_conf[:, 0],
                "pacf_high": pacf_conf[:, 1],
            }
        )
        .set_index("lag")
        .copy()
    )

    fig1 = plt.figure()
    plt.stem(df.index, df["acf"], basefmt=" ")
    plt.title("ACF (niveau)")
    plt.axhline(0)

    fig2 = plt.figure()
    plt.stem(df.index, df["pacf"], basefmt=" ")
    plt.title("PACF (niveau)")
    plt.axhline(0)

    return fig1, fig2, df


# ============================================================
# ADF / TS vs DS (existant)
# ============================================================
def adf_table(series: pd.Series) -> pd.DataFrame:
    x = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    rows = []
    for reg in ["n", "c", "ct"]:
        stat, pval, usedlag, nobs, crit, _ = adfuller(x, regression=reg, autolag="AIC")
        rows.append(
            {
                "spec": reg,
                "adf_stat": float(stat),
                "pvalue": float(pval),
                "usedlag": int(usedlag),
                "nobs": int(nobs),
                "crit_1": float(crit["1%"]),
                "crit_5": float(crit["5%"]),
                "crit_10": float(crit["10%"]),
            }
        )
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
    return pd.DataFrame(
        [
            {
                "acf_outside_ratio": float(outside.mean()),
                "acf_max_consecutive_outside": int(max_run),
                "acf_abs_area_1_24": float(np.abs(x["acf"]).sum()),
            }
        ]
    ).set_index(pd.Index(["band"]))


def ts_vs_ds_decision(
    tbl_adf: pd.DataFrame, tbl_band: pd.DataFrame, alpha: float = 0.05
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    """
    Décision TS vs DS fondée EXCLUSIVEMENT sur ADF.
    Règle:
      - si ADF(ct) rejette H0 (p<alpha) => TS (stationnaire autour d'une tendance)
      - sinon si ADF(c) rejette H0 => TS (stationnaire autour d'une constante)
      - sinon => DS
    tbl_band est conservé pour traçabilité (lecture persistance), mais n'entre pas dans la décision.
    """

    def _p(spec: str) -> float | None:
        try:
            return float(tbl_adf.loc[spec, "pvalue"])
        except Exception:
            return None

    p_c = _p("c")
    p_ct = _p("ct")

    if p_ct is not None and p_ct < alpha:
        verdict = "TS"
        rule = "ADF(ct) rejette H0 => TS (tendance déterministe)."
    elif p_c is not None and p_c < alpha:
        verdict = "TS"
        rule = "ADF(c) rejette H0 => TS (constante)."
    else:
        verdict = "DS"
        rule = "ADF(c) et ADF(ct) ne rejettent pas => DS (différenciation requise)."

    tbl_dec = pd.DataFrame(
        [
            {
                "verdict": verdict,
                "adf_p_c": p_c,
                "adf_p_ct": p_ct,
                "alpha": alpha,
                "rule": rule,
            }
        ]
    ).set_index(pd.Index(["ts_vs_ds"]))

    metrics = {"verdict": verdict, "adf_p_c": p_c, "adf_p_ct": p_ct, "alpha": alpha, "rule": rule}
    return tbl_dec, metrics


def ljungbox_diff(series: pd.Series, lags: int = 24) -> pd.DataFrame:
    x = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    dx = x.diff().dropna()
    lb = acorr_ljungbox(dx, lags=[lags], return_df=True)
    lb.index = ["diff"]
    return lb


# ============================================================
# NEW: Stationnarité multi-variables + stationnarisation auto
# ============================================================
def _adf_one(
    s: pd.Series,
    *,
    regression: str = "c",
    autolag: str = "AIC",
    maxlag: Optional[int] = None,
) -> dict[str, Any]:
    x = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    if x.shape[0] < 15:
        return {"ok": False, "nobs": int(x.shape[0]), "stat": np.nan, "pvalue": np.nan, "reg": regression}
    try:
        stat, pval, usedlag, nobs, _, _ = adfuller(x, regression=regression, autolag=autolag, maxlag=maxlag)
        return {
            "ok": True,
            "nobs": int(nobs),
            "stat": float(stat),
            "pvalue": float(pval),
            "reg": regression,
            "usedlag": int(usedlag),
        }
    except Exception as e:
        return {
            "ok": False,
            "nobs": int(x.shape[0]),
            "stat": np.nan,
            "pvalue": np.nan,
            "reg": regression,
            "error": type(e).__name__,
        }


def _is_stationary(p: float, alpha: float) -> bool:
    return bool(np.isfinite(p) and p < alpha)


def _safe_log(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return np.log(x.where(x > 0))


@dataclass(frozen=True)
class StationarizeResult:
    series: pd.Series
    transform: str
    adf_p_level: Optional[float]
    adf_p_final: Optional[float]
    nobs_final: int


def stationarize_series(
    s: pd.Series,
    *,
    alpha: float = 0.05,
    seasonal_period: int = 12,
    max_d: int = 2,
    allow_log: bool = True,
) -> StationarizeResult:
    """
    Rend une série stationnaire via une stratégie déterministe.

    Ordre d'essai:
      1) level
      2) diff(d=1..max_d)
      3) seasdiff(D=1,s) ; seasdiff + diff(1)
      4) log(level) ; diff(1) sur log (si positif)

    Critère:
      - prend la première transformation qui passe ADF(c) (p < alpha)
      - sinon prend celle avec p-value minimale (fallback)
    """
    s0 = pd.to_numeric(s, errors="coerce")

    r0 = _adf_one(s0, regression="c")
    p0 = float(r0.get("pvalue", np.nan))
    best = {"series": s0, "transform": "level", "pvalue": p0}

    candidates: list[tuple[str, pd.Series]] = [("level", s0)]
    for d in range(1, int(max_d) + 1):
        candidates.append((f"diff(d={d})", s0.diff(d)))

    candidates.append((f"seasdiff(D=1,s={seasonal_period})", s0.diff(int(seasonal_period))))
    candidates.append((f"seasdiff(D=1,s={seasonal_period})+diff(d=1)", s0.diff(int(seasonal_period)).diff(1)))

    if allow_log:
        slog = _safe_log(s0)
        candidates.append(("log(level)", slog))
        candidates.append(("diff(d=1,on log)", slog.diff(1)))

    adf_p_level = None if not np.isfinite(p0) else float(p0)

    first_stationary: Optional[dict[str, Any]] = None
    best_p = float("inf")

    for tr, xs in candidates:
        rr = _adf_one(xs, regression="c")
        p = float(rr.get("pvalue", np.nan))

        if np.isfinite(p) and p < best_p:
            best_p = p
            best = {"series": xs, "transform": tr, "pvalue": p}

        if _is_stationary(p, alpha):
            first_stationary = {"series": xs, "transform": tr, "pvalue": p}
            break

    chosen = first_stationary or best
    out = pd.to_numeric(chosen["series"], errors="coerce").dropna()

    p_final = float(chosen["pvalue"])
    adf_p_final = None if not np.isfinite(p_final) else float(p_final)

    return StationarizeResult(
        series=out,
        transform=str(chosen["transform"]),
        adf_p_level=adf_p_level,
        adf_p_final=adf_p_final,
        nobs_final=int(out.shape[0]),
    )


def stationarity_and_stationarize_pack(
    df: pd.DataFrame,
    *,
    variables: list[str],
    alpha: float = 0.05,
    seasonal_period: int = 12,
    max_d: int = 2,
    allow_log: bool = True,
) -> dict[str, Any]:
    """
    Stationnarité variable par variable + stationnarisation auto.

    Sorties:
      - tables["tbl.diag.stationarity"] : ADF niveau + ADF final + transform
      - data["df.stationary"] : DataFrame final stationnaire (dropna conjoint)
      - metrics["m.diag.stationarity_meta"] : audit fenêtre / params
    """
    rows = []
    out_cols: dict[str, pd.Series] = {}

    for v in variables:
        if v not in df.columns:
            rows.append({"var": v, "present": False})
            continue

        r = stationarize_series(
            df[v],
            alpha=alpha,
            seasonal_period=seasonal_period,
            max_d=max_d,
            allow_log=allow_log,
        )

        rows.append(
            {
                "var": v,
                "present": True,
                "alpha": float(alpha),
                "adf_p_level": r.adf_p_level,
                "adf_p_final": r.adf_p_final,
                "transform": r.transform,
                "nobs_final": int(r.nobs_final),
            }
        )
        out_cols[v] = r.series

    tbl = pd.DataFrame(rows)

    if out_cols:
        Y = pd.concat(out_cols.values(), axis=1)
        Y.columns = list(out_cols.keys())
        Y = Y.dropna()
    else:
        Y = pd.DataFrame([])

    meta = {
        "vars": list(variables),
        "alpha": float(alpha),
        "seasonal_period": int(seasonal_period),
        "max_d": int(max_d),
        "allow_log": bool(allow_log),
        "nobs_used": int(Y.shape[0]),
        "start": str(Y.index.min()) if not Y.empty else None,
        "end": str(Y.index.max()) if not Y.empty else None,
    }

    return {
        "tables": {"tbl.diag.stationarity": tbl},
        "metrics": {"m.diag.stationarity_meta": meta},
        "data": {"df.stationary": Y},
    }
