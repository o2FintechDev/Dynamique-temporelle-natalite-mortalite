# econometrics/univariate.py
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

def hurst_exponent(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 50:
        return float("nan")
    lags = range(2, 20)
    tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return float(poly[0] * 2.0)

def rescaled_range(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 50:
        return float("nan")
    y = x - np.mean(x)
    z = np.cumsum(y)
    r = np.max(z) - np.min(z)
    s = np.std(x)
    return float(r / s) if s > 0 else float("nan")

def arima_grid(series: pd.Series, p_max: int = 4, d_max: int = 2, q_max: int = 4) -> tuple[pd.DataFrame, dict, Any]:
    y = series.dropna().astype(float)

    best = None
    best_aic = np.inf
    best_res = None

    rows = []
    n_fit_error = 0
    n_nonconverged = 0

    # exploration: on supprime les ConvergenceWarning (ils ne sont pas informatifs en grille)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", message=".*failed to converge.*")

        for p in range(p_max + 1):
            for d in range(d_max + 1):
                for q in range(q_max + 1):
                    if p == 0 and d == 0 and q == 0:
                        continue
                    try:
                        m = SARIMAX(
                            y,
                            order=(p, d, q),
                            trend="c",
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )
                        res = m.fit(disp=False)

                        converged = bool(res.mle_retvals.get("converged", False))
                        if not converged:
                            n_nonconverged += 1
                            continue

                        aic = float(res.aic)
                        bic = float(res.bic)
                        rows.append({"p": p, "d": d, "q": q, "aic": aic, "bic": bic})

                        if aic < best_aic:
                            best_aic = aic
                            best_res = res
                            best = {"order": (p, d, q), "aic": aic, "bic": bic}

                    except Exception:
                        n_fit_error += 1
                        continue

    grid = pd.DataFrame(rows).sort_values(["aic", "bic"]).reset_index(drop=True)

    # trace minimale dans "best"
    if best is None:
        best = {"order": None, "aic": None, "bic": None, "note": "No converged model found."}

    best["grid_n_ok"] = int(len(grid))
    best["grid_n_nonconverged"] = int(n_nonconverged)
    best["grid_n_fit_error"] = int(n_fit_error)

    return grid, best, best_res

def residual_diagnostics(resid: np.ndarray, lags: int = 24) -> pd.DataFrame:
    resid = resid[~np.isnan(resid)]
    lb = acorr_ljungbox(resid, lags=[lags], return_df=True)

    jb = stats.jarque_bera(resid)
    jb_stat = float(getattr(jb, "statistic", jb[0]))
    jb_p = float(getattr(jb, "pvalue", jb[1]))

    arch_stat, arch_p, _, _ = het_arch(resid, nlags=min(lags, 12))

    out = pd.DataFrame([{
        "ljungbox_stat": float(lb["lb_stat"].iloc[0]),
        "ljungbox_p": float(lb["lb_pvalue"].iloc[0]),
        "jarque_bera_stat": jb_stat,
        "jarque_bera_p": jb_p,
        "arch_stat": float(arch_stat),
        "arch_p": float(arch_p),
    }]).set_index(pd.Index(["diag"]))
    return out

def figs_fit(series: pd.Series, best_res) -> dict[str, plt.Figure]:
    y = series.dropna().astype(float)
    pred = best_res.get_prediction()
    mu = pred.predicted_mean

    fig_fit = plt.figure()
    plt.plot(y.index, y.values, label="y")
    plt.plot(mu.index, mu.values, label="fit")
    plt.title("Fit ARIMA (in-sample)")
    plt.legend()

    resid = best_res.resid
    fig_acf = plt.figure()
    from statsmodels.tsa.stattools import acf
    a = acf(resid[~np.isnan(resid)], nlags=24)
    plt.stem(range(len(a)), a, basefmt=" ")
    plt.title("ACF résidus")
    plt.axhline(0)

    fig_qq = plt.figure()
    stats.probplot(resid[~np.isnan(resid)], dist="norm", plot=plt)
    plt.title("QQ-plot résidus")

    return {"fig.uni.fit": fig_fit, "fig.uni.resid_acf": fig_acf, "fig.uni.qq": fig_qq}
