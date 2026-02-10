# src/econometrics/univariate.py
from __future__ import annotations
from typing import Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

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

def _sig_stats_from_res(res: Any, *, alpha: float = 0.05) -> dict[str, Any]:
    if res is None:
        return {"is_significant": False}

    try:
        pvals = res.pvalues.dropna()
    except Exception:
        return {"is_significant": False}

    keep = [k for k in pvals.index.astype(str) if k.startswith(("ar.L", "ma.L"))]
    if not keep:
        return {"is_significant": False}


    tested = pvals.loc[keep]

    return {
        "is_significant": bool((tested < alpha).all()),
        "max_pvalue": float(tested.max()),
        "n_params_tested": int(len(tested)),
        "n_sig": int((tested < alpha).sum()),
    }

def _arma_grid_core(y, orders, *, alpha_sig=0.05):
    rows = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", message=".*failed to converge.*")

        for p, d, q in orders:
            try:
                res = SARIMAX(
                    y,
                    order=(p, d, q),
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)

                if not bool(res.mle_retvals.get("converged", False)):
                    continue

                rows.append({
                    "p": int(p),
                    "q": int(q),
                    "aic": float(res.aic),
                    "bic": float(res.bic),
                    **_sig_stats_from_res(res, alpha=alpha_sig),
                })
            except Exception:
                continue

    return pd.DataFrame(rows).sort_values(["aic", "bic"]).reset_index(drop=True)


def ar_grid(series: pd.Series, p_max: int = 6, *, alpha_sig: float = 0.05) -> pd.DataFrame:
    y = series.dropna().astype(float)
    return _arma_grid_core(y, [(p, 0, 0) for p in range(1, p_max + 1)], alpha_sig=alpha_sig)

def ma_grid(series: pd.Series, q_max: int = 6, *, alpha_sig: float = 0.05) -> pd.DataFrame:
    y = series.dropna().astype(float)
    return _arma_grid_core(y, [(0, 0, q) for q in range(1, q_max + 1)], alpha_sig=alpha_sig)

def arma_grid(series: pd.Series, p_max: int = 6, q_max: int = 6, *, alpha_sig: float = 0.05) -> pd.DataFrame:
    y = series.dropna().astype(float)
    return _arma_grid_core(
        y,
        [(p, 0, q) for p in range(1, p_max + 1) for q in range(1, q_max + 1)],
        alpha_sig=alpha_sig,
    )




def ar_grid(series: pd.Series, p_max: int = 6, *, alpha_sig: float = 0.05) -> pd.DataFrame:
    y = series.dropna().astype(float)
    rows = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", message=".*failed to converge.*")

        for p in range(1, p_max + 1):
            try:
                m = SARIMAX(
                    y,
                    order=(p, 0, 0),
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = m.fit(disp=False)

                if not bool(res.mle_retvals.get("converged", False)):
                    continue

                sig = _sig_stats_from_res(res, alpha=alpha_sig)
                rows.append({
                    "p": int(p),
                    "aic": float(res.aic),
                    "bic": float(res.bic),
                    **sig,
                })
            except Exception:
                continue
    return pd.DataFrame(rows).sort_values(["aic", "bic"]).reset_index(drop=True)

def ma_grid(series: pd.Series, q_max: int = 6, *, alpha_sig: float = 0.05) -> pd.DataFrame:
    y = series.dropna().astype(float)
    rows = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", message=".*failed to converge.*")

        for q in range(1, q_max + 1):
            try:
                m = SARIMAX(
                    y,
                    order=(0, 0, q),
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = m.fit(disp=False)

                if not bool(res.mle_retvals.get("converged", False)):
                    continue

                sig = _sig_stats_from_res(res, alpha=alpha_sig)
                rows.append({
                    "q": int(q),
                    "aic": float(res.aic),
                    "bic": float(res.bic),
                    **sig,
                })
            except Exception:
                continue
    return pd.DataFrame(rows).sort_values(["aic", "bic"]).reset_index(drop=True)


def arma_grid(series: pd.Series, p_max: int = 6, q_max: int = 6, *, alpha_sig: float = 0.05) -> pd.DataFrame:
    y = series.dropna().astype(float)
    rows = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", message=".*failed to converge.*")

        for p in range(1, p_max + 1):
            for q in range(1, q_max + 1):
                try:
                    m = SARIMAX(
                        y,
                        order=(p, 0, q),
                        trend="c",
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    res = m.fit(disp=False)

                    if not bool(res.mle_retvals.get("converged", False)):
                        continue

                    sig = _sig_stats_from_res(res, alpha=alpha_sig)
                    rows.append({
                        "p": int(p),
                        "q": int(q),
                        "aic": float(res.aic),
                        "bic": float(res.bic),
                        **sig,
                    })
                except Exception:
                    continue


    return pd.DataFrame(rows).sort_values(["aic", "bic"]).reset_index(drop=True)

def arima_grid(
    series: pd.Series,
    *,
    p_max: int = 6,
    d_max: int = 2,
    q_max: int = 6,
    d_force: Optional[int] = None,
    trend: Optional[str] = None,
    alpha_sig: float = 0.05,
):
    y = series.dropna().astype(float)

    rows, best_res, best = [], None, None
    best_aic = np.inf

    trend_use = trend if trend is not None else ("n" if d_force == 1 else "c")
    d_vals = [d_force] if d_force is not None else range(d_max + 1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        for p in range(p_max + 1):
            for d in d_vals:
                for q in range(q_max + 1):
                    if (p, d, q) == (0, 0, 0):
                        continue
                    try:
                        res = SARIMAX(
                            y,
                            order=(p, d, q),
                            trend=trend_use,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit(disp=False)

                        if not res.mle_retvals.get("converged", False):
                            continue

                        row = {
                            "p": p, "d": d, "q": q,
                            "aic": float(res.aic),
                            "bic": float(res.bic),
                            **_sig_stats_from_res(res, alpha=alpha_sig),
                        }
                        rows.append(row)

                        if res.aic < best_aic:
                            best_aic, best, best_res = res.aic, row, res
                    except Exception:
                        continue

    grid = pd.DataFrame(rows).sort_values(["aic", "bic"]).reset_index(drop=True)
    return grid, best, best_res


def fit_sarimax_safe(
    series: pd.Series,
    order: tuple[int, int, int],
    *,
    trend: str = "c",
) -> Any | None:
    """Fit SARIMAX et retourne res (ou None si échec/non convergence)."""
    y = series.dropna().astype(float)
    if len(y) < 30:
        return None
    try:
        m = SARIMAX(
            y,
            order=order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = m.fit(disp=False)
        if not bool(res.mle_retvals.get("converged", False)):
            return None
        return res
    except Exception:
        return None


def residual_diagnostics(resid: np.ndarray, lags: int = 24) -> pd.DataFrame:
    resid = resid[~np.isnan(resid)]
    lb = acorr_ljungbox(resid, lags=[lags], return_df=True)

    jb = stats.jarque_bera(resid)
    jb_stat = float(getattr(jb, "statistic", jb[0]))
    jb_p = float(getattr(jb, "pvalue", jb[1]))

    arch_stat, arch_p, _, _ = het_arch(resid, nlags=min(lags, 12))

    out = pd.DataFrame([{
        "ljungbox_stat": float(lb["lb_stat"].iloc[0]),
        "ljungbox_p": round(float(lb["lb_pvalue"].iloc[0]), 2),
        "jarque_bera_stat": jb_stat,
        "jarque_bera_p": jb_p,
        "arch_stat": float(arch_stat),
        "arch_p": float(arch_p),
    }]).set_index(pd.Index(["diag"]))
    return out

def figs_fit(series: pd.Series, best_res) -> dict[str, plt.Figure]:
    if best_res is None:
        return {}
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
