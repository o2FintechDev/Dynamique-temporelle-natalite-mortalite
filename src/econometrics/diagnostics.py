# econometrics/diagnostics.py

from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm

from src.agent.schemas import Artefact
from src.agent.tools import ToolContext, _next_id
from src.visualization.charts import save_timeseries_png, FigureSpec
from src.visualization.tables import save_table_csv


def _norm_cdf(x: float) -> float:
    # CDF normale standard sans SciPy
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _pp_test_statsmodels_only(y: pd.Series, trend: str = "c", lags: int | None = None) -> dict:
    """
    Phillips–Perron (défensif, offline):
    - t-stat HAC(Newey-West) sur y_{t-1} dans régression Δy_t ~ y_{t-1} (+ c + trend)
    - p-value approx normale (proxy produit)
    """
    y = y.dropna().astype(float)
    n = len(y)
    if n < 30:
        raise ValueError("PP: série trop courte (<30).")

    dy = y.diff()
    y_lag = y.shift(1)
    df = pd.concat([dy, y_lag], axis=1).dropna()
    df.columns = ["dy", "y_lag"]

    if lags is None:
        # Schwert (1989) rule-of-thumb (cap défensif)
        lags = int(np.floor(12 * (n / 100) ** 0.25))
        lags = max(0, min(lags, 20))

    X = df[["y_lag"]].copy()
    if trend == "c":
        X = sm.add_constant(X, has_constant="add")
    elif trend == "ct":
        X = sm.add_constant(X, has_constant="add")
        X["trend"] = np.arange(len(X), dtype=float)
    elif trend == "n":
        pass
    else:
        raise ValueError("trend doit être dans {'n','c','ct'}")

    res = sm.OLS(df["dy"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    # t-stat sur y_lag
    if "y_lag" in res.tvalues:
        t_stat = float(res.tvalues["y_lag"])
    else:
        t_stat = float(res.tvalues.iloc[-1])

    # p-value approx normale (proxy produit)
    pval_approx = float(2.0 * (1.0 - _norm_cdf(abs(t_stat))))

    return {
        "stat": t_stat,
        "pvalue_approx": pval_approx,
        "usedlag": int(lags),
        "nobs": int(res.nobs),
        "spec": trend,
        "note": "PP proxy: t-stat HAC(Newey-West), p-value approx normale (non DF exacte).",
    }


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


def acf_pacf_artefacts(ctx: ToolContext, var: str, lags: int = 48) -> list[Artefact]:
    s = _get_series(ctx, var)
    if len(s) < 20:
        tab = pd.DataFrame([{"error": f"ACF/PACF: série trop courte (n={len(s)})"}])
        aid_t = _next_id(ctx, "table")
        p_csv = ctx.run_dirs.tables_dir / f"{aid_t}_acf_pacf_{var}.csv"
        save_table_csv(tab, p_csv)
        return [Artefact(artefact_id=aid_t, kind="table", name=f"acf_pacf_{var}", path=str(p_csv), meta={"lags": lags})]

    a_acf = acf(s.values, nlags=lags, fft=True)
    a_pacf = pacf(s.values, nlags=lags, method="ywm")

    tab = pd.DataFrame({"lag": np.arange(0, lags + 1), "acf": a_acf, "pacf": a_pacf})

    aid_t = _next_id(ctx, "table")
    p_csv = ctx.run_dirs.tables_dir / f"{aid_t}_acf_pacf_{var}.csv"
    save_table_csv(tab, p_csv)

    aid_f1 = _next_id(ctx, "fig")
    p1 = ctx.run_dirs.figures_dir / f"{aid_f1}_acf_{var}.png"
    save_timeseries_png(
        pd.Series(a_acf, index=np.arange(0, lags + 1)),
        p1,
        FigureSpec(title=f"ACF — {var}", xlabel="Lag", ylabel="ACF"),
    )

    aid_f2 = _next_id(ctx, "fig")
    p2 = ctx.run_dirs.figures_dir / f"{aid_f2}_pacf_{var}.png"
    save_timeseries_png(
        pd.Series(a_pacf, index=np.arange(0, lags + 1)),
        p2,
        FigureSpec(title=f"PACF — {var}", xlabel="Lag", ylabel="PACF"),
    )

    return [
        Artefact(artefact_id=aid_t, kind="table", name=f"acf_pacf_{var}", path=str(p_csv), meta={"lags": lags}),
        Artefact(artefact_id=aid_f1, kind="figure", name=f"acf_{var}", path=str(p1), meta={"lags": lags}),
        Artefact(artefact_id=aid_f2, kind="figure", name=f"pacf_{var}", path=str(p2), meta={"lags": lags}),
    ]


def stationarity_tests_artefacts(ctx: ToolContext, var: str) -> list[Artefact]:
    s = _get_series(ctx, var)

    rows: list[dict] = []
    if len(s) < 30:
        rows.append({"error": f"Stationarity: série trop courte (n={len(s)})"})
    else:
        # ADF 3 specs
        for reg in ["c", "ct", "n"]:
            try:
                stat, pval, usedlag, nobs, crit, _ = adfuller(s.values, regression=reg, autolag="AIC")
                rows.append({
                    "test": "ADF",
                    "spec": reg,
                    "stat": float(stat),
                    "pvalue": float(pval),
                    "usedlag": int(usedlag),
                    "nobs": int(nobs),
                    "crit_1": float(crit["1%"]),
                    "crit_5": float(crit["5%"]),
                    "crit_10": float(crit["10%"]),
                })
            except Exception as e:
                rows.append({"test": "ADF", "spec": reg, "error": str(e)})

        # PP proxy
        for reg in ["c", "ct", "n"]:
            try:
                pp = _pp_test_statsmodels_only(s, trend=reg, lags=None)
                rows.append({
                    "test": "PP_proxy",
                    "spec": reg,
                    "stat": float(pp["stat"]),
                    "pvalue": float(pp["pvalue_approx"]),
                    "usedlag": int(pp["usedlag"]),
                    "nobs": int(pp["nobs"]),
                    "note": pp["note"],
                })
            except Exception as e:
                rows.append({"test": "PP_proxy", "spec": reg, "error": str(e)})

        # Proxy “bande DF”: Ljung-Box sur diff
        try:
            lb = acorr_ljungbox(s.diff().dropna().values, lags=[12], return_df=True)
            rows.append({
                "test": "DF_band_proxy",
                "spec": "LjungBox(diff,12)",
                "stat": float(lb["lb_stat"].iloc[0]),
                "pvalue": float(lb["lb_pvalue"].iloc[0]),
            })
        except Exception as e:
            rows.append({"test": "DF_band_proxy", "spec": "LjungBox(diff,12)", "error": str(e)})

    tab = pd.DataFrame(rows)

    aid = _next_id(ctx, "table")
    p = ctx.run_dirs.tables_dir / f"{aid}_stationarity_{var}.csv"
    save_table_csv(tab, p)

    return [Artefact(artefact_id=aid, kind="table", name=f"stationarity_tests_{var}", path=str(p), meta={})]


def decide_ts_ds(ctx: ToolContext, var: str) -> tuple[str, Artefact]:
    """
    Décision produit: ADF(c) p<0.05 => TS else DS
    """
    s = _get_series(ctx, var)
    payload = {"rule": "ADF(c) p<0.05 => TS else DS", "adf_c_pvalue": None, "decision": None}
    decision = "DS"

    try:
        if len(s) < 30:
            raise ValueError(f"ADF(c): série trop courte (n={len(s)})")
        stat, pval, usedlag, nobs, crit, _ = adfuller(s.values, regression="c", autolag="AIC")
        payload["adf_c_pvalue"] = float(pval)
        payload["usedlag"] = int(usedlag)
        payload["nobs"] = int(nobs)
        decision = "TS" if float(pval) < 0.05 else "DS"
    except Exception as e:
        payload["error"] = str(e)
        decision = "DS (fallback)"

    payload["decision"] = decision

    aid = _next_id(ctx, "metric")
    p = ctx.run_dirs.metrics_dir / f"{aid}_ts_ds_{var}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return decision, Artefact(artefact_id=aid, kind="metric", name=f"ts_ds_decision_{var}", path=str(p), meta=payload)