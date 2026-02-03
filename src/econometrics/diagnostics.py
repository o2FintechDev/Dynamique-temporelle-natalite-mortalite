from __future__ import annotations

import json
import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm

from src.agent.schemas import Artefact
from src.agent.tools import ToolContext, _next_id
from src.visualization.charts import save_timeseries_png, FigureSpec
from src.visualization.tables import save_table_csv


# -----------------------------
# Phillips–Perron (implémentation défensive)
# Principe: test de racine unitaire via régression à la Dickey-Fuller
# avec estimation HAC (Newey–West) de la variance => p-value approx (asymptotique)
# Note produit: la p-value est une approximation, mais stable et traçable.
# -----------------------------
def _pp_test_statsmodels_only(y: pd.Series, trend: str = "c", lags: int | None = None) -> dict:
    """
    Retourne dict: stat, pvalue_approx, usedlag, nobs, spec.
    - trend: 'n' (none), 'c' (const), 'ct' (const+trend)
    - lags: si None, règle de Schwert (defensive)
    """
    y = y.dropna().astype(float)
    n = len(y)
    if n < 30:
        raise ValueError("PP: série trop courte (<30).")

    dy = y.diff().dropna()
    y_lag = y.shift(1).dropna()
    # align
    df = pd.concat([dy, y_lag], axis=1).dropna()
    df.columns = ["dy", "y_lag"]

    # Schwert (1989) rule-of-thumb
    if lags is None:
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

    model = sm.OLS(df["dy"], X)
    # HAC / Newey-West covariance
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    # Stat t sur coefficient de y_{t-1}
    # Sous H0 racine unitaire: distribution non standard (type DF).
    # On fournit une p-value approx normale (produit) + on expose la stat.
    # La décision peut utiliser la stat et/ou s'appuyer sur ADF en priorité.
    t_stat = float(res.tvalues["y_lag"] if "y_lag" in res.tvalues else res.tvalues.iloc[-1])

    # Approx p-value normale (proxy) -> traçable, pas "économétriquement parfaite"
    from scipy import stats
    pval_approx = float(2 * (1 - stats.norm.cdf(abs(t_stat))))

    return {
        "stat": t_stat,
        "pvalue_approx": pval_approx,
        "usedlag": int(lags),
        "nobs": int(res.nobs),
        "spec": trend,
        "note": "PP approx: t-stat HAC(Newey-West), p-value approx normale (proxy produit).",
    }


def acf_pacf_artefacts(ctx: ToolContext, var: str, lags: int = 48) -> list[Artefact]:
    df: pd.DataFrame = ctx.memory["df_ms"]
    s = df[var].dropna().astype(float)

    a_acf = acf(s, nlags=lags, fft=True)
    a_pacf = pacf(s, nlags=lags, method="ywm")

    tab = pd.DataFrame({"lag": np.arange(len(a_acf)), "acf": a_acf, "pacf": a_pacf})
    aid_t = _next_id(ctx, "table")
    p_csv = ctx.run_dirs.tables_dir / f"{aid_t}_acf_pacf_{var}.csv"
    save_table_csv(tab, p_csv)

    aid_f1 = _next_id(ctx, "fig")
    p1 = ctx.run_dirs.figures_dir / f"{aid_f1}_acf_{var}.png"
    save_timeseries_png(pd.Series(a_acf, index=np.arange(len(a_acf))), p1,
                        FigureSpec(title=f"ACF — {var}", xlabel="Lag", ylabel="ACF"))

    aid_f2 = _next_id(ctx, "fig")
    p2 = ctx.run_dirs.figures_dir / f"{aid_f2}_pacf_{var}.png"
    save_timeseries_png(pd.Series(a_pacf, index=np.arange(len(a_pacf))), p2,
                        FigureSpec(title=f"PACF — {var}", xlabel="Lag", ylabel="PACF"))

    return [
        Artefact(artefact_id=aid_t, kind="table", name=f"acf_pacf_{var}", path=str(p_csv), meta={"lags": lags}),
        Artefact(artefact_id=aid_f1, kind="figure", name=f"acf_{var}", path=str(p1), meta={"lags": lags}),
        Artefact(artefact_id=aid_f2, kind="figure", name=f"pacf_{var}", path=str(p2), meta={"lags": lags}),
    ]


def stationarity_tests_artefacts(ctx: ToolContext, var: str) -> list[Artefact]:
    df: pd.DataFrame = ctx.memory["df_ms"]
    s = df[var].dropna().astype(float)

    rows: list[dict] = []

    # ADF 3 specs
    for reg in ["c", "ct", "n"]:
        try:
            stat, pval, usedlag, nobs, crit, _ = adfuller(s, regression=reg, autolag="AIC")
            rows.append({
                "test": "ADF",
                "spec": reg,
                "stat": float(stat),
                "pvalue": float(pval),
                "usedlag": int(usedlag),
                "nobs": int(nobs),
            })
        except Exception as e:
            rows.append({"test": "ADF", "spec": reg, "error": str(e)})

    # PP (statsmodels-only)
    for reg in ["c", "ct", "n"]:
        try:
            pp = _pp_test_statsmodels_only(s, trend=reg, lags=None)
            rows.append({
                "test": "PP",
                "spec": reg,
                "stat": float(pp["stat"]),
                "pvalue": float(pp["pvalue_approx"]),
                "usedlag": int(pp["usedlag"]),
                "nobs": int(pp["nobs"]),
                "note": pp["note"],
            })
        except Exception as e:
            rows.append({"test": "PP", "spec": reg, "error": str(e)})

    # “bande DF” fallback défensif = Ljung-Box sur diff
    try:
        lb = acorr_ljungbox(s.diff().dropna(), lags=[12], return_df=True)
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
    # Décision produit: on s'appuie sur ADF(c) (standard) + traçabilité.
    df: pd.DataFrame = ctx.memory["df_ms"]
    s = df[var].dropna().astype(float)

    payload = {"rule": "ADF(c) p<0.05 => TS else DS", "adf_c_pvalue": None, "decision": None}
    decision = "DS"

    try:
        stat, pval, usedlag, nobs, crit, _ = adfuller(s, regression="c", autolag="AIC")
        payload["adf_c_pvalue"] = float(pval)
        decision = "TS" if float(pval) < 0.05 else "DS"
    except Exception as e:
        payload["error"] = str(e)
        decision = "DS (fallback)"

    payload["decision"] = decision

    aid = _next_id(ctx, "metric")
    p = ctx.run_dirs.metrics_dir / f"{aid}_ts_ds_{var}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return decision, Artefact(artefact_id=aid, kind="metric", name=f"ts_ds_decision_{var}", path=str(p), meta=payload)
