from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# PP via statsmodels>=0.14 (si absent -> fallback explicite)
try:
    from statsmodels.tsa.stattools import phillips_perron  # returns (stat, pvalue, usedlag, nobs, crit)
except Exception:
    phillips_perron = None

from src.agent.schemas import Artefact
from src.agent.tools import ToolContext, _next_id
from src.visualization.charts import save_timeseries_png, FigureSpec
from src.visualization.tables import save_table_csv


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
    save_timeseries_png(
        pd.Series(a_acf, index=np.arange(len(a_acf))),
        p1,
        FigureSpec(title=f"ACF — {var}", xlabel="Lag", ylabel="ACF"),
    )

    aid_f2 = _next_id(ctx, "fig")
    p2 = ctx.run_dirs.figures_dir / f"{aid_f2}_pacf_{var}.png"
    save_timeseries_png(
        pd.Series(a_pacf, index=np.arange(len(a_pacf))),
        p2,
        FigureSpec(title=f"PACF — {var}", xlabel="Lag", ylabel="PACF"),
    )

    return [
        Artefact(artefact_id=aid_t, kind="table", name=f"acf_pacf_{var}", path=str(p_csv), meta={"lags": lags}),
        Artefact(artefact_id=aid_f1, kind="figure", name=f"acf_{var}", path=str(p1), meta={"lags": lags}),
        Artefact(artefact_id=aid_f2, kind="figure", name=f"pacf_{var}", path=str(p2), meta={"lags": lags}),
    ]


def stationarity_tests_artefacts(ctx: ToolContext, var: str) -> list[Artefact]:
    df: pd.DataFrame = ctx.memory["df_ms"]
    s = df[var].dropna().astype(float)

    rows: list[dict] = []

    # ADF 3 specs: constant, constant+trend, none
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

    # Phillips–Perron (statsmodels>=0.14)
    if phillips_perron is not None:
        try:
            stat, pval, usedlag, nobs, crit = phillips_perron(s)
            rows.append({
                "test": "PP",
                "spec": "default",
                "stat": float(stat),
                "pvalue": float(pval),
                "usedlag": int(usedlag),
                "nobs": int(nobs),
            })
        except Exception as e:
            rows.append({"test": "PP", "spec": "default", "error": str(e)})
    else:
        rows.append({"test": "PP", "spec": "default", "error": "statsmodels.tsa.stattools.phillips_perron indisponible"})

    # “Bande Dickey-Fuller” (implémentation défensive) :
    # proxy: Ljung-Box sur la série différenciée (diagnostic d’autocorrélation résiduelle)
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
    # Décision produit, traçable:
    # ADF(c) p<0.05 => TS, sinon DS. Fallback DS si erreur.
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
