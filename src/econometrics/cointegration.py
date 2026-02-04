# economics/cointegration.py

from __future__ import annotations

from typing import Any

import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


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


def run_cointegration_pack(df: pd.DataFrame, *, vars: list[str]) -> dict[str, Any]:
    X = _prep_df(df, vars)
    nobs = int(X.shape[0])
    k = int(X.shape[1])

    if k < 2 or nobs < 50:
        tab = pd.DataFrame([{"error": f"Cointegration: nécessite >=2 variables et nobs>=50 (k={k}, nobs={nobs})"}])
        return {"tables": {"engle_granger": tab, "johansen_trace": tab.copy()}, "metrics": {"cointegration_meta": {"vars": vars, "nobs": nobs}}}

    # Engle-Granger pairwise
    rows = []
    cols = list(X.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            try:
                stat, pval, crit = coint(X[a].values, X[b].values)
                rows.append({
                    "x": a,
                    "y": b,
                    "stat": float(stat),
                    "pvalue": float(pval),
                    "nobs": nobs,
                    "crit_1": float(crit[0]),
                    "crit_5": float(crit[1]),
                    "crit_10": float(crit[2]),
                })
            except Exception as e:
                rows.append({"x": a, "y": b, "error": str(e)})

    eg = pd.DataFrame(rows).sort_values(["pvalue"], na_position="last")

    # Johansen (trace)
    try:
        joh = coint_johansen(X.values, det_order=0, k_ar_diff=1)
        joh_tab = pd.DataFrame({
            "rank": list(range(len(joh.lr1))),
            "trace_stat": joh.lr1,
            "crit_90": joh.cvt[:, 0],
            "crit_95": joh.cvt[:, 1],
            "crit_99": joh.cvt[:, 2],
        })
    except Exception as e:
        joh_tab = pd.DataFrame([{"error": str(e)}])

    return {
        "tables": {"engle_granger": eg, "johansen_trace": joh_tab},
        "metrics": {"cointegration_meta": {"vars": cols, "nobs": nobs}},
    }
