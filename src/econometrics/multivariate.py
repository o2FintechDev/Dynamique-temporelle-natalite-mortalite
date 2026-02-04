# econometrics/multivariate.py
from __future__ import annotations

from typing import Any

import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests


def _prep_df(df: pd.DataFrame, vars: list[str]) -> pd.DataFrame:
    cols = [v for v in vars if v in df.columns]
    if "Date" in df.columns:
        d = df[["Date"] + cols].copy()
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d = d.dropna(subset=["Date"]).sort_values("Date")
        x = d[cols].apply(pd.to_numeric, errors="coerce")
    else:
        x = df[cols].apply(pd.to_numeric, errors="coerce")
    return x.dropna()


def run_var_pack(df: pd.DataFrame, *, vars: list[str], maxlags: int = 12) -> dict[str, Any]:
    """
    Produit:
      - table var_selection (AIC/BIC/HQIC/FPE + lag choisi AIC)
      - model var_model (VARResults picklable)
    """
    X = _prep_df(df, vars)
    nobs = int(X.shape[0])
    k = int(X.shape[1])

    if k < 2 or nobs < 50:
        tab = pd.DataFrame([{"error": f"VAR: nécessite >=2 variables et nobs>=50 (k={k}, nobs={nobs})"}])
        return {"tables": {"var_selection": tab}, "metrics": {"var_meta": {"vars": vars, "nobs": nobs}}}

    model = VAR(X)
    sel = model.select_order(maxlags=maxlags)

    def _safe_int(v: Any) -> int | None:
        try:
            return None if v is None else int(v)
        except Exception:
            return None

    selected_aic = _safe_int(sel.aic) or 1
    selected_bic = _safe_int(sel.bic)
    selected_hqic = _safe_int(getattr(sel, "hqic", None))
    selected_fpe = _safe_int(getattr(sel, "fpe", None))

    tab = pd.DataFrame([
        {"criterion": "aic", "selected_lag": selected_aic},
        {"criterion": "bic", "selected_lag": selected_bic},
        {"criterion": "hqic", "selected_lag": selected_hqic},
        {"criterion": "fpe", "selected_lag": selected_fpe},
    ])

    res = model.fit(selected_aic)

    return {
        "tables": {"var_selection": tab},
        "metrics": {"var_meta": {"vars": list(X.columns), "nobs": nobs, "selected_lag_aic": selected_aic}},
        "models": {"var_model": res},
    }


def run_granger_pack(df: pd.DataFrame, *, vars: list[str], maxlag: int = 6) -> dict[str, Any]:
    """
    Optionnel (si tu veux l'appeler): table granger_pairwise
    """
    X = _prep_df(df, vars)
    nobs = int(X.shape[0])
    k = int(X.shape[1])

    if k < 2 or nobs < 50:
        tab = pd.DataFrame([{"error": f"Granger: nécessite >=2 variables et nobs>=50 (k={k}, nobs={nobs})"}])
        return {"tables": {"granger_pairwise": tab}, "metrics": {"granger_meta": {"vars": vars, "nobs": nobs}}}

    out_rows = []
    cols = list(X.columns)

    for caused in cols:
        for causing in cols:
            if caused == causing:
                continue
            try:
                res = grangercausalitytests(X[[caused, causing]], maxlag=maxlag, verbose=False)
                pvals = [res[l][0]["ssr_ftest"][1] for l in range(1, maxlag + 1)]
                out_rows.append({"caused": caused, "causing": causing, "min_pvalue_ssr_ftest": float(min(pvals))})
            except Exception as e:
                out_rows.append({"caused": caused, "causing": causing, "error": str(e)})

    tab = pd.DataFrame(out_rows).sort_values(["min_pvalue_ssr_ftest"], na_position="last")

    return {"tables": {"granger_pairwise": tab}, "metrics": {"granger_meta": {"vars": cols, "maxlag": maxlag}}}
