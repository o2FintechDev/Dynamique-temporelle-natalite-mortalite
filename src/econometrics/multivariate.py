# econometrics/multivariate.py
from __future__ import annotations
from typing import Any
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR

def var_pack(df_vars: pd.DataFrame, maxlags: int = 12) -> dict[str, Any]:
    model = VAR(df_vars.dropna())
    sel = model.select_order(maxlags=maxlags)

    # sélection AIC par défaut
    p = int(sel.aic)
    res = model.fit(p)

    # table sélection
    tbl_sel = pd.DataFrame({
        "aic": [sel.aic],
        "bic": [sel.bic],
        "hqic": [sel.hqic],
        "fpe": [sel.fpe],
        "selected_aic": [p],
    }, index=["lag_selection"])

    # Granger (pairwise)
    granger_rows = []
    cols = list(df_vars.columns)
    for caused in cols:
        for causing in cols:
            if caused == causing:
                continue
            try:
                test = res.test_causality(caused=caused, causing=causing, kind="f")
                granger_rows.append({
                    "caused": caused, "causing": causing,
                    "stat": float(test.test_statistic),
                    "pvalue": float(test.pvalue),
                })
            except Exception:
                continue
    tbl_granger = pd.DataFrame(granger_rows)

    # IRF figure
    irf = res.irf(12)
    fig_irf = irf.plot(orth=False)
    fig_irf.suptitle("IRF (VAR)")

    # FEVD table (horizon 12)
    fevd = res.fevd(12)
    # flatten
    fevd_rows = []
    for i, target in enumerate(cols):
        m = fevd.decomp[:, i, :]  # (h, shocks)
        for h in range(m.shape[0]):
            for j, shock in enumerate(cols):
                fevd_rows.append({"target": target, "shock": shock, "h": h, "share": float(m[h, j])})
    tbl_fevd = pd.DataFrame(fevd_rows)

    metrics = {"vars": cols, "selected_lag_aic": p, "nobs": int(res.nobs)}
    
    note5 = (
    f"**Étape 5 — VAR(p)** : sélection AIC → p={p}, variables={cols}, nobs={int(res.nobs)}. "
    "IRF et FEVD décrivent la dynamique interne entre composantes STL (level/trend/seasonal). "
    "Les tests de Granger sont reportés à titre descriptif (non causal).")
    return {
        "tables": {
            "tbl.var.lag_selection": tbl_sel,
            "tbl.var.granger": tbl_granger,
            "tbl.var.fevd": tbl_fevd,
        },
        "metrics": {
            "m.var.meta": metrics,
            "m.note.step5": {"markdown": note5, "key_points": metrics},
        },
        "models": {"model.var.best": res},
        "figures": {"fig.var.irf": fig_irf},
    }
