# econometrics/multivariate.py
from __future__ import annotations
from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.tsa.api import VAR

def sims_causality_test(
    X: pd.DataFrame, *, caused: str, causing: str, p: int, q: int
) -> dict[str, float]:
    """
    Causalité à la Sims (anticipative) :
    H0 : les leads de `causing` (X_{t+1}..X_{t+q}) n'expliquent pas `caused` (Y_t)
    Test = Wald sur les coefficients des leads dans une régression OLS
    incluant les lags (1..p) de toutes les variables du système.
    """
    df = X.copy()

    # Variable dépendante Y_t
    y = df[caused]

    # Régressseurs = lags de toutes les variables
    reg = {}
    for var in df.columns:
        for i in range(1, p + 1):
            reg[f"{var}_lag{i}"] = df[var].shift(i)

    # Leads de la variable causing
    lead_names = []
    for j in range(1, q + 1):
        name = f"{causing}_lead{j}"
        reg[name] = df[causing].shift(-j)
        lead_names.append(name)

    Z = pd.DataFrame(reg)
    data = pd.concat([y, Z], axis=1).dropna()

    # Si trop peu d'observations => on renvoie NA
    if data.shape[0] < (p + q + 5):
        return {"stat": float("nan"), "pvalue": float("nan")}

    y_clean = data[caused]
    X_clean = sm.add_constant(data.drop(columns=[caused]))

    res_ols = sm.OLS(y_clean, X_clean).fit()

    # Wald: coefficients des leads = 0
    lead_cols = [c for c in X_clean.columns if c in lead_names]

    if len(lead_cols) == 0:
        return {"stat": float("nan"), "pvalue": float("nan")}

    # matrice de restriction R * beta = 0
    R = []
    for col in lead_cols:
        r = [0.0] * X_clean.shape[1]
        r[X_clean.columns.get_loc(col)] = 1.0
        R.append(r)

    test = res_ols.f_test(R)

    stat = float(test.fvalue)
    pval = float(test.pvalue)


    return {"stat": float(stat), "pvalue": float(pval)}


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

    # Sims (causalité anticipative) : tester q = 1..p
    sims_rows = []
    for caused in cols:
        for causing in cols:
            if caused == causing:
                continue
            for q in range(1, p + 1):
                try:
                    t = sims_causality_test(df_vars.dropna(), caused=caused, causing=causing, p=p, q=q)
                    sims_rows.append({
                        "caused": caused,
                        "causing": causing,
                        "lead_q": q,
                        "stat": float(t["stat"]),
                        "pvalue": float(t["pvalue"]),
                    })
                except Exception as e:
                    sims_rows.append({
                        "caused": caused,
                        "causing": causing,
                        "lead_q": q,
                        "stat": None,
                        "pvalue": None,
                        "error": type(e).__name__,
                    })
    tbl_sims = pd.DataFrame(sims_rows)

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
    
    sims_meta = {
        "lead_q_tested": list(range(1, p + 1)),
        "method": "joint_f_test_on_leads (OLS eq-by-eq, Sims)",}

    note5 = (
    f"**Étape 5 — VAR(p)** : sélection AIC → p={p}, variables={cols}, nobs={int(res.nobs)}. "
    "IRF et FEVD décrivent la dynamique interne entre composantes STL (level/trend/seasonal). "
    "Les tests de Granger sont reportés à titre descriptif (non causal).")
    return {
        "tables": {
            "tbl.var.lag_selection": tbl_sel,
            "tbl.var.granger": tbl_granger,
            "tbl.var.sims": tbl_sims,
            "tbl.var.fevd": tbl_fevd,
        },
        "metrics": {
            "m.var.meta": metrics,
            "m.var.sims": sims_meta,
            "m.note.step5": {"markdown": note5, "key_points": metrics},
        },
        "models": {"model.var.best": res},
        "figures": {"fig.var.irf": fig_irf},
    }
