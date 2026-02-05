# economics/cointegration.py

from __future__ import annotations
from typing import Any
import pandas as pd

from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

def cointegration_pack(df_vars: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> dict[str, Any]:
    X = df_vars.dropna()
    cols = list(X.columns)

    # Engle–Granger pairwise (indicatif)
    eg_rows = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            a, b = cols[i], cols[j]
            stat, pval, crit = coint(X[a], X[b])
            eg_rows.append({"x": a, "y": b, "stat": float(stat), "pvalue": float(pval),
                            "crit_1": float(crit[0]), "crit_5": float(crit[1]), "crit_10": float(crit[2])})
    tbl_eg = pd.DataFrame(eg_rows)

    # Johansen (multivarié)
    joh = coint_johansen(X, det_order=det_order, k_ar_diff=k_ar_diff)
    # rang: trace stat vs crit 5%
    rank = 0
    for r in range(len(cols)):
        if joh.lr1[r] > joh.cvt[r, 1]:
            rank += 1
    tbl_joh = pd.DataFrame({
        "r": list(range(len(cols))),
        "trace_stat": joh.lr1,
        "crit_5": joh.cvt[:, 1],
        "reject_5": [bool(joh.lr1[r] > joh.cvt[r, 1]) for r in range(len(cols))],
    })

    # Choix VAR diff vs VECM
    use_vecm = rank > 0
    tbl_choice = pd.DataFrame([{
        "rank": int(rank),
        "choice": "VECM" if use_vecm else "VAR_diff",
        "rule": "VECM si rang Johansen > 0, sinon VAR en différences",
    }]).set_index(pd.Index(["choice"]))

    out: dict[str, Any] = {
        "tables": {
            "tbl.coint.eg": tbl_eg,
            "tbl.coint.johansen": tbl_joh,
            "tbl.coint.var_vs_vecm_choice": tbl_choice,
        },
        "metrics": {"m.coint.meta": {"vars": cols, "rank": int(rank), "nobs": int(X.shape[0]), "choice": choice}},

    }

    if use_vecm:
        vecm = VECM(X, k_ar_diff=k_ar_diff, deterministic="co").fit()
        # params table: alpha/beta (simple dump)
        beta = pd.DataFrame(vecm.beta, index=cols)
        alpha = pd.DataFrame(vecm.alpha, index=cols)
        out["tables"]["tbl.vecm.params"] = pd.DataFrame({
            "alpha": alpha.iloc[:, 0].values,
            "beta": beta.iloc[:, 0].values,
        }, index=cols)
        out["models"] = {"model.vecm": vecm}
    choice = "VECM" if use_vecm else "VAR_diff"

    note6 = (
        f"**Étape 6 — Cointégration** : rang Johansen = {int(rank)} ⇒ choix **{choice}**. "
        "Si VECM : relation(s) de long terme (beta) + vitesse d’ajustement (alpha) via terme de correction d’erreur."
    )
    out["metrics"]["m.note.step6"] = {
    "markdown": note6,
    "key_points": {"rank": int(rank), "choice": choice, "vars": cols, "nobs": int(X.shape[0])}}
    return out
