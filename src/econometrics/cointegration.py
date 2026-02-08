# economics/cointegration.py

from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd

from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.stattools import coint


def cointegration_pack(df_vars: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> dict[str, Any]:
    X = df_vars.dropna().astype(float)
    cols = list(X.columns)
    nobs = int(X.shape[0])
    k = int(X.shape[1])

    # -----------------------------
    # 1) Engle–Granger (pairwise, indicatif)
    # -----------------------------
    eg_rows = []
    eg_fail = 0
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            try:
                stat, pval, crit = coint(X[a], X[b])
                eg_rows.append({
                    "x": a,
                    "y": b,
                    "stat": float(stat),
                    "pvalue": float(pval),
                    "crit_1": float(crit[0]),
                    "crit_5": float(crit[1]),
                    "crit_10": float(crit[2]),
                })
            except Exception:
                eg_fail += 1
                continue

    tbl_eg = pd.DataFrame(eg_rows)
    if not tbl_eg.empty:
        eg_p_min = float(tbl_eg["pvalue"].min())
        eg_p_q10 = float(tbl_eg["pvalue"].quantile(0.10))
        eg_n = int(len(tbl_eg))
    else:
        eg_p_min = None
        eg_p_q10 = None
        eg_n = 0

    # -----------------------------
    # 2) Johansen (multivarié)
    # -----------------------------
    joh = coint_johansen(X, det_order=det_order, k_ar_diff=k_ar_diff)

    # rank via trace test @ 5%
    # joh.lr1 : trace statistics
    # joh.cvt : critical values for trace; column 1 = 5%
    trace_stat = np.asarray(joh.lr1, dtype=float)
    trace_crit5 = np.asarray(joh.cvt[:, 1], dtype=float)
    reject5 = (trace_stat > trace_crit5)

    rank = int(np.sum(reject5))  # nombre de rejets successifs
    # garde-fou : rank ∈ [0, k-1]
    rank = max(0, min(rank, k - 1))

    tbl_joh = pd.DataFrame({
        "r": list(range(len(cols))),
        "trace_stat": trace_stat,
        "crit_5": trace_crit5,
        "reject_5": [bool(x) for x in reject5],
    })

    # Audit Johansen : valeurs utiles + paramètres
    joh_audit = {
        "det_order": int(det_order),
        "k_ar_diff": int(k_ar_diff),
        "nobs": nobs,
        "k": k,
        "cols": cols,
        "trace_stat": trace_stat.tolist(),
        "trace_crit_90": np.asarray(joh.cvt[:, 0], dtype=float).tolist(),
        "trace_crit_95": trace_crit5.tolist(),
        "trace_crit_99": np.asarray(joh.cvt[:, 2], dtype=float).tolist(),
        "reject_95": [bool(x) for x in reject5],
        # max eigen test (si tu veux l’audit complet)
        "maxeig_stat": np.asarray(joh.lr2, dtype=float).tolist(),
        "maxeig_crit_90": np.asarray(joh.cvm[:, 0], dtype=float).tolist(),
        "maxeig_crit_95": np.asarray(joh.cvm[:, 1], dtype=float).tolist(),
        "maxeig_crit_99": np.asarray(joh.cvm[:, 2], dtype=float).tolist(),
    }

    # -----------------------------
    # 3) Décision VAR diff vs VECM (règle explicite)
    # -----------------------------
    use_vecm = rank > 0
    choice = "VECM" if use_vecm else "VAR_diff"

    tbl_choice = pd.DataFrame([{
        "rank": int(rank),
        "choice": choice,
        "det_order": int(det_order),
        "k_ar_diff": int(k_ar_diff),
        "rule": "VECM si rang Johansen (trace @5%) > 0, sinon VAR en différences",
    }]).set_index(pd.Index(["choice"]))

    # -----------------------------
    # 4) Métriques d’auditabilité (meta + audit)
    # -----------------------------
    coint_meta = {
        # identité
        "vars": cols,
        "nobs": nobs,
        "k": k,
        # paramétrage test
        "det_order": int(det_order),
        "k_ar_diff": int(k_ar_diff),
        # résultat Johansen
        "rank": int(rank),
        "choice": choice,
        "rule": "VECM si rank>0 (trace@5%), sinon VAR_diff",
        # résumés EG (indicatif)
        "eg_n_pairs": int(eg_n),
        "eg_n_fail": int(eg_fail),
        "eg_p_min": eg_p_min,
        "eg_p_q10": eg_p_q10,
        # résumé Johansen (trace)
        "trace_stat_0": float(trace_stat[0]) if len(trace_stat) else None,
        "trace_crit5_0": float(trace_crit5[0]) if len(trace_crit5) else None,
        "trace_reject5_0": bool(reject5[0]) if len(reject5) else None,
    }

    # Audit détaillé séparé (traçabilité complète)
    coint_audit = {
        "engle_granger": {
            "n_pairs": int(eg_n),
            "n_fail": int(eg_fail),
            "p_min": eg_p_min,
            "p_q10": eg_p_q10,
        },
        "johansen": joh_audit,
        "decision": {
            "rank": int(rank),
            "choice": choice,
            "rule": "VECM si rank>0 (trace@5%), sinon VAR_diff",
        },
    }

    out: dict[str, Any] = {
        "tables": {
            "tbl.coint.eg": tbl_eg,
            "tbl.coint.johansen": tbl_joh,
            "tbl.coint.var_vs_vecm_choice": tbl_choice,
        },
        "metrics": {
            "m.coint.meta": coint_meta,
            "m.coint.audit": coint_audit,
        },
    }

    # -----------------------------
    # 5) Si VECM : paramètres + audit VECM
    # -----------------------------
    if use_vecm:
        vecm = VECM(X, k_ar_diff=k_ar_diff, deterministic="co").fit()

        # beta (k x rank) / alpha (k x rank)
        beta = pd.DataFrame(vecm.beta, index=cols)
        alpha = pd.DataFrame(vecm.alpha, index=cols)

        # table lisible : première relation (col 0) si rank>=1
        rnk = int(beta.shape[1]) if beta.ndim == 2 else 0
        if rnk >= 1:
            out["tables"]["tbl.vecm.params"] = pd.DataFrame({
                "alpha_1": alpha.iloc[:, 0].values,
                "beta_1": beta.iloc[:, 0].values,
            }, index=cols)

        # audit VECM (complet)
        out["metrics"]["m.vecm.meta"] = {
            "nobs": nobs,
            "vars": cols,
            "rank": int(rank),
            "k_ar_diff": int(k_ar_diff),
            "deterministic": "co",
            "alpha_shape": list(vecm.alpha.shape) if hasattr(vecm, "alpha") else None,
            "beta_shape": list(vecm.beta.shape) if hasattr(vecm, "beta") else None,
            "alpha": alpha.to_dict(orient="list"),
            "beta": beta.to_dict(orient="list"),
        }

        out["models"] = {"model.vecm": vecm}

    # -----------------------------
    # 6) Note narrative step6 (avec key_points auditables)
    # -----------------------------
    note6 = (
        f"**Étape 6 — Cointégration** : Johansen (trace @5%) ⇒ rang = **{int(rank)}** "
        f"⇒ choix **{choice}** (det_order={int(det_order)}, k_ar_diff={int(k_ar_diff)}). "
        "Si VECM : interprétation via vecteurs de cointégration (β) et vitesses d’ajustement (α)."
    )

    out["metrics"]["m.note.step6"] = {
        "markdown": note6,
        "key_points": {
            "rank": int(rank),
            "choice": choice,
            "vars": cols,
            "nobs": nobs,
            "det_order": int(det_order),
            "k_ar_diff": int(k_ar_diff),
            "eg_p_min": eg_p_min,
            "eg_p_q10": eg_p_q10,
            "trace_stat_0": coint_meta["trace_stat_0"],
            "trace_crit5_0": coint_meta["trace_crit5_0"],
            "trace_reject5_0": coint_meta["trace_reject5_0"],
        },
    }

    return out
