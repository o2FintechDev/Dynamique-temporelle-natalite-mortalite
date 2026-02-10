# src/econometrics/cointegration.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

DISPLAY_NAME_MAP = {
    "Croissance_Naturelle": "CN",
    "Masse_monetaire": "M3",
    "Nb_mariages": "Mariages",
}

def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None

def _disp(name: str) -> str:
    return DISPLAY_NAME_MAP.get(name, name)


def _rename_df_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out.columns = [_disp(c) for c in out.columns]
    out.index = [_disp(i) if isinstance(i, str) else i for i in out.index]
    return out


def _rename_cols_in_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: _disp(x) if isinstance(x, str) else x)
    return out


def _select_var_lag_aic(X: pd.DataFrame, maxlags: int) -> int:
    sel = VAR(X).select_order(maxlags=maxlags)
    try:
        p = int(sel.aic)
    except Exception:
        p = 1
    return max(1, min(int(maxlags), p))


def _engle_granger_pairwise(X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cols = list(X.columns)
    eg_rows = []
    eg_fail = 0

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            try:
                stat, pval, crit = coint(X[a], X[b])
                eg_rows.append(
                    {
                        "x": a,
                        "y": b,
                        "stat": _safe_float(stat),
                        "pvalue": _safe_float(pval),
                        "crit_1": _safe_float(crit[0]) if len(crit) > 0 else None,
                        "crit_5": _safe_float(crit[1]) if len(crit) > 1 else None,
                        "crit_10": _safe_float(crit[2]) if len(crit) > 2 else None,
                    }
                )
            except Exception:
                eg_fail += 1

    tbl_eg = pd.DataFrame(eg_rows)

    if not tbl_eg.empty and "pvalue" in tbl_eg.columns:
        pvals = pd.to_numeric(tbl_eg["pvalue"], errors="coerce")
        eg_p_min = _safe_float(pvals.min())
        eg_p_q10 = _safe_float(pvals.quantile(0.10))
        eg_n = int(tbl_eg.shape[0])
    else:
        eg_p_min = None
        eg_p_q10 = None
        eg_n = 0

    meta = {
        "mode": "pairwise_audit_only",
        "n_pairs": int(eg_n),
        "n_fail": int(eg_fail),
        "eg_p_min": eg_p_min,
        "eg_p_q10": eg_p_q10,
        "warning": "Engle–Granger est bivarié. Ici: tests pairwise, usage audit/indication, pas décision système.",
    }
    return tbl_eg, meta


def _johansen(X: pd.DataFrame, *, det_order: int, k_ar_diff: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cols = list(X.columns)
    k = int(len(cols))

    try:
        joh = coint_johansen(X, det_order=det_order, k_ar_diff=k_ar_diff)
    except Exception as e:
        tbl = pd.DataFrame([{"note": "Johansen indisponible", "error": type(e).__name__}])
        meta = {"available": False, "error": type(e).__name__}
        return tbl, meta

    trace_stat = np.asarray(joh.lr1, dtype=float)
    trace_crit5 = np.asarray(joh.cvt[:, 1], dtype=float)  # col 1 = 95%
    reject5 = (trace_stat > trace_crit5)

    # rang via rejets successifs (trace)
    rank = int(np.sum(reject5))
    rank = max(0, min(rank, k - 1))

    tbl_joh = pd.DataFrame(
        {
            "r": list(range(k)),
            "trace_stat": trace_stat,
            "trace_cv_95": trace_crit5,
            "reject_95": [bool(x) for x in reject5],
            "maxeig_stat": np.asarray(joh.lr2, dtype=float),
            "maxeig_cv_95": np.asarray(joh.cvm[:, 1], dtype=float),
        }
    )

    meta = {
        "available": True,
        "det_order": int(det_order),
        "k_ar_diff": int(k_ar_diff),
        "rank_selected_trace_95": int(rank),
        "eigenvalues": [float(x) for x in np.asarray(joh.eig, dtype=float).ravel().tolist()] if getattr(joh, "eig", None) is not None else None,
    }
    return tbl_joh, meta


def _vecm_params_table(res_vecm: Any, cols: list[str]) -> pd.DataFrame:
    rows = []

    beta = getattr(res_vecm, "beta", None)
    alpha = getattr(res_vecm, "alpha", None)
    gamma = getattr(res_vecm, "gamma", None)

    if beta is not None:
        B = np.asarray(beta, dtype=float)  # (k, r)
        for j in range(B.shape[1]):
            for i, var in enumerate(cols):
                rows.append({"block": "beta", "eq": int(j + 1), "var": var, "value": float(B[i, j])})

    if alpha is not None:
        A = np.asarray(alpha, dtype=float)  # (k, r)
        for j in range(A.shape[1]):
            for i, var in enumerate(cols):
                rows.append({"block": "alpha", "eq": int(j + 1), "var": var, "value": float(A[i, j])})

    if gamma is not None:
        G = np.asarray(gamma, dtype=float)
        # mapping exact dépend de la structure interne; on garde un format stable pour audit
        for i, var in enumerate(cols):
            for j in range(G.shape[1]):
                rows.append({"block": "gamma", "eq": None, "var": var, "param": f"g{j+1}", "value": float(G[i, j])})

    if not rows:
        return pd.DataFrame([{"note": "Paramètres VECM indisponibles"}])

    tbl = pd.DataFrame(rows)
    if "param" not in tbl.columns:
        tbl["param"] = None
    return tbl


def cointegration_pack(
    df_vars: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Pack cointégration auditable + décision VAR_diff vs VECM.

    Paramètres (compatibles avec anciens appels):
      - det_order: composante déterministe Johansen (statsmodels)
      - k_ar_diff: nombre de retards en différences (Johansen/VECM)
      - kwargs: accepte alias historiques (k_ar, lags, nlags, lag_order)

    Sorties:
      tables:
        - tbl.coint.eg
        - tbl.coint.johansen
        - tbl.coint.var_vs_vecm_choice
        - (optionnel) tbl.vecm.params
      metrics:
        - m.coint.meta
        - m.coint.audit
        - (optionnel) m.vecm.meta
        - m.note.step6
      models (optionnel):
        - model.vecm
    """

    # -----------------------------
    # Compat kwargs (anti-crash)
    # -----------------------------
    if "k_ar" in kwargs and (k_ar_diff is None or int(k_ar_diff) == 1):
        k_ar_diff = int(kwargs["k_ar"])
    if "lags" in kwargs and (k_ar_diff is None or int(k_ar_diff) == 1):
        k_ar_diff = int(kwargs["lags"])
    if "nlags" in kwargs and (k_ar_diff is None or int(k_ar_diff) == 1):
        k_ar_diff = int(kwargs["nlags"])
    if "lag_order" in kwargs and (k_ar_diff is None or int(k_ar_diff) == 1):
        k_ar_diff = int(kwargs["lag_order"])

    det_order = int(det_order)
    k_ar_diff = int(k_ar_diff)

    # -----------------------------
    # Données
    # -----------------------------
    X = df_vars.dropna().astype(float)
    cols = list(X.columns)
    nobs = int(X.shape[0])
    k = int(X.shape[1])

    # garde-fous
    if k < 2 or nobs < (k_ar_diff + 10):
        tbl_empty = pd.DataFrame()
        meta = {
            "vars": cols,
            "nobs": nobs,
            "k": k,
            "det_order": det_order,
            "k_ar_diff": k_ar_diff,
            "rank": 0,
            "choice": "VAR_diff",
            "rule": "VECM si rank>0 (trace@5%), sinon VAR_diff",
            "eg_n_pairs": 0,
            "eg_n_fail": 0,
            "eg_p_min": None,
            "eg_p_q10": None,
            "trace_stat_0": None,
            "trace_crit5_0": None,
            "trace_reject5_0": None,
        }
        audit = {
            "engle_granger": {"n_pairs": 0, "n_fail": 0, "p_min": None, "p_q10": None},
            "johansen": {
                "det_order": det_order,
                "k_ar_diff": k_ar_diff,
                "nobs": nobs,
                "k": k,
                "cols": cols,
                "error": "insufficient_data",
            },
            "decision": {"rank": 0, "choice": "VAR_diff", "rule": meta["rule"]},
        }
        
        note6 = (
            f"**Cointégration** : données insuffisantes pour tester la cointégration multivariée "
            f"(k={int(k)} variables, nobs={int(nobs)} observations).\n\n"
            "Les tests de cointégration (Engle-Granger et Johansen) ne peuvent pas être appliqués de manière fiable "
            "dans ces conditions.\n\n"
            "Décision conservatrice : modélisation en **VAR sur séries différenciées**, sans estimation de VECM."
        )


        return {
            "tables": {
                "tbl.coint.eg": tbl_empty,
                "tbl.coint.johansen": tbl_empty,
                "tbl.coint.var_vs_vecm_choice": pd.DataFrame(
                    [{"rank": 0, "choice": "VAR_diff", "det_order": det_order, "k_ar_diff": k_ar_diff, "rule": meta["rule"]}],
                    index=["choice"],
                ),
            },
            "metrics": {
                "m.coint.meta": meta,
                "m.coint.audit": audit,
                "m.note.step6": {"markdown": note6, "key_points": meta},
            },
        }

    # -----------------------------
    # 1) Engle–Granger (pairwise, indicatif)
    # -----------------------------
    eg_rows: list[dict[str, Any]] = []
    eg_fail = 0
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            try:
                stat, pval, crit = coint(X[a], X[b])
                eg_rows.append(
                    {
                        "x": a,
                        "y": b,
                        "stat": float(stat),
                        "pvalue": float(pval),
                        "crit_1": float(crit[0]),
                        "crit_5": float(crit[1]),
                        "crit_10": float(crit[2]),
                    }
                )
            except Exception:
                eg_fail += 1

    tbl_eg = pd.DataFrame(eg_rows)

    if not tbl_eg.empty and "pvalue" in tbl_eg.columns:
        eg_p_min = float(pd.to_numeric(tbl_eg["pvalue"], errors="coerce").min())
        eg_p_q10 = float(pd.to_numeric(tbl_eg["pvalue"], errors="coerce").quantile(0.10))
        eg_n = int(len(tbl_eg))
    else:
        eg_p_min = None
        eg_p_q10 = None
        eg_n = 0

    # -----------------------------
    # 2) Johansen (multivarié)
    # -----------------------------
    joh_error: str | None = None
    try:
        joh = coint_johansen(X, det_order=det_order, k_ar_diff=k_ar_diff)

        trace_stat = np.asarray(joh.lr1, dtype=float)          # trace stats
        trace_crit5 = np.asarray(joh.cvt[:, 1], dtype=float)   # 5%
        reject5 = trace_stat > trace_crit5

        # rank via trace au seuil de 5% = nb de rejets successifs
        rank = int(np.sum(reject5))
        rank = max(0, min(rank, k - 1))

        tbl_joh = pd.DataFrame(
            {
                "r": list(range(len(trace_stat))),
                "trace_stat": trace_stat,
                "crit_5": trace_crit5,
                "reject_5": [bool(x) for x in reject5],
                "maxeig_stat": np.asarray(joh.lr2, dtype=float),
                "maxeig_crit_5": np.asarray(joh.cvm[:, 1], dtype=float),
            }
        )

        joh_audit = {
            "det_order": det_order,
            "k_ar_diff": k_ar_diff,
            "nobs": nobs,
            "k": k,
            "cols": cols,
            "trace_stat": trace_stat.tolist(),
            "trace_crit_90": np.asarray(joh.cvt[:, 0], dtype=float).tolist(),
            "trace_crit_95": trace_crit5.tolist(),
            "trace_crit_99": np.asarray(joh.cvt[:, 2], dtype=float).tolist(),
            "reject_95": [bool(x) for x in reject5],
            "maxeig_stat": np.asarray(joh.lr2, dtype=float).tolist(),
            "maxeig_crit_90": np.asarray(joh.cvm[:, 0], dtype=float).tolist(),
            "maxeig_crit_95": np.asarray(joh.cvm[:, 1], dtype=float).tolist(),
            "maxeig_crit_99": np.asarray(joh.cvm[:, 2], dtype=float).tolist(),
        }

        trace_stat_0 = float(trace_stat[0]) if len(trace_stat) else None
        trace_crit5_0 = float(trace_crit5[0]) if len(trace_crit5) else None
        trace_reject5_0 = bool(reject5[0]) if len(reject5) else None

    except Exception as e:
        joh_error = type(e).__name__
        rank = 0
        tbl_joh = pd.DataFrame(
            [
                {
                    "r": 0,
                    "trace_stat": np.nan,
                    "crit_5": np.nan,
                    "reject_5": False,
                    "maxeig_stat": np.nan,
                    "maxeig_crit_5": np.nan,
                    "error": joh_error,
                }
            ]
        )
        joh_audit = {
            "det_order": det_order,
            "k_ar_diff": k_ar_diff,
            "nobs": nobs,
            "k": k,
            "cols": cols,
            "error": joh_error,
        }
        trace_stat_0 = None
        trace_crit5_0 = None
        trace_reject5_0 = None

    # -----------------------------
    # 3) Décision VAR diff vs VECM
    # -----------------------------
    use_vecm = (rank > 0) and (joh_error is None)
    choice = "VECM" if use_vecm else "VAR_diff"

    tbl_choice = pd.DataFrame(
        [
            {
                "rank": int(rank),
                "choice": choice,
                "det_order": det_order,
                "k_ar_diff": k_ar_diff,
                "rule": "VECM si rang Johansen (trace au seuil de 5%) > 0, sinon VAR en différences",
            }
        ],
        index=pd.Index(["choice"]),
    )

    # -----------------------------
    # 4) Metrics (meta + audit)
    # -----------------------------
    coint_meta = {
        "vars": cols,
        "nobs": nobs,
        "k": k,
        "det_order": det_order,
        "k_ar_diff": k_ar_diff,
        "rank": int(rank),
        "choice": choice,
        "rule": "VECM si rank>0 (trace au seuil 5%), sinon VAR_diff",
        "eg_n_pairs": int(eg_n),
        "eg_n_fail": int(eg_fail),
        "eg_p_min": eg_p_min,
        "eg_p_q10": eg_p_q10,
        "trace_stat_0": trace_stat_0,
        "trace_crit5_0": trace_crit5_0,
        "trace_reject5_0": trace_reject5_0,
        "johansen_error": joh_error,
    }

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
            "rule": "VECM si rank>0 (trace au seuil de 5%), sinon VAR_diff",
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

        beta = pd.DataFrame(vecm.beta, index=cols)   # (k x rank)
        alpha = pd.DataFrame(vecm.alpha, index=cols) # (k x rank)

        # table lisible : 1ère relation (col 0)
        if beta.shape[1] >= 1 and alpha.shape[1] >= 1:
            out["tables"]["tbl.vecm.params"] = pd.DataFrame(
                {
                    "alpha_1": alpha.iloc[:, 0].astype(float).values,
                    "beta_1": beta.iloc[:, 0].astype(float).values,
                },
                index=cols,
            )

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
    # 6) Note narrative Step6
    # -----------------------------
    note6 = (
            f"**Cointégration** : le test de Johansen (statistique de trace au seuil de 5\\%) "
            f"indique un rang de cointégration **r = {int(rank)}**, révélant l'existence de relations de long terme "
            f"entre les variables considérées. \n\n Conformément à la règle décisionnelle "
            f"(VECM si rang > 0, sinon VAR en différences), le modèle retenu est un **{choice}**, "
            f"avec det_order={det_order} et k_ar_diff={k_ar_diff}.\n\n "
            + (
                "Le VECM est estimé afin de modéliser conjointement les équilibres de long terme "
                "via les vecteurs de cointégration (β) et les mécanismes d'ajustement vers ces équilibres "
                "à court terme (α).\n\n"
                if use_vecm
                else
                "L'absence de cointégration conduit à privilégier un VAR en différences, "
                "sans modélisation explicite de relations de long terme."
            )
            + (f" (Échec du test de Johansen : {joh_error})." if joh_error else "")
        )

    out["metrics"]["m.note.step6"] = {
        "markdown": note6,
        "key_points": {
            "rank": int(rank),
            "choice": choice,
            "vars": cols,
            "nobs": nobs,
            "det_order": det_order,
            "k_ar_diff": k_ar_diff,
            "eg_p_min": eg_p_min,
            "eg_p_q10": eg_p_q10,
            "trace_stat_0": trace_stat_0,
            "trace_crit5_0": trace_crit5_0,
            "trace_reject5_0": trace_reject5_0,
            "johansen_error": joh_error,
        },
    }
    # --------------------------------------------------
    # DISPLAY ONLY — renommage des variables (UI / LaTeX)
    # --------------------------------------------------

    out["tables"]["tbl.coint.eg"] = _rename_cols_in_df(
        out["tables"]["tbl.coint.eg"], ["x", "y"]
    )

    out["tables"]["tbl.coint.johansen"] = _rename_df_display(
        out["tables"]["tbl.coint.johansen"]
    )

    out["tables"]["tbl.coint.var_vs_vecm_choice"] = _rename_df_display(
        out["tables"]["tbl.coint.var_vs_vecm_choice"]
    )

    if "tbl.vecm.params" in out["tables"]:
        out["tables"]["tbl.vecm.params"] = _rename_df_display(
            out["tables"]["tbl.vecm.params"]
        )

    return out
