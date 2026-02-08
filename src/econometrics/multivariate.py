# src/econometrics/multivariate.py
from __future__ import annotations
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def sims_causality_test(
    X: pd.DataFrame, *, caused: str, causing: str, p: int, q: int
) -> dict[str, float | int | None]:
    """
    Causalité à la Sims (anticipative):
    H0 : les leads de `causing` (t+1..t+q) n'expliquent pas `caused` (t)
    Test = Wald (F-test) sur coefficients des leads dans une OLS eq-by-eq,
    avec lags 1..p de toutes les variables comme contrôles.

    Retourne:
      - stat, pvalue (float | nan)
      - nobs_used (int)
      - df_denom, df_num (optionnels)
    """
    df = X.copy()

    y = df[caused]
    reg: dict[str, Any] = {}

    # Lags de toutes les variables
    for var in df.columns:
        for i in range(1, p + 1):
            reg[f"{var}_lag{i}"] = df[var].shift(i)

    # Leads de causing
    lead_names = []
    for j in range(1, q + 1):
        name = f"{causing}_lead{j}"
        reg[name] = df[causing].shift(-j)
        lead_names.append(name)

    Z = pd.DataFrame(reg)
    data = pd.concat([y, Z], axis=1).dropna()

    nobs_used = int(data.shape[0])

    # Trop peu d'observations => NA
    if nobs_used < (p + q + 5):
        return {"stat": float("nan"), "pvalue": float("nan"), "nobs_used": nobs_used}

    y_clean = data[caused]
    X_clean = sm.add_constant(data.drop(columns=[caused]), has_constant="add")

    res_ols = sm.OLS(y_clean, X_clean).fit()

    lead_cols = [c for c in X_clean.columns if c in lead_names]
    if len(lead_cols) == 0:
        return {"stat": float("nan"), "pvalue": float("nan"), "nobs_used": nobs_used}

    # R beta = 0 pour les leads
    R = []
    for col in lead_cols:
        r = [0.0] * X_clean.shape[1]
        r[X_clean.columns.get_loc(col)] = 1.0
        R.append(r)

    test = res_ols.f_test(R)

    stat = float(test.fvalue)
    pval = float(test.pvalue)

    # df_num/df_denom si dispo
    df_num = getattr(test, "df_num", None)
    df_denom = getattr(test, "df_denom", None)

    out: dict[str, float | int | None] = {
        "stat": float(stat),
        "pvalue": float(pval),
        "nobs_used": nobs_used,
    }
    if df_num is not None:
        out["df_num"] = int(df_num)
    if df_denom is not None:
        out["df_denom"] = int(df_denom)
    return out


def var_pack(df_vars: pd.DataFrame, maxlags: int = 12) -> dict[str, Any]:
    """
    Pack VAR auditable:
      - sélection de lag (critères)
      - estimation VAR(p)
      - diagnostics: stabilité (roots), résidus (whiteness/normalité)
      - Granger pairwise (fit VAR)
      - Sims leads tests + audit erreurs
      - IRF + FEVD + paramètres horizon
      - meta + audit payloads (reproductibilité)
    """
    X0 = df_vars.copy()
    cols = list(X0.columns)

    # ---- couverture / nettoyage ----
    X = X0.dropna().astype(float)
    nobs_raw = int(X0.shape[0])
    nobs_used = int(X.shape[0])
    dropna_rows = int(nobs_raw - nobs_used)

    model = VAR(X)
    sel = model.select_order(maxlags=maxlags)

    # sélection AIC par défaut (statsmodels peut renvoyer np.int64/float)
    p = int(sel.aic)

    res = model.fit(p)

    # -----------------------------
    # Table sélection de lag
    # -----------------------------
    tbl_sel = pd.DataFrame(
        {
            "aic": [float(sel.aic)],
            "bic": [float(sel.bic)],
            "hqic": [float(sel.hqic)],
            "fpe": [float(sel.fpe)],
            "maxlags": [int(maxlags)],
            "selected_aic": [int(p)],
            "nobs_raw": [nobs_raw],
            "nobs_used": [int(res.nobs)],
            "rows_dropped_dropna": [dropna_rows],
        },
        index=["lag_selection"],
    )

    # -----------------------------
    # Granger (pairwise)
    # -----------------------------
    granger_rows = []
    for caused in cols:
        for causing in cols:
            if caused == causing:
                continue
            try:
                test = res.test_causality(caused=caused, causing=causing, kind="f")
                granger_rows.append(
                    {
                        "caused": caused,
                        "causing": causing,
                        "stat": _safe_float(test.test_statistic),
                        "pvalue": _safe_float(test.pvalue),
                        "df_denom": _safe_float(getattr(test, "df_denom", None)),
                        "df_num": _safe_float(getattr(test, "df_num", None)),
                    }
                )
            except Exception as e:
                granger_rows.append(
                    {
                        "caused": caused,
                        "causing": causing,
                        "stat": None,
                        "pvalue": None,
                        "error": type(e).__name__,
                    }
                )
    tbl_granger = pd.DataFrame(granger_rows)

    # -----------------------------
    # Sims (leads) : q=1..p
    # + audit erreurs / couverture
    # -----------------------------
    sims_rows = []
    sims_errors = 0
    for caused in cols:
        for causing in cols:
            if caused == causing:
                continue
            for q in range(1, p + 1):
                try:
                    t = sims_causality_test(X, caused=caused, causing=causing, p=p, q=q)
                    sims_rows.append(
                        {
                            "caused": caused,
                            "causing": causing,
                            "lead_q": int(q),
                            "stat": _safe_float(t.get("stat")),
                            "pvalue": _safe_float(t.get("pvalue")),
                            "nobs_used": int(t.get("nobs_used", 0)),
                            "df_num": t.get("df_num"),
                            "df_denom": t.get("df_denom"),
                        }
                    )
                except Exception as e:
                    sims_errors += 1
                    sims_rows.append(
                        {
                            "caused": caused,
                            "causing": causing,
                            "lead_q": int(q),
                            "stat": None,
                            "pvalue": None,
                            "nobs_used": None,
                            "error": type(e).__name__,
                        }
                    )
    tbl_sims = pd.DataFrame(sims_rows)

    # -----------------------------
    # IRF figure
    # -----------------------------
    irf_h = 12
    irf = res.irf(irf_h)
    fig_irf = irf.plot(orth=False)
    fig_irf.suptitle("IRF (VAR)")

    # -----------------------------
    # FEVD (horizon 12)
    # -----------------------------
    fevd_h = 12
    fevd = res.fevd(fevd_h)

    fevd_rows = []
    for i, target in enumerate(cols):
        m = fevd.decomp[:, i, :]  # (h, shocks)
        for h in range(m.shape[0]):
            for j, shock in enumerate(cols):
                fevd_rows.append(
                    {"target": target, "shock": shock, "h": int(h), "share": float(m[h, j])}
                )
    tbl_fevd = pd.DataFrame(fevd_rows)

    # -----------------------------
    # Diagnostics VAR: stabilité + résidus
    # -----------------------------
    # Stabilité: roots (toutes doivent être < 1 en module)
    roots = getattr(res, "roots", None)
    roots_abs_max = None
    stable = None
    if roots is not None:
        try:
            roots = np.asarray(roots, dtype=complex)
            roots_abs = np.abs(roots)
            roots_abs_max = float(np.max(roots_abs)) if roots_abs.size else None
            stable = bool(roots_abs_max is not None and roots_abs_max < 1.0)
        except Exception:
            roots_abs_max = None
            stable = None

    # Résidus: matrice covariance + tests normalité/whiteness si dispo
    # res.test_whiteness / res.test_normality existent selon version statsmodels
    whiteness_p = None
    normality_p = None
    try:
        w = res.test_whiteness(nlags=min(12, max(1, p)))
        whiteness_p = _safe_float(getattr(w, "pvalue", None))
    except Exception:
        whiteness_p = None

    try:
        n = res.test_normality()
        normality_p = _safe_float(getattr(n, "pvalue", None))
    except Exception:
        normality_p = None

    sigma_u = None
    try:
        sigma_u = res.sigma_u
        sigma_u = np.asarray(sigma_u, dtype=float).tolist()
    except Exception:
        sigma_u = None

    # -----------------------------
    # Metrics (meta + audit)
    # -----------------------------
    metrics_meta = {
        "vars": cols,
        "selected_lag_aic": int(p),
        "maxlags": int(maxlags),
        "nobs_raw": nobs_raw,
        "nobs_used": int(res.nobs),
        "rows_dropped_dropna": dropna_rows,
        "irf_h": int(irf_h),
        "fevd_h": int(fevd_h),
        "stable": stable,
        "roots_abs_max": roots_abs_max,
        "whiteness_p": whiteness_p,
        "normality_p": normality_p,
    }

    sims_meta = {
        "lead_q_tested": list(range(1, p + 1)),
        "method": "joint_f_test_on_leads (OLS eq-by-eq, Sims)",
        "n_errors": int(sims_errors),
        "min_nobs_used": int(tbl_sims["nobs_used"].min()) if "nobs_used" in tbl_sims.columns and not tbl_sims.empty else None,
    }

    # Audit complet: ce qui permet de reproduire la décision + vérifier les hypothèses
    metrics_audit = {
        "selection": {
            "aic": float(sel.aic),
            "bic": float(sel.bic),
            "hqic": float(sel.hqic),
            "fpe": float(sel.fpe),
            "selected_lag_aic": int(p),
            "maxlags": int(maxlags),
        },
        "data": {
            "vars": cols,
            "nobs_raw": nobs_raw,
            "nobs_used_dropna": nobs_used,
            "rows_dropped_dropna": dropna_rows,
            "index_start": str(X.index.min()) if nobs_used else None,
            "index_end": str(X.index.max()) if nobs_used else None,
        },
        "diagnostics": {
            "stable": stable,
            "roots_abs_max": roots_abs_max,
            "roots": [complex(r).real for r in np.asarray(roots).ravel()] if roots is not None else None,
            "sigma_u": sigma_u,
            "whiteness_p": whiteness_p,
            "normality_p": normality_p,
        },
        "tests": {
            "granger_n": int(len(tbl_granger)) if isinstance(tbl_granger, pd.DataFrame) else None,
            "sims_n": int(len(tbl_sims)) if isinstance(tbl_sims, pd.DataFrame) else None,
            "sims_errors": int(sims_errors),
        },
    }

    note5 = (
        f"**Étape 5 — VAR(p)** : sélection AIC → p={p}, variables={cols}, nobs={int(res.nobs)}. "
        f"Stabilité (roots) : {stable} (max|root|={roots_abs_max}). "
        "IRF et FEVD décrivent la dynamique interne entre composantes STL. "
        "Granger/Sims sont reportés à titre descriptif (pas de causalité structurelle)."
    )

    return {
        "tables": {
            "tbl.var.lag_selection": tbl_sel,
            "tbl.var.granger": tbl_granger,
            "tbl.var.sims": tbl_sims,
            "tbl.var.fevd": tbl_fevd,
        },
        "metrics": {
            "m.var.meta": metrics_meta,
            "m.var.sims": sims_meta,
            "m.var.audit": metrics_audit,
            "m.note.step5": {"markdown": note5, "key_points": metrics_meta},
        },
        "models": {"model.var.best": res},
        "figures": {"fig.var.irf": fig_irf},
    }