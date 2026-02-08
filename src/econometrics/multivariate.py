# src/econometrics/multivariate.py
from __future__ import annotations

from typing import Any, Optional

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
      - stat, pvalue
      - nobs_used
      - df_denom, df_num (si dispo)
    """
    df = X.copy()

    if caused not in df.columns or causing not in df.columns:
        return {"stat": float("nan"), "pvalue": float("nan"), "nobs_used": 0}

    y = df[caused]
    reg: dict[str, Any] = {}

    # Lags de toutes les variables
    for var in df.columns:
        for i in range(1, p + 1):
            reg[f"{var}_lag{i}"] = df[var].shift(i)

    # Leads de causing
    lead_names: list[str] = []
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
    if not lead_cols:
        return {"stat": float("nan"), "pvalue": float("nan"), "nobs_used": nobs_used}

    # R beta = 0 pour les leads
    R = []
    for col in lead_cols:
        r = [0.0] * X_clean.shape[1]
        r[X_clean.columns.get_loc(col)] = 1.0
        R.append(r)

    test = res_ols.f_test(R)

    out: dict[str, float | int | None] = {
        "stat": float(getattr(test, "fvalue", np.nan)),
        "pvalue": float(getattr(test, "pvalue", np.nan)),
        "nobs_used": nobs_used,
    }

    df_num = getattr(test, "df_num", None)
    df_denom = getattr(test, "df_denom", None)
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
      - IRF + FEVD
      - meta + audit payloads
    """
    X0 = df_vars.copy()
    cols = list(X0.columns)

    # ---- couverture / nettoyage ----
    X = X0.dropna().astype(float)
    nobs_raw = int(X0.shape[0])
    nobs_used_dropna = int(X.shape[0])
    dropna_rows = int(nobs_raw - nobs_used_dropna)

    # Protection minimale
    if X.shape[0] < 10 or X.shape[1] < 2:
        note5 = (
            "**Étape 5 — VAR** : données insuffisantes pour estimer un VAR multivarié "
            f"(nobs_used={X.shape[0]}, k={X.shape[1]})."
        )
        return {
            "tables": {
                "tbl.var.lag_selection": pd.DataFrame(
                    {
                        "aic": [np.nan],
                        "bic": [np.nan],
                        "hqic": [np.nan],
                        "fpe": [np.nan],
                        "maxlags": [int(maxlags)],
                        "selected_aic": [np.nan],
                        "nobs_raw": [nobs_raw],
                        "nobs_used": [nobs_used_dropna],
                        "rows_dropped_dropna": [dropna_rows],
                    },
                    index=["lag_selection"],
                ),
                "tbl.var.granger": pd.DataFrame([]),
                "tbl.var.sims": pd.DataFrame([]),
                "tbl.var.fevd": pd.DataFrame([]),
            },
            "metrics": {
                "m.var.meta": {
                    "vars": cols,
                    "k": int(len(cols)),
                    "selected_lag_aic": None,
                    "maxlags": int(maxlags),
                    "nobs_raw": nobs_raw,
                    "nobs_used": nobs_used_dropna,
                    "rows_dropped_dropna": dropna_rows,
                    "irf_h": None,
                    "fevd_h": None,
                },
                "m.var.sims": {
                    "lead_q_tested": [],
                    "method": "joint_f_test_on_leads (OLS eq-by-eq, Sims)",
                    "n_errors": 0,
                    "min_nobs_used": None,
                },
                "m.var.audit": {
                    "selection": None,
                    "data": {
                        "vars": cols,
                        "nobs_raw": nobs_raw,
                        "nobs_used_dropna": nobs_used_dropna,
                        "rows_dropped_dropna": dropna_rows,
                        "index_start": str(X.index.min()) if nobs_used_dropna else None,
                        "index_end": str(X.index.max()) if nobs_used_dropna else None,
                    },
                    "diagnostics": None,
                    "tests": None,
                },
                "m.note.step5": {"markdown": note5, "key_points": {}},
            },
            "models": {},
            "figures": {},
        }

    model = VAR(X)
    sel = model.select_order(maxlags=maxlags)

    # sélection AIC par défaut
    # (statsmodels peut renvoyer float/np.int64)
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
    granger_rows: list[dict[str, Any]] = []
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
                        "stat": _safe_float(getattr(test, "test_statistic", None)),
                        "pvalue": _safe_float(getattr(test, "pvalue", None)),
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

    tbl_granger = pd.DataFrame(granger_rows).replace([np.inf, -np.inf], np.nan)

    # Nettoyage df_* si totalement vides
    for c in ["df_num", "df_denom"]:
        if c in tbl_granger.columns and tbl_granger[c].isna().all():
            tbl_granger = tbl_granger.drop(columns=[c])
        elif c in tbl_granger.columns:
            tbl_granger[c] = pd.to_numeric(tbl_granger[c], errors="coerce")

    # -----------------------------
    # Sims (leads) : q=1..p
    # + audit erreurs / couverture
    # -----------------------------
    sims_rows: list[dict[str, Any]] = []
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
    try:
        fig_irf.suptitle("IRF (VAR)")
    except Exception:
        pass

    # -----------------------------
    # FEVD (horizon 12)
    # -----------------------------
    fevd_h = 12
    fevd = res.fevd(fevd_h)

    fevd_rows: list[dict[str, Any]] = []
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
    stable: Optional[bool] = None
    max_root_modulus: Optional[float] = None
    roots_modulus: Optional[list[float]] = None

    # Stabilité native (source de vérité si dispo)
    try:
        stable = bool(res.is_stable())
    except Exception:
        stable = None

    roots = getattr(res, "roots", None)
    if roots is not None:
        try:
            roots_arr = np.asarray(roots, dtype=complex).ravel()
            mod = np.abs(roots_arr)
            roots_modulus = [float(x) for x in mod.tolist()]
            max_root_modulus = float(np.max(mod)) if mod.size else None
            if stable is None and max_root_modulus is not None:
                stable = bool(max_root_modulus < 1.0)
        except Exception:
            max_root_modulus = None
            roots_modulus = None

    whiteness_pvalue: Optional[float] = None
    normality_pvalue: Optional[float] = None

    try:
        w = res.test_whiteness(nlags=min(12, max(1, p)))
        whiteness_pvalue = _safe_float(getattr(w, "pvalue", None))
    except Exception:
        whiteness_pvalue = None

    try:
        n = res.test_normality()
        normality_pvalue = _safe_float(getattr(n, "pvalue", None))
    except Exception:
        normality_pvalue = None

    sigma_u = None
    try:
        sigma_u = np.asarray(res.sigma_u, dtype=float).tolist()
    except Exception:
        sigma_u = None

    # -----------------------------
    # Metrics (meta + sims + audit)
    # -----------------------------
    metrics_meta = {
        "vars": cols,
        "k": int(len(cols)),
        "selected_lag_aic": int(p),
        "maxlags": int(maxlags),
        "nobs_raw": nobs_raw,
        "nobs_used": int(res.nobs),
        "rows_dropped_dropna": dropna_rows,
        "irf_h": int(irf_h),
        "fevd_h": int(fevd_h),
    }

    sims_meta = {
        "lead_q_tested": list(range(1, p + 1)),
        "method": "joint_f_test_on_leads (OLS eq-by-eq, Sims)",
        "n_errors": int(sims_errors),
        "min_nobs_used": (
            int(tbl_sims["nobs_used"].min())
            if "nobs_used" in tbl_sims.columns and not tbl_sims.empty
            else None
        ),
    }

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
            "nobs_used_dropna": nobs_used_dropna,
            "rows_dropped_dropna": dropna_rows,
            "index_start": str(X.index.min()) if nobs_used_dropna else None,
            "index_end": str(X.index.max()) if nobs_used_dropna else None,
        },
        "diagnostics": {
            "stable": stable,
            "max_root_modulus": max_root_modulus,
            "roots_modulus": roots_modulus,
            "sigma_u": sigma_u,
            "whiteness_pvalue": whiteness_pvalue,
            "normality_pvalue": normality_pvalue,
        },
        "tests": {
            "granger_n": int(len(tbl_granger)) if isinstance(tbl_granger, pd.DataFrame) else None,
            "sims_n": int(len(tbl_sims)) if isinstance(tbl_sims, pd.DataFrame) else None,
            "sims_errors": int(sims_errors),
        },
    }

    note5 = (
        f"**Étape 5 — VAR(p)** : sélection AIC → p={p}, variables={cols}, nobs={int(res.nobs)}. "
        f"Stabilité={stable} (max|root|={max_root_modulus}). "
        f"Whiteness p={whiteness_pvalue}, normalité p={normality_pvalue}. "
        "IRF/FEVD décrivent la dynamique interne; Granger/Sims = dépendance prédictive (pas causalité structurelle)."
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
