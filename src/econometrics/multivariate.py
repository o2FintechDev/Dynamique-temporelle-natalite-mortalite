# src/econometrics/multivariate.py
from __future__ import annotations

from typing import Any, Optional, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR


# -----------------------------
# Utils
# -----------------------------
def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def _coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _align_endog_exog(
    df_endog: pd.DataFrame, df_exog: Optional[pd.DataFrame]
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Alignement strict index endog/exog, coercition numérique.
    Ne dropna PAS ici (géré au niveau pack).
    """
    Y = _coerce_numeric_df(df_endog)
    if df_exog is None or df_exog.empty:
        return Y, None
    X = _coerce_numeric_df(df_exog)

    # intersection d'index, ordre endog
    idx = Y.index.intersection(X.index)
    Y = Y.loc[idx]
    X = X.loc[idx]
    return Y, X


# -----------------------------
# Corr Matrix + Heatmap
# -----------------------------
def corr_matrix(df_endog: pd.DataFrame, df_exog: pd.DataFrame | None = None) -> pd.DataFrame:
    mat = df_endog.copy()
    if df_exog is not None and not df_exog.empty:
        mat = pd.concat([mat, df_exog], axis=1)
    mat = mat.apply(pd.to_numeric, errors="coerce").dropna()
    if mat.empty:
        return pd.DataFrame([])
    return mat.corr()

def corr_heatmap_figure(corr: pd.DataFrame, title: str = "Heatmap corrélation (endog + exog)"):
    fig, ax = plt.subplots(figsize=(8, 6))
    if corr is None or corr.empty:
        ax.text(0.5, 0.5, "Corrélation indisponible (données vides après dropna)", ha="center", va="center")
        ax.axis("off")
        fig.suptitle(title)
        return fig

    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(corr.shape[0]))
    ax.set_yticks(range(corr.shape[0]))
    ax.set_xticklabels(list(corr.columns), rotation=45, ha="right")
    ax.set_yticklabels(list(corr.index))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    return fig

# -----------------------------
# Sims causality (unchanged)
# -----------------------------
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

    if nobs_used < (p + q + 5):
        return {"stat": float("nan"), "pvalue": float("nan"), "nobs_used": nobs_used}

    y_clean = data[caused]
    X_clean = sm.add_constant(data.drop(columns=[caused]), has_constant="add")
    res_ols = sm.OLS(y_clean, X_clean).fit()

    lead_cols = [c for c in X_clean.columns if c in lead_names]
    if not lead_cols:
        return {"stat": float("nan"), "pvalue": float("nan"), "nobs_used": nobs_used}

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


# -----------------------------
# Lag selection robust (VAR / VARX)
# -----------------------------
def _lag_grid(
    model: VAR,
    *,
    maxlags: int,
    p_min: int = 1,
    whiteness_nlags: Optional[int] = None,
) -> tuple[pd.DataFrame, dict[int, Any]]:
    """
    Fit p=1..maxlags, stocke IC + diagnostics (stabilité, whiteness).
    Retourne:
      - tbl_grid
      - fitted_results_by_p
    """
    rows: list[dict[str, Any]] = []
    fitted: dict[int, Any] = {}

    for p in range(p_min, maxlags + 1):
        try:
            res = model.fit(p)
            fitted[p] = res

            stable = None
            max_root_mod = None
            try:
                stable = bool(res.is_stable())
            except Exception:
                stable = None

            roots = getattr(res, "roots", None)
            if roots is not None:
                try:
                    r = np.asarray(roots, dtype=complex).ravel()
                    mod = np.abs(r)
                    max_root_mod = float(np.max(mod)) if mod.size else None
                    if stable is None and max_root_mod is not None:
                        stable = bool(max_root_mod < 1.0)
                except Exception:
                    max_root_mod = None

            w_p = None
            try:
                nl = whiteness_nlags
                if nl is None:
                    nl = min(12, max(1, p))
                w = res.test_whiteness(nlags=int(nl))
                w_p = _safe_float(getattr(w, "pvalue", None))
            except Exception:
                w_p = None

            rows.append(
                {
                    "p": int(p),
                    "aic": _safe_float(getattr(res, "aic", None)),
                    "bic": _safe_float(getattr(res, "bic", None)),
                    "hqic": _safe_float(getattr(res, "hqic", None)),
                    "fpe": _safe_float(getattr(res, "fpe", None)),
                    "nobs": int(getattr(res, "nobs", 0)),
                    "stable": stable,
                    "max_root_modulus": max_root_mod,
                    "whiteness_pvalue": w_p,
                }
            )
        except Exception as e:
            rows.append({"p": int(p), "error": type(e).__name__})
            continue

    tbl = pd.DataFrame(rows)
    return tbl, fitted


def _choose_p_from_grid(
    grid: pd.DataFrame,
    *,
    prefer: str = "bic",
    whiteness_alpha: float = 0.05,
    require_stable: bool = True,
) -> Optional[int]:
    """
    Politique ferme:
    - Filtre: stable==True si require_stable
    - Filtre: whiteness_pvalue > alpha si dispo
    - Sélection: min(prefer) sinon fallback min(aic)
    """
    if grid is None or grid.empty or "p" not in grid.columns:
        return None

    g = grid.copy()

    # garde uniquement lignes valides
    g = g[g["p"].notna()]

    if require_stable and "stable" in g.columns:
        g = g[g["stable"] == True]  # noqa: E712

    if "whiteness_pvalue" in g.columns:
        g_ok = g[g["whiteness_pvalue"].notna() & (g["whiteness_pvalue"] > float(whiteness_alpha))]
        if not g_ok.empty:
            g = g_ok  # sinon on garde g (dégradé)

    # critère
    crit = prefer.lower()
    if crit not in g.columns or g[crit].isna().all():
        crit = "aic"
    if crit not in g.columns or g[crit].isna().all():
        return None

    g2 = g[g[crit].notna()].sort_values(by=crit, ascending=True)
    if g2.empty:
        return None
    return int(g2.iloc[0]["p"])


# -----------------------------
# Main pack: VAR / VARX
# -----------------------------
def var_pack(
    df_vars: pd.DataFrame,
    maxlags: int = 12,
    *,
    df_exog: pd.DataFrame | None = None,
    prefer_ic: str = "bic",
    whiteness_alpha: float = 0.05,
    require_stable: bool = True,
) -> dict[str, Any]:
    """
    Pack VAR / VARX auditable.

    Corrige/renforce:
      - paramètres manquants (prefer_ic, whiteness_alpha, require_stable)
      - alignement strict + dropna conjoint endog/exog
      - robustesse statsmodels (VAR exog selon version)
      - corr + heatmap renvoyées dans tables/figures
      - fallback sélection p si grid/IC indisponible
    """
    # -----------------------------
    # 0) Coercition + alignement
    # -----------------------------
    Y0 = df_vars.copy()
    for c in Y0.columns:
        Y0[c] = pd.to_numeric(Y0[c], errors="coerce")

    X0 = None
    if df_exog is not None and not df_exog.empty:
        X0 = df_exog.copy()
        for c in X0.columns:
            X0[c] = pd.to_numeric(X0[c], errors="coerce")

        # align index intersection
        idx = Y0.index.intersection(X0.index)
        Y0 = Y0.loc[idx]
        X0 = X0.loc[idx]

    cols = list(Y0.columns)
    exog_cols = list(X0.columns) if X0 is not None else []

    # -----------------------------
    # 1) CORR (endog + exog) AVANT dropna (la fonction dropna elle-même)
    # -----------------------------
    corr = corr_matrix(Y0, df_exog=X0)
    fig_corr = corr_heatmap_figure(corr, title="Heatmap corrélation (endog + exog)")

    # -----------------------------
    # 2) Dropna conjoint (source of truth)
    # -----------------------------
    if X0 is None:
        nobs_raw = int(Y0.shape[0])
        Y = Y0.dropna().astype(float)
        X = None
    else:
        XY = pd.concat([Y0, X0], axis=1)
        nobs_raw = int(XY.shape[0])
        XY = XY.dropna().astype(float)
        Y = XY[cols]
        X = XY[exog_cols]

    nobs_used_dropna = int(Y.shape[0])
    dropna_rows = int(nobs_raw - nobs_used_dropna)

    # -----------------------------
    # 3) Protection minimale
    # -----------------------------
    if Y.shape[0] < 10 or Y.shape[1] < 2:
        note5 = (
            "**Étape 5 — VAR/VARX** : données insuffisantes pour estimer un VAR multivarié "
            f"(nobs_used={Y.shape[0]}, k={Y.shape[1]})."
        )
        return {
            "tables": {
                "tbl.var.corr": corr,
                "tbl.var.lag_selection": pd.DataFrame(
                    {
                        "aic": [np.nan],
                        "bic": [np.nan],
                        "hqic": [np.nan],
                        "fpe": [np.nan],
                        "maxlags": [int(maxlags)],
                        "selected_p": [np.nan],
                        "prefer_ic": [str(prefer_ic)],
                        "require_stable": [bool(require_stable)],
                        "whiteness_alpha": [float(whiteness_alpha)],
                        "has_exog": [bool(exog_cols)],
                        "nobs_raw": [nobs_raw],
                        "nobs_used": [nobs_used_dropna],
                        "rows_dropped_dropna": [dropna_rows],
                        "endog_vars": [", ".join(cols)],
                        "exog_vars": [", ".join(exog_cols) if exog_cols else ""],
                    },
                    index=["lag_selection"],
                ),
                "tbl.var.lag_grid": pd.DataFrame([]),
                "tbl.var.granger": pd.DataFrame([]),
                "tbl.var.sims": pd.DataFrame([]),
                "tbl.var.fevd": pd.DataFrame([]),
            },
            "metrics": {
                "m.var.meta": {
                    "vars": cols,
                    "exog": exog_cols,
                    "k": int(len(cols)),
                    "selected_p": None,
                    "prefer_ic": str(prefer_ic),
                    "maxlags": int(maxlags),
                    "nobs_raw": nobs_raw,
                    "nobs_used": nobs_used_dropna,
                    "rows_dropped_dropna": dropna_rows,
                    "irf_h": None,
                    "fevd_h": None,
                },
                "m.note.step5": {"markdown": note5, "key_points": {}},
            },
            "models": {},
            "figures": {
                "fig.var.corr_heatmap": fig_corr,
            },
        }

    # -----------------------------
    # 4) Build VAR / VARX (robuste selon version statsmodels)
    # -----------------------------
    has_exog = X is not None and not X.empty
    try:
        model = VAR(Y, exog=X) if has_exog else VAR(Y)
    except TypeError:
        # fallback si version statsmodels n'accepte pas exog dans VAR(...)
        # => on continue en VAR simple (contrôles exog ignorés)
        has_exog = False
        model = VAR(Y)

    # -----------------------------
    # 5) Lag grid + choix p robuste
    # -----------------------------
    grid, fitted = _lag_grid(model, maxlags=int(maxlags))
    p = _choose_p_from_grid(
        grid,
        prefer=str(prefer_ic),
        whiteness_alpha=float(whiteness_alpha),
        require_stable=bool(require_stable),
    )

    if p is None:
        # fallback: select_order
        try:
            sel = model.select_order(maxlags=int(maxlags))
            if str(prefer_ic).lower() == "bic":
                p = int(sel.bic)
            elif str(prefer_ic).lower() == "hqic":
                p = int(sel.hqic)
            else:
                p = int(sel.aic)
        except Exception:
            p = 1

    res = fitted.get(int(p))
    if res is None:
        res = model.fit(int(p))

    # -----------------------------
    # 6) Table sélection (résumé)
    # -----------------------------
    tbl_sel = pd.DataFrame(
        {
            "aic": [_safe_float(getattr(res, "aic", None))],
            "bic": [_safe_float(getattr(res, "bic", None))],
            "hqic": [_safe_float(getattr(res, "hqic", None))],
            "fpe": [_safe_float(getattr(res, "fpe", None))],
            "maxlags": [int(maxlags)],
            "selected_p": [int(p)],
            "prefer_ic": [str(prefer_ic)],
            "require_stable": [bool(require_stable)],
            "whiteness_alpha": [float(whiteness_alpha)],
            "has_exog": [bool(exog_cols) and has_exog],
            "nobs_raw": [nobs_raw],
            "nobs_used": [int(getattr(res, "nobs", Y.shape[0]))],
            "rows_dropped_dropna": [dropna_rows],
            "endog_vars": [", ".join(cols)],
            "exog_vars": [", ".join(exog_cols) if exog_cols else ""],
            "exog_used": [", ".join(exog_cols) if (bool(exog_cols) and has_exog) else ""],
        },
        index=["lag_selection"],
    )

    # -----------------------------
    # 7) Granger (pairwise) endog
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
                    {"caused": caused, "causing": causing, "stat": None, "pvalue": None, "error": type(e).__name__}
                )
    tbl_granger = pd.DataFrame(granger_rows).replace([np.inf, -np.inf], np.nan)
    for c in ["df_num", "df_denom"]:
        if c in tbl_granger.columns and tbl_granger[c].isna().all():
            tbl_granger = tbl_granger.drop(columns=[c])
        elif c in tbl_granger.columns:
            tbl_granger[c] = pd.to_numeric(tbl_granger[c], errors="coerce")

    # -----------------------------
    # 8) Sims (leads) q=1..p sur endog
    # -----------------------------
    sims_rows: list[dict[str, Any]] = []
    sims_errors = 0
    for caused in cols:
        for causing in cols:
            if caused == causing:
                continue
            for q in range(1, int(p) + 1):
                try:
                    t = sims_causality_test(Y, caused=caused, causing=causing, p=int(p), q=int(q))
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
    # 9) IRF figure
    # -----------------------------
    irf_h = 12
    irf = res.irf(irf_h)
    fig_irf = irf.plot(orth=False)
    try:
        fig_irf.suptitle("IRF (VAR/VARX)")
    except Exception:
        pass

    # -----------------------------
    # 10) FEVD (endog-only)
    # -----------------------------
    fevd_h = 12
    fevd = res.fevd(fevd_h)
    fevd_rows: list[dict[str, Any]] = []
    for i, target in enumerate(cols):
        m = fevd.decomp[:, i, :]
        for h in range(m.shape[0]):
            for j, shock in enumerate(cols):
                fevd_rows.append({"target": target, "shock": shock, "h": int(h), "share": float(m[h, j])})
    tbl_fevd = pd.DataFrame(fevd_rows)

    # -----------------------------
    # 11) Diagnostics
    # -----------------------------
    stable: Optional[bool] = None
    max_root_modulus: Optional[float] = None
    roots_modulus: Optional[list[float]] = None

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
        w = res.test_whiteness(nlags=min(12, max(1, int(p))))
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
    # 12) Metrics + note
    # -----------------------------
    metrics_meta = {
        "vars": cols,
        "exog": exog_cols,
        "k": int(len(cols)),
        "selected_p": int(p),
        "prefer_ic": str(prefer_ic),
        "maxlags": int(maxlags),
        "nobs_raw": nobs_raw,
        "nobs_used": int(getattr(res, "nobs", Y.shape[0])),
        "rows_dropped_dropna": dropna_rows,
        "irf_h": int(irf_h),
        "fevd_h": int(fevd_h),
        "require_stable": bool(require_stable),
        "whiteness_alpha": float(whiteness_alpha),
        "has_exog_requested": bool(exog_cols),
        "has_exog_used": bool(exog_cols) and has_exog,
    }

    sims_meta = {
        "lead_q_tested": list(range(1, int(p) + 1)),
        "method": "joint_f_test_on_leads (OLS eq-by-eq, Sims)",
        "n_errors": int(sims_errors),
        "min_nobs_used": (
            int(tbl_sims["nobs_used"].min())
            if "nobs_used" in tbl_sims.columns and not tbl_sims.empty
            else None
        ),
    }

    metrics_audit = {
        "data": {
            "endog_vars": cols,
            "exog_vars": exog_cols,
            "nobs_raw": nobs_raw,
            "nobs_used_dropna": nobs_used_dropna,
            "rows_dropped_dropna": dropna_rows,
            "index_start": str(Y.index.min()) if nobs_used_dropna else None,
            "index_end": str(Y.index.max()) if nobs_used_dropna else None,
        },
        "lag_grid": {
            "prefer_ic": str(prefer_ic),
            "require_stable": bool(require_stable),
            "whiteness_alpha": float(whiteness_alpha),
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
        "exog": {
            "requested": exog_cols,
            "used": exog_cols if has_exog else [],
            "note": "VAR exog support dépend de la version statsmodels; si non supporté, exog ignorées (fallback).",
        },
    }

    note5 = (
        f"**Étape 5 — VAR/VARX(p)** : choix p={p} (préférence={prefer_ic}, stable={require_stable}, "
        f"blancheur α={whiteness_alpha}). nobs={int(getattr(res, 'nobs', Y.shape[0]))}, "
        f"endog={cols}, exog={'∅' if not exog_cols else exog_cols}. "
        f"Stabilité={stable} (max|root|={max_root_modulus}). "
        f"Whiteness p={whiteness_pvalue}, normalité p={normality_pvalue}. "
        "FEVD/IRF décrivent la dynamique des endogènes; les exogènes sont des contrôles (pas de FEVD par exog)."
    )

    return {
        "tables": {
            "tbl.var.corr": corr,
            "tbl.var.lag_selection": tbl_sel,
            "tbl.var.lag_grid": grid,
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
        "figures": {
            "fig.var.corr_heatmap": fig_corr,
            "fig.var.irf": fig_irf,
        },
    }

def var_pack_dual(
    df_vars: pd.DataFrame,
    *,
    df_exog: pd.DataFrame | None = None,
    maxlags: int = 12,
    prefer_ic: str = "bic",
    whiteness_alpha: float = 0.05,
    require_stable: bool = True,
) -> dict[str, Any]:
    # 1) VAR
    out_var = var_pack(
        df_vars,
        maxlags=maxlags,
        df_exog=None,
        prefer_ic=prefer_ic,
        whiteness_alpha=whiteness_alpha,
        require_stable=require_stable,
    )

    # 2) VARX (si exog dispo)
    out_varx = None
    if df_exog is not None and not df_exog.empty:
        out_varx = var_pack(
            df_vars,
            maxlags=maxlags,
            df_exog=df_exog,
            prefer_ic=prefer_ic,
            whiteness_alpha=whiteness_alpha,
            require_stable=require_stable,
        )

    # 3) Merge avec préfixes pour éviter collisions
    def _prefix_pack(pack: dict[str, Any], prefix: str) -> dict[str, Any]:
        out: dict[str, Any] = {"tables": {}, "figures": {}, "metrics": {}, "models": {}}
        for k in ["tables", "figures", "metrics", "models"]:
            d = pack.get(k, {}) or {}
            out[k] = {f"{prefix}.{key}": val for key, val in d.items()}
        return out

    merged = _prefix_pack(out_var, "var")

    if out_varx is not None:
        p2 = _prefix_pack(out_varx, "varx")
        for k in ["tables", "figures", "metrics", "models"]:
            merged[k].update(p2[k])

        # comparatif simple
        try:
            sel_var = (out_var.get("tables", {}) or {}).get("tbl.var.lag_selection")
            sel_varx = (out_varx.get("tables", {}) or {}).get("tbl.var.lag_selection")
            if isinstance(sel_var, pd.DataFrame) and isinstance(sel_varx, pd.DataFrame) and not sel_var.empty and not sel_varx.empty:
                cmp = pd.DataFrame({
                    "model": ["VAR", "VARX"],
                    "selected_p": [int(sel_var["selected_p"].iloc[0]), int(sel_varx["selected_p"].iloc[0])],
                    "aic": [float(sel_var["aic"].iloc[0]), float(sel_varx["aic"].iloc[0])],
                    "bic": [float(sel_var["bic"].iloc[0]), float(sel_varx["bic"].iloc[0])],
                    "whiteness_pvalue": [
                        (out_var.get("metrics", {}) or {}).get("m.var.audit", {}).get("diagnostics", {}).get("whiteness_pvalue"),
                        (out_varx.get("metrics", {}) or {}).get("m.var.audit", {}).get("diagnostics", {}).get("whiteness_pvalue"),
                    ],
                })
                merged["tables"]["tbl.var.compare_var_varx"] = cmp
        except Exception:
            pass

    return merged
