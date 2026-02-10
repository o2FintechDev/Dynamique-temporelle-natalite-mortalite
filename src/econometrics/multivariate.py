# src/econometrics/multivariate.py
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

DISPLAY_NAME_MAP = {
    "Croissance_Naturelle" : "CN",
    "Masse_monetaire": "M3",
    "Nb_mariages": "Mariages",
}

# ============================================================
# Utils
# ============================================================
def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def _coerce_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index)
        except Exception:
            pass
    return out

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

def _apply_display_names_in_fig(fig) -> None:
    if fig is None:
        return

    # remplacements texte : "long" -> "court"
    rep = {k: v for k, v in DISPLAY_NAME_MAP.items() if isinstance(k, str) and isinstance(v, str)}

    for ax in getattr(fig, "axes", []):
        # titre
        t = ax.get_title()
        if t:
            for k, v in rep.items():
                t = t.replace(k, v)
            ax.set_title(t)

        # labels axes
        xl = ax.get_xlabel()
        yl = ax.get_ylabel()

        if xl:
            for k, v in rep.items():
                xl = xl.replace(k, v)
            ax.set_xlabel(xl)

        if yl:
            for k, v in rep.items():
                yl = yl.replace(k, v)
            ax.set_ylabel(yl)

        # légende
        leg = ax.get_legend()
        if leg:
            for txt in leg.get_texts():
                s = txt.get_text()
                for k, v in rep.items():
                    s = s.replace(k, v)
                txt.set_text(s)

# ============================================================
# Sims causality test (inchangé)
# ============================================================
def sims_causality_test(
    X: pd.DataFrame, *, caused: str, causing: str, p: int, q: int
) -> dict[str, float | int | None]:
    """
    Causalité à la Sims (anticipative):
    H0 : les leads de `causing` (t+1..t+q) n'expliquent pas `caused` (t)
    Test = Wald (F-test) sur coefficients des leads dans une OLS eq-by-eq,
    avec lags 1..p de toutes les variables comme contrôles.
    """
    df = X.copy()

    if caused not in df.columns or causing not in df.columns:
        return {"stat": float("nan"), "pvalue": float("nan"), "nobs_used": 0}

    y = df[caused]
    reg: dict[str, Any] = {}

    for var in df.columns:
        for i in range(1, p + 1):
            reg[f"{var}_lag{i}"] = df[var].shift(i)

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


# ============================================================
# Stationnarisation multi-variables (ADF + transformations)
#   -> VAR uniquement (pas de VARX)
# ============================================================
def _adf_one(s: pd.Series, *, regression: str = "c", autolag: str = "AIC") -> dict[str, Any]:
    x = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    if x.shape[0] < 15:
        return {"ok": False, "pvalue": np.nan, "stat": np.nan, "nobs": int(x.shape[0]), "reg": regression}
    try:
        stat, pval, usedlag, nobs, _, _ = sm.tsa.stattools.adfuller(x, regression=regression, autolag=autolag)
        return {"ok": True, "pvalue": float(pval), "stat": float(stat), "nobs": int(nobs), "usedlag": int(usedlag), "reg": regression}
    except Exception as e:
        return {"ok": False, "pvalue": np.nan, "stat": np.nan, "nobs": int(x.shape[0]), "reg": regression, "error": type(e).__name__}


def _safe_log_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return np.log(x.where(x > 0))


def _stationarize_one(
    s: pd.Series,
    *,
    alpha: float = 0.05,
    seasonal_period: int = 12,
    max_d: int = 2,
    allow_log: bool = True,
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Retour:
      - série transformée (dropna)
      - audit: p_level, p_final, transform
    Stratégie déterministe: level -> diff -> seasdiff -> seasdiff+diff -> log -> logdiff
    """
    s0 = pd.to_numeric(s, errors="coerce")

    r0 = _adf_one(s0, regression="c")
    p0 = float(r0.get("pvalue", np.nan))

    candidates: list[tuple[str, pd.Series]] = [("level", s0)]
    for d in range(1, int(max_d) + 1):
        candidates.append((f"diff(d={d})", s0.diff(d)))

    candidates.append((f"seasdiff(D=1,s={int(seasonal_period)})", s0.diff(int(seasonal_period))))
    candidates.append((f"seasdiff(D=1,s={int(seasonal_period)})+diff(d=1)", s0.diff(int(seasonal_period)).diff(1)))

    if allow_log:
        slog = _safe_log_series(s0)
        candidates.append(("log(level)", slog))
        candidates.append(("diff(d=1,on log)", slog.diff(1)))

    best = {"transform": "level", "p": p0, "series": s0}
    best_p = float("inf")
    chosen = None

    for tr, xs in candidates:
        rr = _adf_one(xs, regression="c")
        p = float(rr.get("pvalue", np.nan))

        if np.isfinite(p) and p < best_p:
            best_p = p
            best = {"transform": tr, "p": p, "series": xs}

        if np.isfinite(p) and p < float(alpha):
            chosen = {"transform": tr, "p": p, "series": xs}
            break

    out = (chosen or best)
    s_out = pd.to_numeric(out["series"], errors="coerce").dropna()

    audit = {
        "adf_p_level": None if not np.isfinite(p0) else float(p0),
        "adf_p_final": None if not np.isfinite(out["p"]) else float(out["p"]),
        "transform": str(out["transform"]),
        "nobs_final": int(s_out.shape[0]),
    }
    return s_out, audit


def stationarize_df(
    Y0: pd.DataFrame,
    *,
    alpha: float = 0.05,
    seasonal_period: int = 12,
    max_d: int = 2,
    allow_log: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Rend toutes les colonnes stationnaires + dropna conjoint.
    Retour:
      - Y_stationary (dropna conjoint)
      - tbl_stationarity (audit par variable)
      - meta (fenêtre effective)
    """
    Y = _coerce_df(Y0)
    cols = list(Y.columns)

    series_map: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []

    for c in cols:
        s_trans, aud = _stationarize_one(
            Y[c],
            alpha=alpha,
            seasonal_period=seasonal_period,
            max_d=max_d,
            allow_log=allow_log,
        )
        rows.append({"var": c, **aud})
        series_map[c] = s_trans

    Yt = pd.concat(series_map.values(), axis=1)
    Yt.columns = list(series_map.keys())
    Yt = Yt.dropna()

    meta = {
        "vars": cols,
        "alpha": float(alpha),
        "seasonal_period": int(seasonal_period),
        "max_d": int(max_d),
        "allow_log": bool(allow_log),
        "nobs_used": int(Yt.shape[0]),
        "start": str(Yt.index.min()) if not Yt.empty else None,
        "end": str(Yt.index.max()) if not Yt.empty else None,
    }
    return Yt, pd.DataFrame(rows), meta


# ============================================================
# Corr(Y ↔ X) : ici X = autres variables (endog) si besoin.
# Tu as demandé heatmap entre Y et X1..Xk, mais VAR-only => pas d'exog.
# Donc on fait une heatmap de corr entre les 4 endogènes (utile en multivarié).
# ============================================================
def corr_heatmap_figure_square(corr: pd.DataFrame, title: str = "Heatmap corrélation (variables VAR)"):
    fig, ax = plt.subplots(figsize=(7, 6))
    if corr is None or corr.empty:
        ax.text(0.5, 0.5, "Corrélation indisponible", ha="center", va="center")
        ax.axis("off")
        fig.suptitle(title)
        return fig

    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(corr.shape[1]))
    ax.set_yticks(range(corr.shape[0]))
    ax.set_xticklabels(list(corr.columns), rotation=45, ha="right")
    ax.set_yticklabels(list(corr.index))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


# ============================================================
# VAR pack (VAR only, K=4 typiquement)
# ============================================================
def var_pack(
    df_vars: pd.DataFrame,
    maxlags: int = 12,
    *,
    prefer_ic: str = "bic",
    whiteness_alpha: float = 0.05,
    require_stable: bool = True,
    p_cap: int = 5,
    alpha_stationarity: float = 0.05,
    seasonal_period: int = 12,
    max_d: int = 2,
    allow_log: bool = True,
) -> dict[str, Any]:
    """
    Pack VAR auditable (VAR(p) uniquement).

    Fait TOUT ce qui est VAR ici:
      - stationnarité des K variables + stationnarisation auto (ADF)
      - corr (heatmap) entre variables utilisées
      - choix p robuste (grid sur p<=p_cap, IC + stabilité + whiteness + significativité)
      - estimation VAR(p)
      - matrices A1..Ap + const
      - Granger / Sims
      - IRF + FEVD
    """
    # 0) Nettoyage
    Y0 = _coerce_df(df_vars)
    cols = list(Y0.columns)

    nobs_raw = int(Y0.shape[0])
    Y0 = Y0.dropna()
    nobs_used_dropna0 = int(Y0.shape[0])
    dropna_rows0 = int(nobs_raw - nobs_used_dropna0)

    # 1) Stationnarité + stationnarisation auto (obligatoire)
    Y, tbl_stat, m_stat = stationarize_df(
        Y0,
        alpha=alpha_stationarity,
        seasonal_period=seasonal_period,
        max_d=max_d,
        allow_log=allow_log,
    )

    # 2) Corr endog-endog (après stationnarisation)

    corr = Y.corr() if not Y.empty else pd.DataFrame([])
    corr_disp = _rename_df_display(corr)
    fig_corr = corr_heatmap_figure_square(corr_disp, title="Heatmap corrélation (variables VAR stationnaires)")

    # Protection minimale
    if Y.shape[0] < 30 or Y.shape[1] < 2:
        
        note5 = (
            f"**VAR(p)** : l'échantillon exploitable est insuffisant pour estimer un VAR multivarié.\n\n"
            f"Données disponibles : **nobs={int(Y.shape[0])}** observations, **k={int(Y.shape[1])}** variables.\n\n"
            "Conséquence : la sélection de l'ordre p et les diagnostics (stabilité, blancheur, normalité, IRF/FEVD) "
            "ne sont pas réalisés à cette étape."
        )
        return {
            "tables": {
                "tbl.var.stationarity": tbl_stat,
                "tbl.var.corr": corr,
                "tbl.var.lag_selection": pd.DataFrame(
                    {
                        "selected_p": [np.nan],
                        "prefer_ic": [str(prefer_ic)],
                        "maxlags": [int(maxlags)],
                        "p_cap": [int(p_cap)],
                        "require_stable": [bool(require_stable)],
                        "whiteness_alpha": [float(whiteness_alpha)],
                        "nobs_raw": [nobs_raw],
                        "nobs_after_dropna": [nobs_used_dropna0],
                        "rows_dropped_dropna": [dropna_rows0],
                        "nobs_stationary": [int(Y.shape[0])],
                        "start": [m_stat.get("start")],
                        "end": [m_stat.get("end")],
                    },
                    index=["lag_selection"],
                ),
                "tbl.var.lag_grid": pd.DataFrame([]),
                "tbl.var.granger": pd.DataFrame([]),
                "tbl.var.sims": pd.DataFrame([]),
                "tbl.var.fevd": pd.DataFrame([]),
            },
            "metrics": {
                "m.var.stationarity_meta": m_stat,
                "m.note.step5": {"markdown": note5, "key_points": {}},
            },
            "models": {},
            "figures": {"fig.var.corr_heatmap": fig_corr},
        }

    # 3) Fit grid p=1..min(maxlags,p_cap)
    pmax = int(min(maxlags, max(1, p_cap)))
    model = VAR(Y)

    grid_rows: list[dict[str, Any]] = []
    fitted: dict[int, Any] = {}

    for p in range(1, pmax + 1):
        try:
            res_p = model.fit(p)
            fitted[p] = res_p

            # stabilité
            try:
                stable = bool(res_p.is_stable())
            except Exception:
                stable = None

            # whiteness
            w_pv = None
            try:
                w = res_p.test_whiteness(nlags=min(12, max(1, p)))
                w_pv = _safe_float(getattr(w, "pvalue", None))
            except Exception:
                w_pv = None

            # significativité lag p (p-values sur paramètres Lp.*)
            share_sig_lp = None
            try:
                pv = res_p.pvalues  # DataFrame eq x params
                cols_lp = [c for c in pv.columns if c.startswith(f"L{p}.")]
                if cols_lp:
                    share_sig_lp = float((pv[cols_lp] < 0.05).values.mean())
            except Exception:
                share_sig_lp = None

            grid_rows.append(
                {
                    "p": int(p),
                    "aic": _safe_float(getattr(res_p, "aic", None)),
                    "bic": _safe_float(getattr(res_p, "bic", None)),
                    "hqic": _safe_float(getattr(res_p, "hqic", None)),
                    "fpe": _safe_float(getattr(res_p, "fpe", None)),
                    "stable": stable,
                    "whiteness_pvalue": w_pv,
                    "whiteness_ok": (w_pv is not None and w_pv >= float(whiteness_alpha)),
                    "share_sig_lag_p": share_sig_lp,
                }
            )
        except Exception as e:
            grid_rows.append({"p": int(p), "error": type(e).__name__})

    tbl_grid = pd.DataFrame(grid_rows).set_index("p") if grid_rows else pd.DataFrame([])

    # 4) Choix p (robuste)
    prefer = str(prefer_ic).lower()
    ic_col = "bic" if prefer == "bic" else ("hqic" if prefer == "hqic" else "aic")

    cand = tbl_grid.copy()
    if not cand.empty:
        if require_stable and "stable" in cand.columns:
            cand = cand[cand["stable"] == True]  # noqa: E712
        if "whiteness_ok" in cand.columns:
            cand = cand[cand["whiteness_ok"] == True]  # noqa: E712

    p_selected: Optional[int] = None
    if not cand.empty and ic_col in cand.columns:
        tmp = cand[ic_col].dropna()
        if not tmp.empty:
            p_selected = int(tmp.idxmin())

    # fallback: BIC/IC sans filtres
    if p_selected is None and (ic_col in tbl_grid.columns):
        tmp = tbl_grid[ic_col].dropna()
        if not tmp.empty:
            p_selected = int(tmp.idxmin())

    # fallback: 1
    if p_selected is None:
        p_selected = 1

    res = fitted.get(int(p_selected))
    if res is None:
        res = model.fit(int(p_selected))

    # 5) Matrices VAR: const + A1..Ap
    const_vec = None
    try:
        const_vec = pd.DataFrame({"const": np.asarray(res.intercept).ravel().tolist()}, index=res.names)
    except Exception:
        const_vec = pd.DataFrame([])

    A_mats: dict[str, pd.DataFrame] = {}
    try:
        coefs = np.asarray(res.coefs)  # (p,k,k)
        for i in range(coefs.shape[0]):
            A_mats[f"A{i+1}"] = pd.DataFrame(coefs[i], index=res.names, columns=res.names)
    except Exception:
        A_mats = {}

    # 6) Granger
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
                granger_rows.append({"caused": caused, "causing": causing, "stat": None, "pvalue": None, "error": type(e).__name__})
    tbl_granger = pd.DataFrame(granger_rows).replace([np.inf, -np.inf], np.nan)

    # 7) Sims leads (q=1..p)
    sims_rows: list[dict[str, Any]] = []
    sims_errors = 0
    for caused in cols:
        for causing in cols:
            if caused == causing:
                continue
            for q in range(1, int(p_selected) + 1):
                try:
                    t = sims_causality_test(Y, caused=caused, causing=causing, p=int(p_selected), q=int(q))
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
                    sims_rows.append({"caused": caused, "causing": causing, "lead_q": int(q), "stat": None, "pvalue": None, "error": type(e).__name__})
    tbl_sims = pd.DataFrame(sims_rows)

    # 8) IRF
    irf_h = 12
    irf = res.irf(irf_h)
    fig_irf = irf.plot(orth=False)

    _apply_display_names_in_fig(fig_irf)
    
    try:
        fig_irf.suptitle("IRF (VAR)")
    except Exception:
        pass

    # 9) FEVD
    fevd_h = 12
    fevd = res.fevd(fevd_h)
    fevd_rows: list[dict[str, Any]] = []
    for i, target in enumerate(cols):
        m = fevd.decomp[:, i, :]
        for h in range(m.shape[0]):
            for j, shock in enumerate(cols):
                fevd_rows.append({"target": target, "shock": shock, "h": int(h), "share": float(m[h, j])})
    tbl_fevd = pd.DataFrame(fevd_rows)

    # 10) Diagnostics
    stable = None
    max_root_modulus = None
    roots_modulus = None
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
        except Exception:
            max_root_modulus = None
            roots_modulus = None

    whiteness_pvalue = None
    normality_pvalue = None
    try:
        w = res.test_whiteness(nlags=min(12, max(1, int(p_selected))))
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

    # 11) Tables selection + pvalues + lag-significance
    pv_tbl = None
    lag_sig_tbl = None
    try:
        pv_tbl = res.pvalues.copy()
        lag_rows = []
        for L in range(1, int(p_selected) + 1):
            colsL = [c for c in pv_tbl.columns if c.startswith(f"L{L}.")]
            if colsL:
                lag_rows.append({"lag": L, "share_sig": float((pv_tbl[colsL] < 0.05).values.mean()), "n_params": int(pv_tbl[colsL].size)})
        lag_sig_tbl = pd.DataFrame(lag_rows)
    except Exception:
        pv_tbl = pd.DataFrame([])
        lag_sig_tbl = pd.DataFrame([])

    # 12) Notes + metrics
    tbl_sel = pd.DataFrame(
        {
            "selected_p": [int(p_selected)],
            "prefer_ic": [str(prefer_ic)],
            "ic_used": [ic_col],
            "maxlags": [int(maxlags)],
            "p_cap": [int(p_cap)],
            "require_stable": [bool(require_stable)],
            "whiteness_alpha": [float(whiteness_alpha)],
            "aic": [_safe_float(getattr(res, "aic", None))],
            "bic": [_safe_float(getattr(res, "bic", None))],
            "hqic": [_safe_float(getattr(res, "hqic", None))],
            "fpe": [_safe_float(getattr(res, "fpe", None))],
            "nobs_raw": [nobs_raw],
            "nobs_after_dropna": [nobs_used_dropna0],
            "rows_dropped_dropna": [dropna_rows0],
            "nobs_stationary": [int(Y.shape[0])],
            "start": [m_stat.get("start")],
            "end": [m_stat.get("end")],
            "vars": [", ".join(cols)],
        },
        index=["lag_selection"],
    )

    metrics_meta = {
        "vars": cols,
        "k": int(len(cols)),
        "selected_p": int(p_selected),
        "prefer_ic": str(prefer_ic),
        "ic_used": ic_col,
        "maxlags": int(maxlags),
        "p_cap": int(p_cap),
        "require_stable": bool(require_stable),
        "whiteness_alpha": float(whiteness_alpha),
        "nobs_stationary": int(Y.shape[0]),
        "stationarity": m_stat,
        "irf_h": int(irf_h),
        "fevd_h": int(fevd_h),
    }

    note5 = (
        f"**VAR(p)** : estimation d'un VAR sur les variables rendues stationnaires "
        f"(stationnarité via ADF, α={alpha_stationarity}).\n\n"

        f"La sélection de l'ordre retient **p={p_selected}** (IC utilisé : **{ic_col}**, "
        f"p_cap={p_cap}, contrainte de stabilité={require_stable}, test de blancheur des résidus au seuil α={whiteness_alpha}).\n\n"

        f"L'estimation est réalisée sur **{int(Y.shape[0])}** observations, "
        f"sur la période **{m_stat.get('start')} → {m_stat.get('end')}**.\n\n"

        f"Diagnostics : stabilité={stable}"
        + (f" (max|root|={max_root_modulus})." if max_root_modulus is not None else ".")
        + (f" Blancheur (whiteness) : p={whiteness_pvalue}." if whiteness_pvalue is not None else " Blancheur (whiteness) : p indisponible.")
        + (f" Normalité : p={normality_pvalue}." if normality_pvalue is not None else " Normalité : p indisponible.")
    )

    out_tables: dict[str, Any] = {
        "tbl.var.stationarity": tbl_stat,
        "tbl.var.stationary_data": Y,  # utile (audit) ; si trop lourd, tu peux l'enlever
        "tbl.var.corr": corr,
        "tbl.var.lag_grid": tbl_grid,
        "tbl.var.lag_selection": tbl_sel,
        "tbl.var.params_pvalues": pv_tbl if isinstance(pv_tbl, pd.DataFrame) else pd.DataFrame([]),
        "tbl.var.lag_significance": lag_sig_tbl if isinstance(lag_sig_tbl, pd.DataFrame) else pd.DataFrame([]),
        "tbl.var.const": const_vec,
        "tbl.var.fevd": tbl_fevd,
        "tbl.var.granger": tbl_granger,
        "tbl.var.sims": tbl_sims,
    }

    # Matrices A1..Ap (une table par matrice)
    for k, mat in A_mats.items():
        out_tables[f"tbl.var.{k}"] = mat

    # --- Display rename (tables)
    out_tables["tbl.var.stationarity"] = _rename_cols_in_df(out_tables["tbl.var.stationarity"], ["var"])
    out_tables["tbl.var.stationary_data"] = _rename_df_display(out_tables["tbl.var.stationary_data"])
    out_tables["tbl.var.corr"] = _rename_df_display(out_tables["tbl.var.corr"])
    out_tables["tbl.var.const"] = _rename_df_display(out_tables["tbl.var.const"])

    out_tables["tbl.var.granger"] = _rename_cols_in_df(out_tables["tbl.var.granger"], ["caused", "causing"])
    out_tables["tbl.var.sims"] = _rename_cols_in_df(out_tables["tbl.var.sims"], ["caused", "causing"])
    out_tables["tbl.var.fevd"] = _rename_cols_in_df(out_tables["tbl.var.fevd"], ["target", "shock"])

    out_tables["tbl.var.params_pvalues"] = _rename_df_display(out_tables["tbl.var.params_pvalues"])
    out_tables["tbl.var.lag_significance"] = out_tables["tbl.var.lag_significance"]  # pas de noms ici
    out_tables["tbl.var.lag_grid"] = out_tables["tbl.var.lag_grid"]
    out_tables["tbl.var.lag_selection"] = out_tables["tbl.var.lag_selection"]

    # Matrices A1..Ap
    for key in list(out_tables.keys()):
        if key.startswith("tbl.var.A"):
            out_tables[key] = _rename_df_display(out_tables[key])

    metrics_audit = {
        "data": {
            "vars": cols,
            "nobs_raw": nobs_raw,
            "nobs_after_dropna": nobs_used_dropna0,
            "rows_dropped_dropna": dropna_rows0,
            "nobs_stationary": int(Y.shape[0]),
            "start": m_stat.get("start"),
            "end": m_stat.get("end"),
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
        "selection_grid": {
            "ic_used": ic_col,
            "prefer_ic": str(prefer_ic),
            "p_cap": int(p_cap),
            "require_stable": bool(require_stable),
            "whiteness_alpha": float(whiteness_alpha),
        },
    }

    return {
        "tables": out_tables,
        "metrics": {
            "m.var.meta": metrics_meta,
            "m.var.stationarity_meta": m_stat,
            "m.var.audit": metrics_audit,
            "m.note.step5": {"markdown": note5, "key_points": metrics_meta},
        },
        "models": {"model.var.best": res},
        "figures": {
            "fig.var.corr_heatmap": fig_corr,
            "fig.var.irf": fig_irf,
        },
    }
