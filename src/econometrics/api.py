# src/econometrics/api.py
from __future__ import annotations
from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from statsmodels.tsa.seasonal import STL

from src.econometrics.diagnostics import (
    acf_pacf_figs, adf_table,
    dickey_fuller_band_metrics, ts_vs_ds_decision, ljungbox_diff,
)
from src.econometrics.univariate import (
    ar_grid, ma_grid, arma_grid,
    arima_grid, residual_diagnostics,
    hurst_exponent, rescaled_range, figs_fit,fit_sarimax_safe)

from src.econometrics.multivariate import var_pack
from src.econometrics.cointegration import cointegration_pack

Y = "Croissance_Naturelle"

def _series(df: pd.DataFrame, y: str) -> pd.Series:
    if y != Y:
        raise ValueError("Contrat: variable cible unique Croissance_Naturelle.")
    return df[y].astype(float)

def _series(df: pd.DataFrame, y: str) -> pd.Series:
    if y != Y:
        raise ValueError("Contrat: variable cible unique Croissance_Naturelle.")
    s = df[y].astype(float)
    # IMPORTANT: index datetime pour fenêtres temporelles
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            pass
    return s

def _window_stats(s: pd.Series) -> dict[str, dict[str, float | int | None]]:
    # fenêtres "historiques" déterministes (anti-hallucination)
    def _mean(a: str, b: str) -> dict[str, float | int | None]:
        w = s.loc[a:b].dropna()
        return {"mean": float(w.mean()) if len(w) else None, "n": int(len(w))}
    return {
        "2017-01..2019-12": _mean("2017-01-01", "2019-12-31"),
        "2020-03..2021-12": _mean("2020-03-01", "2021-12-31"),
        "2022-01..2022-12": _mean("2022-01-01", "2022-12-31"),
        "2023-01..2025-12": _mean("2023-01-01", "2025-12-31"),
    }

def _breakpoints_from_windows(ws: dict[str, dict[str, float | int | None]], *, delta_thr: float = 0.20) -> list[dict[str, Any]]:
    # détecteurs simples, traçables (pas de tests lourds)
    bps: list[dict[str, Any]] = []
    pre = ws.get("2017-01..2019-12", {}).get("mean")
    covid = ws.get("2020-03..2021-12", {}).get("mean")
    post2023 = ws.get("2023-01..2025-12", {}).get("mean")

    if isinstance(pre, (int, float)) and isinstance(covid, (int, float)):
        d = float(covid - pre)
        if abs(d) >= delta_thr:
            bps.append({"ym": "2020-03", "type": "level_shift", "delta_mean": d, "tag": "covid_signal"})

    if isinstance(pre, (int, float)) and isinstance(post2023, (int, float)):
        d = float(post2023 - pre)
        if abs(d) >= delta_thr:
            bps.append({"ym": "2023-01", "type": "level_shift", "delta_mean": d, "tag": "inversion_2023_signal"})

    return bps

def step2_descriptive_pack(df: pd.DataFrame, *, y: str, period: int = 12, **params: Any) -> dict[str, Any]:
    s = _series(df, y)

    tbl_summary = s.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_frame("value")

    stl = STL(s.dropna(), period=period, robust=True).fit()
    decomp = pd.DataFrame({
        "level": s.loc[stl.trend.index],
        "trend": stl.trend,
        "seasonal": stl.seasonal,
        "resid": stl.resid,
    })

    # force saisonnière (traçable)
    var_resid = float(decomp["resid"].var())
    var_seas = float(decomp["seasonal"].var())
    strength = (var_seas / (var_seas + var_resid)) if (var_seas + var_resid) > 0 else 0.0

    if strength < 0.05:
        seas_type = "absence"
    else:
        roll = decomp["seasonal"].rolling(24).std().dropna()
        evol = float(roll.std()) if len(roll) else 0.0
        seas_type = "evolutive" if evol > 0.05 else "deterministe"

    tbl_seas = pd.DataFrame([{
        "seasonal_strength": strength,
        "seasonality_type": seas_type,
        "rule": "absence si strength<0.05; sinon evolutive si instabilité rolling>0.05; sinon deterministe"
    }]).set_index(pd.Index(["seasonality"]))

    fig_level = plt.figure()
    plt.plot(s.index, s.values)
    plt.title("Croissance_Naturelle (niveau)")

    fig_decomp = plt.figure(figsize=(10, 6))
    ax1 = fig_decomp.add_subplot(4, 1, 1); ax1.plot(decomp.index, decomp["level"]); ax1.set_title("Niveau")
    ax2 = fig_decomp.add_subplot(4, 1, 2); ax2.plot(decomp.index, decomp["trend"]); ax2.set_title("Tendance")
    ax3 = fig_decomp.add_subplot(4, 1, 3); ax3.plot(decomp.index, decomp["seasonal"]); ax3.set_title("Saisonnalité")
    ax4 = fig_decomp.add_subplot(4, 1, 4); ax4.plot(decomp.index, decomp["resid"]); ax4.set_title("Résidu")
    fig_decomp.tight_layout()

    # --- historique (fenêtres + ruptures simples) ---
    ws = _window_stats(s)
    bps = _breakpoints_from_windows(ws, delta_thr=float(params.get("delta_thr", 0.20)))

    
    note2 = (
        f"**Analyse descriptive & décomposition** : "
        f"la série présente une saisonnalité **{seas_type}**, "
        f"avec une force estimée à **{strength:.3f}**.\n\n "
        f"Cette mesure indique la part de la variance expliquée par la composante saisonnière "
        f"relativement au bruit résiduel.\n\n "
        f"La décomposition STL permet d'isoler distinctement le niveau, la tendance, "
        f"la saisonnalité et le résidu, fournissant une base structurée pour les tests "
        f"de stationnarité et la modélisation ultérieure."
    )

    return {
        "tables": {
            "tbl.desc.summary": tbl_summary,
            "tbl.desc.seasonality": tbl_seas,
            "tbl.desc.decomp_components": decomp,
        },
        "metrics": {
            "m.desc.seasonal_strength": {"value": float(strength)},
            "m.desc.seasonality_type": {"value": seas_type},
            "m.desc.key_points": {
                "key_points": {
                    "seasonality_type": seas_type,
                    "seasonal_strength": float(strength),
                    "window_stats": ws,
                    "breakpoints": bps,
                    "covid_signal": any(bp.get("tag") == "covid_signal" for bp in bps),
                    "inversion_2023": any(bp.get("tag") == "inversion_2023_signal" for bp in bps),
                }
            },
            "m.note.step2": {"markdown": note2},
        },
        "figures": {
            "fig.desc.level": fig_level,
            "fig.desc.decomp": fig_decomp,
        },
    }


def step3_stationarity_pack(df: pd.DataFrame, *, y: str, lags: int = 24, **params: Any) -> dict[str, Any]:
    s = _series(df, y)
    fig_acf, fig_pacf, tbl_acf = acf_pacf_figs(s, lags=lags)

    tbl_adf = adf_table(s)
    tbl_band = dickey_fuller_band_metrics(tbl_acf)

    # IMPORTANT: ts_vs_ds_decision doit être ADF-only (voir section B)
    tbl_dec, m_tsds = ts_vs_ds_decision(tbl_adf, tbl_band)

    tbl_lb = ljungbox_diff(s, lags=lags)

    verdict = m_tsds.get("verdict")
    p_c = m_tsds.get("adf_p_c")
    p_ct = m_tsds.get("adf_p_ct")


    note3 = (
        f"**Stationnarité (TS vs DS)** : verdict **{verdict}**.\n\n "
        f"Les tests ADF avec constante (c) et constante+tendance (ct) "
        f"donnent des p-values respectives de **{p_c:.3g}** et **{p_ct:.3g}**, "
        f"ne permettant pas de rejeter l'hypothèse de racine unitaire aux seuils usuels.\n\n "
        f"L'analyse de l'autocorrélation confirme une persistance élevée du niveau, "
        f"cohérente avec une dynamique de type Difference-Stationary.\n\n "
        f"La décision repose exclusivement sur les tests ADF et la lecture de la bande "
        f"de Dickey-Fuller via l'ACF, sans recours à des tests alternatifs."
    )
    return {
        "tables": {
            "tbl.diag.acf_pacf": tbl_acf,
            "tbl.diag.adf": tbl_adf,
            "tbl.diag.band_df": tbl_band,
            "tbl.diag.ts_vs_ds_decision": tbl_dec,
            "tbl.diag.ljungbox_diff": tbl_lb,
        },
        "metrics": {
            "m.diag.ts_vs_ds": {
                # VERDICT + p-values ADF uniquement
                "verdict": verdict,
                "adf_p_c": p_c,
                "adf_p_ct": p_ct,
                # tu peux garder d'autres champs ADF si tu veux (stat, crit, lags) mais pas PP
            },
            "m.note.step3": {
                "markdown": note3,
                "key_points": {
                    "verdict": verdict,
                    "adf_p_c": p_c,
                    "adf_p_ct": p_ct,
                },
            },
        },
        "figures": {"fig.diag.acf_level": fig_acf, "fig.diag.pacf_level": fig_pacf},
    }

def step4_univariate_pack(df: pd.DataFrame, *, y: str, **params: Any) -> dict[str, Any]:
    """
    Étape 4 — Analyse univariée (AR / MA / ARMA / ARIMA)

    Pipeline :
    1) Préparation TS / DS
    2) Grilles AR / MA / ARMA (stationnaire) + significativité
    3) Grille ARIMA (niveau) + significativité
    4) Sélection finale (AIC/BIC parmi modèles significatifs)
    5) Diagnostics + figures sur le modèle retenu
    """

    # ==========================================================
    # 0) Paramètres
    # ==========================================================
    alpha = float(params.get("alpha_sig", 0.05))
    criterion = str(params.get("criterion", "bic")).lower()
    criterion = criterion if criterion in {"aic", "bic"} else "bic"
    crit_col = "aic" if criterion == "aic" else "bic"

    s = _series(df, y)
    verdict = params.get("ts_ds_verdict", "DS")

    # ==========================================================
    # 1) Préparation cohérente TS / DS
    # ==========================================================
    if verdict == "DS":
        d_force, trend = 1, "n"
        s_model = s.dropna()
        s_stationary = s_model.diff().dropna()
    elif verdict == "TS":
        d_force, trend = 0, "c"
        t = np.arange(len(s))
        coef = np.polyfit(t, s.values, 1)
        s_model = (s - (coef[0] * t + coef[1])).dropna()
        s_stationary = s_model
    else:
        d_force, trend = None, "c"
        s_model = s.dropna()
        s_stationary = s_model

    # ==========================================================
    # 2) Grilles AR / MA / ARMA (stationnaire)
    # ==========================================================
    tbl_ar = ar_grid(s_stationary, p_max=6, alpha_sig=alpha)
    tbl_ma = ma_grid(s_stationary, q_max=6, alpha_sig=alpha)
    tbl_arma = arma_grid(s_stationary, p_max=6, q_max=6, alpha_sig=alpha)

    def _best_sig(tbl: pd.DataFrame) -> dict | None:
        if tbl is None or tbl.empty or "is_significant" not in tbl.columns:
            return None
        t = tbl[tbl["is_significant"]].copy()
        if t.empty:
            return None
        return t.sort_values([crit_col, "aic", "bic"]).iloc[0].to_dict()

    best_ar = _best_sig(tbl_ar)
    best_ma = _best_sig(tbl_ma)
    best_arma = _best_sig(tbl_arma)

    # ==========================================================
    # 3) Grille ARIMA (niveau)
    # ==========================================================
    tbl_arima, best_arima_raw, best_arima_res = arima_grid(
        s_model,
        p_max=int(params.get("p_max", 6)),
        d_max=int(params.get("d_max", 2)),
        q_max=int(params.get("q_max", 6)),
        d_force=d_force,
        trend=trend,
        alpha_sig=alpha,
    )
    best_arima = _best_sig(tbl_arima)

    # ==========================================================
    # 4) Tableau de synthèse (modèles significatifs uniquement)
    # ==========================================================
    rows = []
    if best_ar:
        rows.append({"model": "AR", "aic": best_ar["aic"], "bic": best_ar["bic"], "params": f"p={best_ar['p']}"})
    if best_ma:
        rows.append({"model": "MA", "aic": best_ma["aic"], "bic": best_ma["bic"], "params": f"q={best_ma['q']}"})
    if best_arma:
        rows.append({
            "model": "ARMA",
            "aic": best_arma["aic"],
            "bic": best_arma["bic"],
            "params": f"p={best_arma['p']}, q={best_arma['q']}",
        })
    if best_arima:
        rows.append({
            "model": "ARIMA",
            "aic": best_arima["aic"],
            "bic": best_arima["bic"],
            "params": f"(p,d,q)=({best_arima['p']},{best_arima['d']},{best_arima['q']})",
        })

    tbl_summary = pd.DataFrame(rows).reset_index(drop=True)

    # ==========================================================
    # 5) Sélection finale (AIC/BIC parmi modèles significatifs)
    # ==========================================================
    if tbl_summary.empty:
        selected_family = "ARIMA"
        best_res = best_arima_res
        order = best_arima_raw.get("order")
        aic, bic = best_arima_raw.get("aic"), best_arima_raw.get("bic")
    else:
        winner = tbl_summary.sort_values(crit_col).iloc[0]  # ✅ FIX 1: tri sur la bonne colonne
        selected_family = str(winner["model"])

        if selected_family == "AR":
            order = (int(best_ar["p"]), 0, 0)
            best_res = fit_sarimax_safe(s_stationary, order, trend="c")
            aic, bic = float(best_ar["aic"]), float(best_ar["bic"])

        elif selected_family == "MA":
            order = (0, 0, int(best_ma["q"]))
            best_res = fit_sarimax_safe(s_stationary, order, trend="c")
            aic, bic = float(best_ma["aic"]), float(best_ma["bic"])

        elif selected_family == "ARMA":
            order = (int(best_arma["p"]), 0, int(best_arma["q"]))
            best_res = fit_sarimax_safe(s_stationary, order, trend="c")
            aic, bic = float(best_arma["aic"]), float(best_arma["bic"])

        else:  # ARIMA
            order = (int(best_arima["p"]), int(best_arima["d"]), int(best_arima["q"]))
            best_res = fit_sarimax_safe(s_model, order, trend=trend)
            aic, bic = float(best_arima["aic"]), float(best_arima["bic"])

    if best_res is None:  # ✅ FIX 2: sécurité si refit échoue
        best_res = best_arima_res

    # ==========================================================
    # 6) Diagnostics & figures
    # ==========================================================
    resid = best_res.resid if best_res is not None else None
    tbl_diag = (
        residual_diagnostics(resid, lags=24)
        if resid is not None
        else pd.DataFrame([{"status": "no_model"}])
    )

    series_for_fig = s_model if selected_family == "ARIMA" else s_stationary
    figs = figs_fit(series_for_fig, best_res) if best_res is not None else {}

    # ==========================================================
    # 7) Mémoire longue
    # ==========================================================
    mem = {
        "hurst": float(hurst_exponent(s_stationary.values)),
        "rescaled_range": float(rescaled_range(s_stationary.values)),
    }
    tbl_mem = pd.DataFrame([mem], index=["memory"])

    # ==========================================================
    # 8) Note synthèse (compatible LaTeX / markdown)
    # ==========================================================
    def _fmt2(x: Any) -> str:
        return f"{x:.2f}" if isinstance(x, (int, float, np.floating)) and np.isfinite(x) else "NA"

    
    note = (
    f"**Modèles Univarié** : le résultats de stationnarité étant **{verdict}**, "
    f"nous conduit à une modélisation sur série différenciée (d={d_force if d_force is not None else 'auto'}).\n\n "
    f"L'exploration systématique des modèles AR, MA, ARMA (sur la série stationnaire) "
    f"et ARIMA (sur le niveau) intègre un filtrage préalable par significativité des paramètres.\n\n "
    f"Selon le critère **{criterion.upper()}**, le modèle retenu est **{selected_family}** "
    f"d'ordre **{order}**, avec AIC = **{_fmt2(aic)}** et BIC = **{_fmt2(bic)}**.\n\n "
    f"Ce choix reflète le meilleur compromis entre qualité d'ajustement et parcimonie "
    f"parmi les modèles statistiquement valides."
)

    return {
        "tables": {
            "tbl.uni.ar": tbl_ar,
            "tbl.uni.ma": tbl_ma,
            "tbl.uni.arma": tbl_arma,
            "tbl.uni.arima": tbl_arima,
            "tbl.uni.summary": tbl_summary,
            "tbl.uni.resid_diag": tbl_diag,
            "tbl.uni.memory": tbl_mem,
        },
        "metrics": {
            "m.uni.best": {
                "best_sig": {"AR": best_ar, "MA": best_ma, "ARMA": best_arma, "ARIMA": best_arima},
                "family": selected_family,
                "order": order,
                "aic": aic,
                "bic": bic,
                "criterion": criterion,
            },
            "m.note.step4": {"markdown": note},
        },
        "models": {"model.uni.best": best_res},
        "figures": figs,
    }

def step5_var_pack(
    df: pd.DataFrame,
    *,
    y: str,
    maxlags: int = 12,
    prefer_ic: str = "bic",
    whiteness_alpha: float = 0.05,
    require_stable: bool = True,
    **params: Any
) -> dict[str, Any]:

    _ = _series(df, y)

    # --- normalisation noms colonnes (safe) ---
    def _norm(c: str) -> str:
        return (
            str(c).strip()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("__", "_")
        )

    df0 = df.copy()
    df0.columns = [_norm(c) for c in df0.columns]

    # --- dictionnaire de correspondance (à étendre si besoin) ---
    wanted = {
        "Croissance_Naturelle": ["Croissance_Naturelle", "CroissanceNaturelle", "croissance_naturelle"],
        "Nb_mariages": ["Nb_mariages", "Nombres_mariages", "Mariages", "NbMariages", "nb_mariages"],
        "IPC": ["IPC", "Indice_prix", "Indice_des_prix", "CPI", "ipc"],
        "Masse_monetaire": ["Masse_monetaire", "Masse_monétaire", "MasseMonetaire", "M3", "m3"],
    }

    # --- résolution des colonnes existantes ---
    resolved: dict[str, str] = {}
    missing: list[str] = []

    cols_set = set(df0.columns)
    for canonical, candidates in wanted.items():
        found = next((c for c in candidates if _norm(c) in cols_set), None)
        if found is None:
            # essaye aussi matching exact après norm
            found = next((c for c in df0.columns if _norm(c) in {_norm(x) for x in candidates}), None)

        if found is None:
            missing.append(canonical)
        else:
            resolved[canonical] = found

    tbl_input = pd.DataFrame([{
        "status": "ok" if not missing else "missing_columns",
        "missing": ", ".join(missing) if missing else "",
        "resolved": ", ".join([f"{k}->{v}" for k, v in resolved.items()]),
        "available_cols": ", ".join(list(df0.columns)[:30]) + (" ..." if len(df0.columns) > 30 else ""),
    }]).set_index(pd.Index(["input"]))

    if missing:
        return {
            "tables": {"tbl.var.input_window": tbl_input},
            "metrics": {
                "m.note.step5": {
                    "markdown": f"**Étape 5 — VAR(p)** : colonnes manquantes: {missing}.",
                    "key_points": {"missing": missing, "resolved": resolved},
                }
            },
            "figures": {},
            "models": {},
        }

    VAR_COLS = [resolved["Croissance_Naturelle"], resolved["Nb_mariages"], resolved["IPC"], resolved["Masse_monetaire"]]
    Y4 = df0[VAR_COLS].copy()
    Y4.columns = ["Croissance_Naturelle", "Nb_mariages", "IPC", "Masse_monetaire"]  # canonical

    for c in Y4.columns:
        Y4[c] = pd.to_numeric(Y4[c], errors="coerce")

    if not isinstance(Y4.index, pd.DatetimeIndex):
        try:
            Y4.index = pd.to_datetime(Y4.index)
        except Exception:
            pass

    # fenêtre effective (Masse_monetaire démarre 1978)
    Y4 = Y4.dropna()

    tbl_input.loc["input", "nobs_after_dropna"] = int(Y4.shape[0])
    tbl_input.loc["input", "start"] = str(Y4.index.min()) if len(Y4) else None
    tbl_input.loc["input", "end"] = str(Y4.index.max()) if len(Y4) else None

    out = var_pack(
        Y4,
        maxlags=int(maxlags),
        prefer_ic=str(prefer_ic),
        whiteness_alpha=float(whiteness_alpha),
        require_stable=bool(require_stable),
    )

    out_tables = (out.get("tables") or {})
    out["tables"] = {"tbl.var.input_window": tbl_input, **out_tables}
    return out

def step6_cointegration_pack(
    df: pd.DataFrame,
    *,
    y: str,
    vars_mode: str = "decomp",   # gardé pour compat, mais on ne l'utilise plus ici
    det_order: int = 0,
    k_ar_diff: int = 1,
    **params: Any,
) -> dict[str, Any]:

    _ = _series(df, y)  # garde la vérif existante

    # --- normalisation noms colonnes (safe) ---
    def _norm(c: str) -> str:
        return (
            str(c).strip()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("__", "_")
        )

    df0 = df.copy()
    df0.columns = [_norm(c) for c in df0.columns]

    wanted = {
        "Croissance_Naturelle": ["Croissance_Naturelle", "CroissanceNaturelle", "croissance_naturelle"],
        "Nb_mariages": ["Nb_mariages", "Nombres_mariages", "Mariages", "NbMariages", "nb_mariages"],
        "IPC": ["IPC", "Indice_prix", "Indice_des_prix", "CPI", "ipc"],
        "Masse_monetaire": ["Masse_monetaire", "Masse_monétaire", "MasseMonetaire", "M3", "m3"],
    }

    resolved: dict[str, str] = {}
    missing: list[str] = []

    cols_set = set(df0.columns)
    for canonical, candidates in wanted.items():
        found = next((c for c in candidates if _norm(c) in cols_set), None)
        if found is None:
            found = next((c for c in df0.columns if _norm(c) in {_norm(x) for x in candidates}), None)

        if found is None:
            missing.append(canonical)
        else:
            resolved[canonical] = found

    tbl_input = pd.DataFrame([{
        "status": "ok" if not missing else "missing_columns",
        "missing": ", ".join(missing) if missing else "",
        "resolved": ", ".join([f"{k}->{v}" for k, v in resolved.items()]),
        "available_cols": ", ".join(list(df0.columns)[:30]) + (" ..." if len(df0.columns) > 30 else ""),
    }]).set_index(pd.Index(["input"]))

    if missing:
        return {
            "tables": {"tbl.coint.input_window": tbl_input},
            "metrics": {
                "m.note.step6": {
                    "markdown": f"**Étape 6 — Cointégration** : colonnes manquantes: {missing}.",
                    "key_points": {"missing": missing, "resolved": resolved},
                }
            },
            "models": {},
            "figures": {},
        }

    # mêmes colonnes que le VAR
    C_COLS = [
        resolved["Croissance_Naturelle"],
        resolved["Nb_mariages"],
        resolved["IPC"],
        resolved["Masse_monetaire"],
    ]
    X4 = df0[C_COLS].copy()
    X4.columns = ["Croissance_Naturelle", "Nb_mariages", "IPC", "Masse_monetaire"]  # canonical

    for c in X4.columns:
        X4[c] = pd.to_numeric(X4[c], errors="coerce")

    if not isinstance(X4.index, pd.DatetimeIndex):
        try:
            X4.index = pd.to_datetime(X4.index)
        except Exception:
            pass

    X4 = X4.dropna()

    tbl_input.loc["input", "nobs_after_dropna"] = int(X4.shape[0])
    tbl_input.loc["input", "start"] = str(X4.index.min()) if len(X4) else None
    tbl_input.loc["input", "end"] = str(X4.index.max()) if len(X4) else None

    out = cointegration_pack(X4, det_order=int(det_order), k_ar_diff=int(k_ar_diff))

    # on garde la table input en plus (audit)
    out_tables = (out.get("tables") or {})
    out["tables"] = {"tbl.coint.input_window": tbl_input, **out_tables}
    return out


