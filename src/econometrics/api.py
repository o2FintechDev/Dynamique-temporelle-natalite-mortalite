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
    hurst_exponent, rescaled_range, figs_fit)

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
        f"**Étape 2 — Analyse descriptive & décomposition** : saisonnalité **{seas_type}** "
        f"(force ≈ **{strength:.3f}**). Décomposition STL produite (niveau/tendance/saisonnalité/résidu)."
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
        f"**Étape 3 — Stationnarité (TS vs DS)** : verdict **{verdict}**. "
        f"ADF(c) p={p_c:.3g}, ADF(ct) p={p_ct:.3g}. "
        "Décision fondée exclusivement sur ADF et la lecture de persistance (bande DF via ACF)."
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
    Étape 4 — Univarié (ARIMA)
    Règle:
      - verdict="DS"  => ARIMA avec d=1 sur le niveau (trend='n' par défaut), diagnostics et grilles AR/MA/ARMA sur Δs
      - verdict="TS"  => détrend linéaire, ARIMA avec d=0 sur série détrendée (trend='c'), grilles sur série détrendée
    """
    s = _series(df, y)
    verdict = params.get("ts_ds_verdict", "DS")

    # ----------------------------
    # 1) Préparation cohérente (ADF-only)
    # ----------------------------
    if verdict == "DS":
        d_force = 1
        trend = "n"
        s_model = s.dropna()
    elif verdict == "TS":
        d_force = 0
        trend = "c"
        t = np.arange(len(s))
        coef = np.polyfit(t, s.values, 1)
        trend_line = coef[0] * t + coef[1]
        s_model = (s - trend_line).dropna()
    else:
        d_force = None
        trend = "c"
        s_model = s.dropna()

    # Série stationnarisée utilisée pour AR/MA/ARMA + métriques mémoire
    s_for_arma = s_model.diff().dropna() if d_force == 1 else s_model

    # ----------------------------
    # 2) Grilles AR/MA/ARMA (stationnaire)
    # ----------------------------
    tbl_ar = ar_grid(s_for_arma, p_max=8)
    tbl_ma = ma_grid(s_for_arma, q_max=8)
    tbl_arma = arma_grid(s_for_arma, p_max=6, q_max=6)

    def _best_from_table(tbl: pd.DataFrame) -> dict | None:
        if tbl is None or tbl.empty:
            return None
        return tbl.iloc[0].to_dict()

    best_ar = _best_from_table(tbl_ar)
    best_ma = _best_from_table(tbl_ma)
    best_arma = _best_from_table(tbl_arma)

    # ----------------------------
    # 3) Grille ARIMA (cohérente avec d_force)
    # NOTE: nécessite arima_grid(..., d_force=..., trend=...)
    # ----------------------------
    grid, best, best_res = arima_grid(
        s_model,
        p_max=int(params.get("p_max", 4)),
        d_max=int(params.get("d_max", 2)),
        q_max=int(params.get("q_max", 4)),
        d_force=d_force,
        trend=trend,
    )

    resid = best_res.resid if best_res is not None else None

    # ----------------------------
    # 4) Diagnostics résiduels + figures (sur la même série que l'estimation)
    # ----------------------------
    tbl_diag = residual_diagnostics(resid, lags=24) if resid is not None else pd.DataFrame([{"status": "no_model"}])
    figs = figs_fit(s_model, best_res) if best_res is not None else {}

    # ----------------------------
    # 5) Mémoire (sur stationnarisée)
    # ----------------------------
    mem_series = s_for_arma.values
    mem = {
        "rescaled_range": float(rescaled_range(mem_series)),
        "hurst": float(hurst_exponent(mem_series)),
        "arfima_status": "non implémenté (hors périmètre de l'analyse)",
    }
    tbl_mem = pd.DataFrame([mem]).set_index(pd.Index(["memory"]))

    # ----------------------------
    # 6) Synthèse tableau
    # ----------------------------
    tbl_summary = pd.DataFrame([
        {
            "model": "AR",
            "best_aic": best_ar["aic"] if best_ar else None,
            "best_bic": best_ar["bic"] if best_ar else None,
            "params": f"p={best_ar['p']}" if best_ar else None,
        },
        {
            "model": "MA",
            "best_aic": best_ma["aic"] if best_ma else None,
            "best_bic": best_ma["bic"] if best_ma else None,
            "params": f"q={best_ma['q']}" if best_ma else None,
        },
        {
            "model": "ARMA",
            "best_aic": best_arma["aic"] if best_arma else None,
            "best_bic": best_arma["bic"] if best_arma else None,
            "params": f"p={best_arma['p']}, q={best_arma['q']}" if best_arma else None,
        },
        {
            "model": "ARIMA",
            "best_aic": (best or {}).get("aic"),
            "best_bic": (best or {}).get("bic"),
            "params": f"(p,d,q)={(best or {}).get('order')}",
        },
    ])

    order = (best or {}).get("order")
    aic = (best or {}).get("aic")
    bic = (best or {}).get("bic")

    # Diagnostics p-values
    lb_p = jb_p = arch_p = None
    if isinstance(tbl_diag, pd.DataFrame) and "ljungbox_p" in tbl_diag.columns and "diag" in tbl_diag.index:
        lb_p = float(tbl_diag.loc["diag", "ljungbox_p"])
        jb_p = float(tbl_diag.loc["diag", "jarque_bera_p"])
        arch_p = float(tbl_diag.loc["diag", "arch_p"])

    hurst = float(tbl_mem.loc["memory", "hurst"]) if "hurst" in tbl_mem.columns else None
    rs = float(tbl_mem.loc["memory", "rescaled_range"]) if "rescaled_range" in tbl_mem.columns else None

    def _fmt2(x: Any) -> str:
        return f"{x:.2f}" if isinstance(x, (int, float)) else "NA"

    note4 = (
        f"**Étape 4 — Univarié (ARIMA)** : verdict stationnarité={verdict} ⇒ d={d_force if d_force is not None else 'auto'}. "
        f"Modèle sélectionné ordre={order}, AIC={_fmt2(aic)}, BIC={_fmt2(bic)}. "
        + (f"Résidus: Ljung-Box p={lb_p:.3g}, JB p={jb_p:.3g}, ARCH p={arch_p:.3g}. " if lb_p is not None else "")
        + (f"Mémoire: Hurst≈{hurst:.3f}, R/S≈{rs:.3f}." if hurst is not None and rs is not None else "")
    )

    return {
        "tables": {
            "tbl.uni.ar": tbl_ar,
            "tbl.uni.ma": tbl_ma,
            "tbl.uni.arma": tbl_arma,
            "tbl.uni.arima": grid,
            "tbl.uni.summary": tbl_summary,
            "tbl.uni.resid_diag": tbl_diag,
            "tbl.uni.memory": tbl_mem,
        },
        "metrics": {
            "m.uni.best": {
                "best": best,
                "key_points": {
                    "verdict": verdict,
                    "d_force": d_force,
                    "trend": trend,
                    "best_ar": best_ar,
                    "best_ma": best_ma,
                    "best_arma": best_arma,
                    "order": order,
                    "aic": aic,
                    "bic": bic,
                    "lb_p": lb_p,
                    "jb_p": jb_p,
                    "arch_p": arch_p,
                    "hurst": hurst,
                    "rescaled_range": rs,
                },
            },
            "m.note.step4": {"markdown": note4},
        },
        "models": {"model.uni.best": best_res},
        "figures": figs,
    }

def step5_var_pack(df: pd.DataFrame, *, y: str, vars_mode: str = "decomp", **params: Any) -> dict[str, Any]:
    # Conformité: uniquement composantes issues de la décomposition du y
    s = _series(df, y)
    stl = STL(s.dropna(), period=12, robust=True).fit()
    X = pd.DataFrame({
        "level": s.loc[stl.trend.index],
        "trend": stl.trend,
        "seasonal": stl.seasonal,
    }).dropna()
    return var_pack(X, maxlags=12)

def step6_cointegration_pack(df: pd.DataFrame, *, y: str, vars_mode: str = "decomp", **params: Any) -> dict[str, Any]:
    s = _series(df, y)
    stl = STL(s.dropna(), period=12, robust=True).fit()
    X = pd.DataFrame({
        "level": s.loc[stl.trend.index],
        "trend": stl.trend,
        "seasonal": stl.seasonal,
    }).dropna()
    return cointegration_pack(X, det_order=0, k_ar_diff=1)
