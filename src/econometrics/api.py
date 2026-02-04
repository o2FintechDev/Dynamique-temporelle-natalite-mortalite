# src/econometrics/api.py
from __future__ import annotations
from typing import Any
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import STL

from src.econometrics.diagnostics import (
    acf_pacf_figs, adf_table, phillips_perron_table,
    dickey_fuller_band_metrics, ts_vs_ds_decision, ljungbox_diff,
)
from src.econometrics.univariate import arima_grid, residual_diagnostics, hurst_exponent, rescaled_range, figs_fit
from src.econometrics.multivariate import var_pack
from src.econometrics.cointegration import cointegration_pack

Y = "Croissance_Naturelle"

def _series(df: pd.DataFrame, y: str) -> pd.Series:
    if y != Y:
        raise ValueError("Contrat: variable cible unique Croissance_Naturelle.")
    return df[y].astype(float)

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

    # force saisonnière simple
    var_resid = float(decomp["resid"].var())
    var_seas = float(decomp["seasonal"].var())
    strength = (var_seas / (var_seas + var_resid)) if (var_seas + var_resid) > 0 else 0.0

    # qualification (simple, traçable)
    if strength < 0.05:
        seas_type = "absence"
    else:
        # stabilité amplitude saisonnière (rolling std)
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

    note2 = (
    f"**Étape 2 — Analyse descriptive & décomposition** : tendance STL = composante dominante de long terme. "
    f"Saisonnalité qualifiée **{seas_type}** (force ≈ **{strength:.3f}**). "
    "Le résidu concentre la variabilité non expliquée par tendance/saisonnalité."
    )


    return {
        "tables": {
            "tbl.desc.summary": tbl_summary,
            "tbl.desc.seasonality": tbl_seas,
            "tbl.desc.decomp_components": decomp,
        },
        "metrics": {
            "m.desc.seasonal_strength": {"value": strength},
            "m.desc.seasonality_type": {"value": seas_type},
            "m.note.step2": {"markdown": note2},
        },

        "figures": {
            "fig.desc.level": fig_level,
            "fig.desc.decomp": fig_decomp,
        },
        "m.note.step2": {
        "markdown": note2,
        "key_points": {"seasonality_type": seas_type, "seasonal_strength": float(strength)}
        },
    }

def step3_stationarity_pack(df: pd.DataFrame, *, y: str, lags: int = 24, **params: Any) -> dict[str, Any]:
    s = _series(df, y)
    fig_acf, fig_pacf, tbl_acf = acf_pacf_figs(s, lags=lags)

    tbl_adf = adf_table(s)
    tbl_pp = phillips_perron_table(s)
    tbl_band = dickey_fuller_band_metrics(tbl_acf)
    tbl_dec, m_tsds = ts_vs_ds_decision(tbl_adf, tbl_pp, tbl_band)
    tbl_lb = ljungbox_diff(s, lags=lags)
    verdict = m_tsds.get("verdict")
    p_c = m_tsds.get("adf_p_c")
    p_ct = m_tsds.get("adf_p_ct")
    p_pp = m_tsds.get("pp_p")

    note3 = (
        f"**Étape 3 — Stationnarité (TS vs DS)** : verdict **{verdict}**. "
        f"ADF(c) p={p_c:.3g}, ADF(ct) p={p_ct:.3g}"
        + (f", PP p={p_pp:.3g}." if p_pp is not None else ", PP indisponible.")
        + " La décision est fondée sur ADF/PP et la lecture de persistance (bande DF via ACF)."
    )
    return {
        "tables": {
            "tbl.diag.acf_pacf": tbl_acf,
            "tbl.diag.adf": tbl_adf,
            "tbl.diag.pp": tbl_pp,
            "tbl.diag.band_df": tbl_band,
            "tbl.diag.ts_vs_ds_decision": tbl_dec,
            "tbl.diag.ljungbox_diff": tbl_lb,
        },
        "metrics": {
        "m.diag.ts_vs_ds": m_tsds,
        "m.note.step3": {
            "markdown": note3,
            "key_points": {
                "verdict": verdict,
                "adf_p_c": p_c,
                "adf_p_ct": p_ct,
                "pp_p": p_pp,
                },
            },
        },
        "figures": {"fig.diag.acf_level": fig_acf, "fig.diag.pacf_level": fig_pacf},
    }

def step4_univariate_pack(df: pd.DataFrame, *, y: str, **params: Any) -> dict[str, Any]:
    s = _series(df, y)

    grid, best, best_res = arima_grid(s)
    resid = best_res.resid if best_res is not None else None

    mem = {
        "rescaled_range": rescaled_range(s.values),
        "hurst": hurst_exponent(s.values),
        "arfima_status": "unavailable (not implemented in MVP)",
    }
    tbl_mem = pd.DataFrame([mem]).set_index(pd.Index(["memory"]))

    tbl_diag = residual_diagnostics(resid, lags=24) if resid is not None else pd.DataFrame([{"status": "no_model"}])
    figs = figs_fit(s, best_res) if best_res is not None else {}
    order = (best or {}).get("order")
    aic = (best or {}).get("aic")
    bic = (best or {}).get("bic")

    # diagnostics résidus (si dispo)
    lb_p = None
    arch_p = None
    jb_p = None
    if isinstance(tbl_diag, pd.DataFrame) and "ljungbox_p" in tbl_diag.columns:
        lb_p = float(tbl_diag.loc["diag", "ljungbox_p"])
        jb_p = float(tbl_diag.loc["diag", "jarque_bera_p"])
        arch_p = float(tbl_diag.loc["diag", "arch_p"])

    hurst = float(tbl_mem.loc["memory", "hurst"]) if "hurst" in tbl_mem.columns else None
    rs = float(tbl_mem.loc["memory", "rescaled_range"]) if "rescaled_range" in tbl_mem.columns else None

    note4 = (
        f"**Étape 4 — Univarié (ARIMA)** : modèle sélectionné ordre={order}, AIC={aic:.2f}, BIC={bic:.2f}. "
        + (f"Résidus: Ljung-Box p={lb_p:.3g}, JB p={jb_p:.3g}, ARCH p={arch_p:.3g}. " if lb_p is not None else "")
        + (f"Mémoire: Hurst≈{hurst:.3f}, R/S≈{rs:.3f}." if hurst is not None and rs is not None else "")
    )
    return {
        "tables": {"tbl.uni.selection": grid, "tbl.uni.resid_diag": tbl_diag, "tbl.uni.memory": tbl_mem},
        "metrics": {"m.uni.best": {"best": best},
                    "m.note.step4": {"markdown": note4,
                                     "key_points": {
                                         "order": order, "aic": aic, "bic": bic,
                                         "lb_p": lb_p, "jb_p": jb_p, "arch_p": arch_p,
                                         "hurst": hurst, "rescaled_range": rs
            }}},
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
