from __future__ import annotations
from typing import List
from src.narrative.sections.base import SectionSpec

def default_spec() -> List[SectionSpec]:
    return [
        SectionSpec(
            key="sec_data",
            title="Données, construction et qualité",
            intro_md=(
                "Cette section documente la construction de la variable d’intérêt $Y_t$ (solde naturel), "
                "la couverture temporelle, et les diagnostics de qualité (manquants, cohérence, métadonnées)."
            ),
            figure_keys=[],
            table_keys=["tbl.data.desc_stats", "tbl.data.missing_report", "tbl.data.coverage_report"],
            metric_keys=["m.data.dataset_meta", "m.note.step1"],
        ),
        SectionSpec(
            key="sec_descriptive",
            title="Analyse descriptive et décomposition",
            intro_md="On réalise une analyse descriptive et une décomposition STL pour isoler tendance, saisonnalité et résidu.",
            figure_keys=["fig.desc.level", "fig.desc.decomp"],
            table_keys=["tbl.desc.summary", "tbl.desc.seasonality", "tbl.desc.decomp_components"],
            metric_keys=["m.desc.seasonal_strength", "m.desc.seasonality_type", "m.note.step2"],
        ),
        SectionSpec(
            key="sec_stationarity",
            title="Diagnostics de stationnarité",
            intro_md=(
                "La spécification économétrique dépend de la stationnarité. "
                "La décision TS/DS est fondée exclusivement sur ADF (avec/sans tendance)."
            ),
            figure_keys=["fig.diag.acf_level", "fig.diag.pacf_level"],
            table_keys=["tbl.diag.adf", "tbl.diag.ts_vs_ds_decision", "tbl.diag.ljungbox_diff", "tbl.diag.band_df"],
            metric_keys=["m.diag.ts_vs_ds", "m.note.step3"],
        ),
        SectionSpec(
            key="sec_univariate",
            title="Modélisation univariée (ARIMA) et diagnostics",
            intro_md=(
                "On spécifie un modèle ARIMA selon Box–Jenkins. "
                "Le degré d’intégration $d$ est imposé par le verdict TS/DS (ADF-only)."
            ),
            figure_keys=["fig.uni.fit", "fig.uni.resid_acf", "fig.uni.qq"],
            table_keys=["tbl.uni.summary", "tbl.uni.arima", "tbl.uni.resid_diag", "tbl.uni.memory"],
            metric_keys=["m.uni.best", "m.note.step4", "m.diag.ts_vs_ds"],
        ),
        SectionSpec(
            key="sec_multivariate",
            title="Analyse multivariée (VAR)",
            intro_md="On estime un VAR sur les composantes issues de STL pour caractériser les dépendances dynamiques.",
            figure_keys=["fig.var.irf"],
            table_keys=["tbl.var.lag_selection", "tbl.var.granger", "tbl.var.sims", "tbl.var.fevd"],
            metric_keys=["m.var.meta", "m.var.sims", "m.var.audit", "m.note.step5"],
        ),
        SectionSpec(
            key="sec_cointegration",
            title="Cointégration et dynamique de long terme (VECM)",
            intro_md=(
                "On teste la cointégration (Engle–Granger, Johansen). "
                "Si une relation de long terme est détectée, on privilégie un VECM."
            ),
            figure_keys=[],
            table_keys=["tbl.coint.eg", "tbl.coint.johansen", "tbl.coint.var_vs_vecm_choice", "tbl.vecm.params"],
            metric_keys=["m.coint.meta", "m.coint.audit", "m.note.step6"],
        ),
        SectionSpec(
            key="sec_anthropology",
            title="Synthèse historique et anthropologique",
            intro_md="Cette section articule les résultats économétriques avec une lecture historique et anthropologique.",
            figure_keys=[],
            table_keys=[],
            metric_keys=["m.anthro.todd_analysis"],
        ),
    ]
