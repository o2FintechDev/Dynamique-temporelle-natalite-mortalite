# src/visualization/page_layouts.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class SectionSpec:
    title: str
    help_md: str = ""
    figures: Sequence[str] = ()
    tables: Sequence[str] = ()


PAGE_LAYOUTS: Dict[str, List[SectionSpec]] = {
    # ----------------
    # Page 1 — Exploration
    # ----------------
    "1_Exploration": [
        SectionSpec(
            title="Qualité & structure des données",
            help_md="Ces tableaux synthétisent la couverture temporelle, les valeurs manquantes et les statistiques globales.",
            tables=[
                "tbl.data.desc_stats",
                "tbl.data.missing_report",
                "tbl.data.coverage_report",
            ],
        ),
    ],
    # ======================================================
    # PAGE 2 — ANALYSE DESCRIPTIVE
    # ======================================================
    "2_Analyse_Descriptive": [
        SectionSpec(
            title="Visualisation de la série en niveau",
            help_md=(
                "Représentation graphique de la série de croissance naturelle. "
                "Cette étape permet d'identifier visuellement la présence de tendances, "
                "de cycles, de ruptures potentielles et de variations de volatilité."
            ),
            figures=[
                "fig.desc.level",
            ],
        ),
        SectionSpec(
            title="Décomposition de la série (STL)",
            help_md=(
                "La décomposition STL permet de séparer la série en composantes "
                "tendance, saisonnalité et résidu afin de mieux comprendre sa structure."
            ),
            figures=[
                "fig.desc.decomp",
            ],
        ),
        SectionSpec(
            title="Statistiques descriptives",
            help_md=(
                "Statistiques de base (niveau, dispersion, asymétrie) permettant de "
                "quantifier les caractéristiques observées graphiquement."
            ),
            tables=[
                "tbl.desc.summary",
            ],
        ),
        SectionSpec(
            title="Diagnostic de la saisonnalité",
            help_md=(
                "Analyse formelle de la saisonnalité issue de la décomposition STL. "
                "Cette étape est indispensable avant toute modélisation ARMA/ARIMA."
            ),
            tables=[
                "tbl.desc.seasonality",
            ],
        ),
        SectionSpec(
            title="Analyse des composantes de la décomposition",
            help_md=(
                "Analyse détaillée des composantes (tendance, saisonnalité, résidu) "
                "afin d'interpréter économiquement la dynamique de la série."
            ),
            tables=[
                "tbl.desc.decomp_components",
            ],
        ),
    ],

    # ======================================================
    # PAGE 3 — MODÈLES
    # ======================================================
    "3_Modeles": [

        # ----------------------
        # STEP 3 — Diagnostics & stationnarité
        # ----------------------
        SectionSpec(
            title="Analyse de l'autocorrélation (ACF / PACF)",
            help_md=(
                "Les fonctions d'autocorrélation (ACF) et d'autocorrélation partielle (PACF) "
                "permettent d'identifier la dépendance temporelle et d'orienter la spécification "
                "des modèles autorégressifs et à moyenne mobile."
            ),
            figures=[
                "fig.diag.acf_level",
                "fig.diag.pacf_level",
            ],
        ),
        SectionSpec(
            title="Tests de stationnarité et persistance",
            help_md=(
                "Tests formels de racine unitaire et indicateurs de persistance permettant "
                "de statuer sur la stationnarité de la série."
            ),
            tables=[
                "tbl.diag.acf_pacf",
                "tbl.diag.adf",
                "tbl.diag.band_df",
            ],
        ),
        SectionSpec(
            title="Décision TS ou DS et diagnostic résiduel",
            help_md=(
                "Synthèse des tests précédents pour déterminer si la série est trend-stationary "
                "ou difference-stationary, complétée par un test d'autocorrélation résiduelle."
            ),
            tables=[
                "tbl.diag.ts_vs_ds_decision",
                "tbl.diag.ljungbox_diff",
            ],
        ),

        # ----------------------
        # STEP 4 — Modélisation univariée
        # ----------------------
        
        SectionSpec(
            title="Estimation des modèles AR, MA, ARMA et ARIMA",
            help_md=(
                "Estimation successive des modèles univariés candidats afin de comparer "
                "leurs performances."
            ),
            tables=[
                "tbl.uni.ar",
                "tbl.uni.ma",
                "tbl.uni.arma",
                "tbl.uni.arima",
            ],
        ),
        SectionSpec(
            title="Sélection du modèle et diagnostics",
            help_md=(
                "Comparaison des modèles à l'aide des critères d'information (AIC/BIC) "
                "et diagnostics des résidus (autocorrélation, normalité, hétéroscédasticité, mémoire)."
            ),
            tables=[
                "tbl.uni.summary",
                "tbl.uni.resid_diag",
                "tbl.uni.memory",
            ],
        ),
        SectionSpec(
            title="Validation graphique des modèles univariés",
            help_md=(
                "Analyse graphique de l'ajustement du modèle retenu et des résidus "
                "afin de vérifier l'absence de structure non expliquée."
            ),
            figures=[
                "fig.uni.fit",
                "fig.uni.resid_acf",
                "fig.uni.qq",
            ],
        ),

        # ----------------------
        # STEP 5 — Modélisation VAR
        # ----------------------
        SectionSpec(
            title="Prétraitement multivarié & stationnarisation (VAR)",
            help_md=(
                "Pour la suite de l'analyse, de nouvelles variables ont été introduites.\n\n"
                "Afin de faciliter l'affichage, les variables sont renommées de la manière suivante :\n\n"
                "Croissance_Naturelle : CN - Masse_monétaire : M3 - Nb_mariages : mariages.\n\n"
                "Fenêtre d'estimation, contrôles de stationnarité et jeu de données stationnaire "
                "utilisé pour l'estimation du VAR. Cette étape conditionne la validité des tests "
                "de causalité et des IRF."
            ),
            tables=[
                "tbl.var.input_window",
                "tbl.var.stationarity",
                "tbl.var.stationary_data",
            ],
        ),
        SectionSpec(
            title="Corrélations & sélection des retards (VAR)",
            help_md=(
                "Analyse des dépendances contemporaines (corrélations) et sélection du nombre "
                "optimal de retards à l'aide des critères d'information. Les grilles/critères "
                "doivent converger vers un ordre parcimonieux."
            ),
            figures=[
                "fig.var.corr_heatmap",
            ],
            tables=[
                "tbl.var.corr",
                "tbl.var.lag_grid",
                "tbl.var.lag_selection",
            ],
        ),
        SectionSpec(
            title="Estimation, causalité et dynamique (VAR)",
            help_md=(
                "Paramètres estimés, significativité, causalité (Granger + Sims), décomposition "
                "de variance (FEVD) et réponses impulsionnelles (IRF). Les matrices A1..A5 "
                "détaillent la structure dynamique par retard."
            ),
            figures=[
                "fig.var.irf",
            ],
            tables=[
                "tbl.var.params_pvalues",
                "tbl.var.lag_significance",
                "tbl.var.const",
                "tbl.var.granger",
                "tbl.var.sims",
                "tbl.var.fevd",
                "tbl.var.A1",
                "tbl.var.A2",
                "tbl.var.A3",
                "tbl.var.A4",
                "tbl.var.A5",
            ],
        ),
        
    ],

    # ======================================================
    # PAGE 4 — RÉSULTATS : COINTÉGRATION & VECM
    # ======================================================
    "4_Resultats": [
        SectionSpec(
            title="Tests de cointégration",
            help_md=(
                "Tests de cointégration d'Engle-Granger et de Johansen permettant "
                "d'identifier l'existence de relations de long terme entre les variables."
            ),
            tables=[
                "tbl.coint.eg",
                "tbl.coint.johansen",
            ],
        ),
        SectionSpec(
            title="Choix entre VAR en différences et VECM",
            help_md=(
                "Application d'une règle de décision explicite pour déterminer "
                "la spécification appropriée du modèle multivarié."
            ),
            tables=[
                "tbl.coint.var_vs_vecm_choice",
            ],
        ),
        SectionSpec(
            title="Estimation du VECM (si applicable)",
            help_md=(
                "Présentation des paramètres du VECM : vecteurs de cointégration "
                "et vitesses d'ajustement vers l'équilibre de long terme."
            ),
            tables=[
                "tbl.vecm.params",
            ],
        ),
    ],
}