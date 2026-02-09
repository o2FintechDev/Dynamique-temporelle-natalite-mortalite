# src/visualization/ui_labels.py
from __future__ import annotations

import re

UI_LABELS: dict[str, str] = {
    # --- Page 1 : Exploration ---
    "tbl.data.desc_stats": "Statistiques descriptives des variables",
    "tbl.data.missing_report": "Analyse des valeurs manquantes",
    "tbl.data.coverage_report": "Couverture temporelle des données",

    "m.data.dataset_meta": "Métadonnées du jeu de données",
    "m.note.step1": "Note d'interprétation (Étape 1)",

    # --- Page 2 : Analyse descriptive ---
    "tbl.desc.summary": "Statistiques descriptives (série cible)",
    "tbl.desc.seasonality": "Diagnostic de saisonnalité (STL)",
    "tbl.desc.decomp_components": "Composantes STL (niveau, tendance, saisonnalité, résidu)",

    "fig.desc.level": "Évolution de la série Croissance Naturelle",
    "fig.desc.decomp": "Décomposition STL",

    "m.desc.seasonal_strength": "Force de la saisonnalité",
    "m.desc.seasonality_type": "Type de saisonnalité",
    "m.desc.key_points": "Points clés (détection & signaux)",
    "m.note.step2": "Note d'interprétation (Étape 2)",

        # --- Page 3 : Modèles — Step 3 (Diagnostics / Stationnarité) ---
    "tbl.diag.acf_pacf": "Fonction d'autocorrélation & Fonction d'autocorrélation partielle",
    "tbl.diag.adf": "Test de Dickey-Fuller augmenté (racine unitaire)",
    "tbl.diag.band_df": "Indicateurs de persistance (bande DF via ACF)",
    "tbl.diag.ts_vs_ds_decision": "Décision de processus TS ou DS",
    "tbl.diag.ljungbox_diff": "Test de Ljung-Box (autocorrélation résiduelle)",

    "fig.diag.acf_level": "Fonction d'autocorrélation",
    "fig.diag.pacf_level": "Fonction d'autocorrélation partielle",

    "m.diag.ts_vs_ds": "Résumé TS vs DS",
    "m.note.step3": "Note d'interprétation (Étape 3)",

    # --- Page 3 : Modèles — Step 4 (Univarié) ---
    "tbl.uni.ar": "Modèle AR ",
    "tbl.uni.ma": "Modèle MA ",
    "tbl.uni.arma": "Modèle ARMA ",
    "tbl.uni.arima": "Modèle ARIMA ",
    "tbl.uni.summary": "Synthèse des modèles (comparaison AIC/BIC)",
    "tbl.uni.resid_diag": "Diagnostics des résidus (Ljung-Box, JB, ARCH)",
    "tbl.uni.memory": "Indicateurs de mémoire (Hurst, R/S)",

    "fig.uni.fit": "Ajustement du modèle",
    "fig.uni.resid_acf": "Fonction d'autocorrélation des résidus",
    "fig.uni.qq": "QQ-plot des résidus (normalité)",


    "m.uni.best": "Résumé du modèle retenu (Étape 4)",
    "m.note.step4": "Note d'interprétation (Étape 4)",

    # --- Page 3 : Modèles — Step 5 (VAR) ---
    "tbl.var.lag_selection": "Sélection du lag du VAR (AIC/BIC/HQIC/FPE)",
    "tbl.var.granger": "Causalité de Granger",
    "tbl.var.sims": "Causalité de Sims",
    "tbl.var.fevd": "Décomposition de la variance (FEVD)",

    "fig.var.irf": "Fonctions de réponse impulsionnelle (IRF)",

    "m.var.meta": "Résumé du VAR",
    "m.var.sims": "Paramètres et couverture des tests de Sims",
    "m.var.audit": "Audit de reproductibilité (VAR)",
    "m.note.step5": "Note d'interprétation (Étape 5)",

    "model.var.best": "Modèle VAR estimé (résultats)",

        # --- Page 4 : Résultats — Step 6 (Cointégration) ---
    "tbl.coint.eg": "Tests Engle-Granger",
    "tbl.coint.johansen": "Test de Johansen — détermination du rang",
    "tbl.coint.var_vs_vecm_choice": "Décision : VAR en différences ou VECM (règle explicite)",

    "m.coint.meta": "Résumé cointégration ",
    "m.coint.audit": "Audit détaillé cointégration",
    "m.note.step6": "Note d'interprétation (Étape 6)",

    # Si VECM est estimé (rank > 0)
    "tbl.vecm.params": "Paramètres VECM (α vitesse d'ajustement, β relation de long terme)",
    "m.vecm.meta": "Résumé VECM",
    "model.vecm": "Modèle VECM estimé (résultats)",

}

def pretty_label(key: str) -> str:
    """
    Convertit une clé technique (stable pour LaTeX) en titre lisible pour l'UI Streamlit.
    - mapping explicite si connu
    - fallback automatique sinon
    """
    key = (key or "").strip()
    if not key:
        return "Élément"

    if key in UI_LABELS:
        return UI_LABELS[key]

    # fallback: "tbl.data.desc_stats" -> "Data › Desc stats"
    s = re.sub(r"^(tbl|fig|m)\.", "", key)
    s = s.replace(".", " › ").replace("_", " ")
    return s[:1].upper() + s[1:]
