# src/narrative/sections/sec_multivariate.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.narrative.sections.base import (
    SectionSpec,
    lookup,
    md_basic_to_tex,
    include_table_tex,
    include_figure,
    narr_call,
)


def _fmt2(x: Any) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "NA"


def _fmt_p(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return "NA"
    if v < 1e-4:
        return f"{v:.2e}"
    return f"{v:.3f}"


def _yn(x: Any) -> str:
    if x is True:
        return "Oui"
    if x is False:
        return "Non"
    return "NA"


def render_sec_multivariate(
    *,
    run_root: Path,
    manifest: Dict[str, Any],
    sec: SectionSpec,
    metrics_cache: Dict[str, Dict[str, Any]],
) -> str:
    """
    Step5 = VAR sur composantes (STL) ou variables sélectionnées.
    Objectif: dépendances dynamiques (prévisionnelles), IRF, FEVD.
    IMPORTANT:
      - Granger/Sims = dépendance prédictive, pas causalité structurelle.
      - IRF/FEVD interprétables uniquement si VAR stable.
    """

    # ---------------------------
    # Metrics
    # ---------------------------
    m_var = metrics_cache.get("m.var.meta") or {}
    m_sims = metrics_cache.get("m.var.sims") or {}
    m_audit = metrics_cache.get("m.var.audit") or {}

    diag = (m_audit.get("diagnostics") or {}) if isinstance(m_audit, dict) else {}
    sel = (m_audit.get("selection") or {}) if isinstance(m_audit, dict) else {}
    data_a = (m_audit.get("data") or {}) if isinstance(m_audit, dict) else {}

    # meta (robust keys)
    vars_ = m_var.get("vars") or data_a.get("vars") or []
    if not isinstance(vars_, list):
        vars_ = []

    k = m_var.get("k") or m_var.get("n_vars") or (len(vars_) if vars_ else "NA")
    p = (
        m_var.get("selected_lag_aic")
        or m_var.get("p")
        or m_var.get("lag_order")
        or m_var.get("selected_lag")
        or sel.get("selected_lag_aic")
        or "NA"
    )
    nobs = m_var.get("nobs_used") or m_var.get("nobs") or data_a.get("nobs_used_dropna") or "NA"

    stable = diag.get("stable")
    max_root = diag.get("max_root_modulus")
    whiteness_p = diag.get("whiteness_pvalue")
    normality_p = diag.get("normality_pvalue")

    stable_txt = _yn(stable)

    sims_q = m_sims.get("lead_q_tested")
    sims_err = m_sims.get("n_errors")
    sims_min_n = m_sims.get("min_nobs_used")

    # note Step5
    note = metrics_cache.get("m.note.step5") or {}
    note_md = ""
    if isinstance(note, dict):
        note_md = note.get("markdown") or note.get("text") or note.get("summary") or ""
    elif isinstance(note, str):
        note_md = note
    note_md = (note_md or "").replace("−", "-")

    # ---------------------------
    # Artefacts
    # ---------------------------
    # Tables (core)
    tbl_lag = lookup(manifest, "tables", "tbl.var.lag_selection")
    tbl_granger = lookup(manifest, "tables", "tbl.var.granger")
    tbl_sims = lookup(manifest, "tables", "tbl.var.sims")
    tbl_fevd = lookup(manifest, "tables", "tbl.var.fevd")
    tbl_input = lookup(manifest, "tables", "tbl.var.input_window")
    tbl_stat  = lookup(manifest, "tables", "tbl.var.stationarity")
    tbl_corr  = lookup(manifest, "tables", "tbl.var.corr")
    tbl_pvals = lookup(manifest, "tables", "tbl.var.params_pvalues")

    # Tables (annexes)
    tbl_lag_grid = lookup(manifest, "tables", "tbl.var.lag_grid")
    tbl_const    = lookup(manifest, "tables", "tbl.var.const")
    tbl_A1 = lookup(manifest, "tables", "tbl.var.A1")
    tbl_A2 = lookup(manifest, "tables", "tbl.var.A2")
    tbl_A3 = lookup(manifest, "tables", "tbl.var.A3")
    tbl_A4 = lookup(manifest, "tables", "tbl.var.A4")
    tbl_A5 = lookup(manifest, "tables", "tbl.var.A5")
    tbl_stationary_data = lookup(manifest, "tables", "tbl.var.stationary_data")

    # Figures
    fig_corr = lookup(manifest, "figures", "fig.var.corr_heatmap")       
    fig_irf = lookup(manifest, "figures", "fig.var.irf")
    lines: list[str] = []

    # ============================================================
    # SECTION: Cadre VAR (théorie + conditions)
    # ============================================================
    lines += [
        r"\section{Analyse multivariée : modèles VAR et dépendances dynamiques}",
        "",
        md_basic_to_tex(
            "L’analyse univariée décrit la dynamique propre de la série cible, mais ne permet pas de caractériser "
            "les interactions internes entre composantes (tendance, saisonnalité, résidu) ou variables liées. "
            "Le modèle VAR (Vector AutoRegression) fournit un cadre standard pour capturer ces dépendances dynamiques "
            "sans imposer de restrictions structurelles a priori."
        ),
        "",
        r"\subsection*{5.1 Spécification du modèle VAR}",
        "",
        r"\begin{equation}",
        r"Y_t = c + A_1 Y_{t-1} + A_2 Y_{t-2} + \cdots + A_p Y_{t-p} + \varepsilon_t",
        r"\end{equation}",
        "",
        md_basic_to_tex(
            "où $Y_t$ est un vecteur de dimension $k$ et $p$ l’ordre du VAR. "
            "Les coefficients sont rarement interprétés individuellement ; l’analyse se concentre sur : "
            "(i) causalité de Granger, (ii) réponses impulsionnelles (IRF), (iii) décomposition de variance (FEVD)."
        ),
        "",
        r"\subsection*{5.2 Conditions de validité et diagnostics}",
        md_basic_to_tex(
            "Un VAR exploitable économétriquement doit satisfaire au minimum : "
            "(i) stabilité (racines strictement à l’intérieur du cercle unité), "
            "(ii) résidus raisonnablement non autocorrélés (whiteness), "
            "(iii) cohérence de l’ordre $p$ (arbitrage biais–variance). "
            "Sans stabilité, IRF/FEVD ne sont pas interprétables."
        ),
        "",
        r"\subsection*{5.3 Statut des tests de causalité}",
        md_basic_to_tex(
            "Causalité de Granger : dépendance prédictive (gain de prévision). "
            "Causalité à la Sims (leads) : test de robustesse anticipatif ; un signal sur des valeurs futures peut révéler "
            "une mauvaise spécification (retards insuffisants, variables mal choisies, transformations inadéquates) "
            "ou des effets de calendrier/mesure. Aucun de ces tests n’établit une causalité structurelle."
        ),
        "",
    ]

    # ============================================================
    # SECTION 2 : Résultats Step5 (synthèse + artefacts)
    # ============================================================
    vars_txt = ", ".join([str(v) for v in vars_]) if vars_ else "NA"

    lines += [
        r"\section{Résultats empiriques}",
        "",
        md_basic_to_tex(
            f"Synthèse : variables={vars_txt}, dimension $k$={k}, ordre $p$={p}, n={nobs}. "
            f"Stabilité={stable_txt}, max|root|={_fmt2(max_root)}, whiteness p={_fmt_p(whiteness_p)}, normalité p={_fmt_p(normality_p)}."
        ),
        narr_call("m.var.meta"),
        narr_call("m.var.audit"),
        "",
    ]
    if tbl_input:
        lines += [
            r"\paragraph{Tableau 1 — Fenêtre effective et données utilisées}",
            md_basic_to_tex(
                "Lecture : ce tableau fixe l’échantillon effectivement estimé (après dropna/transformations). "
                "Il conditionne la comparabilité des résultats (tests, stabilité, IRF/FEVD)."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_input, caption="tbl.var.input_window", label="tab:tbl-var-input-window"),
            narr_call("tbl.var.input_window"),
            "",
        ]

    if tbl_stat:
        lines += [
            r"\paragraph{Tableau 2 — Stationnarité des variables (pré-requis VAR)}",
            md_basic_to_tex(
                "Lecture : un VAR standard suppose des séries stationnaires (ou un traitement approprié). "
                "Ce tableau valide que les transformations appliquées rendent les composantes/variables exploitables."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_stat, caption="tbl.var.stationarity", label="tab:tbl-var-stationarity"),
            narr_call("tbl.var.stationarity"),
            "",
        ]

    if fig_corr:
        lines += [
            r"\paragraph{Figure 1 — Corrélations (heatmap)}",
            md_basic_to_tex(
                "Lecture : repérer colinéarités, blocs de variables, et structure intuitive des interactions. "
                "Des corrélations très élevées peuvent dégrader l’identification et rendre certains coefficients instables."
            ),
            "",
            include_figure(fig_rel=fig_corr, caption="Corrélations entre variables du modèle VAR", label="fig:fig-var-corr-heatmap"),
            narr_call("fig.var.corr_heatmap"),
            "",
        ]

    if tbl_corr:
        lines += [
            r"\paragraph{Tableau 3 — Matrice de corrélation (audit)}",
            md_basic_to_tex(
                "Lecture : audit numérique des corrélations. "
                "Utile si la heatmap est insuffisante pour lire certaines paires."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_corr, caption="tbl.var.corr", label="tab:tbl-var-corr"),
            narr_call("tbl.var.corr"),
            "",
        ]

    # Lag selection
    if tbl_lag:
        lines += [
            r"\paragraph{Tableau 4 — Sélection du nombre de retards}",
            md_basic_to_tex(
                "Lecture : le choix de $p$ est un arbitrage biais–variance. "
                "Un $p$ trop faible omet de la dynamique (résidus autocorrélés) ; "
                "un $p$ trop élevé dégrade la précision et peut provoquer instabilité."
            ),
            "",
            include_table_tex(
                run_root=run_root,
                tbl_rel=tbl_lag,
                caption="tbl.var.lag_selection",
                label="tab:tbl-var-lag-selection",
            ),
            narr_call("tbl.var.lag_selection"),
            "",
        ]
    # Alerte stabilité
    if stable is False:
        lines += [
            md_basic_to_tex(
                "Alerte : le VAR n’est pas stable (racines $\geq 1$). "
                "Les IRF/FEVD ne sont pas interprétables économiquement. "
                "Priorité à la re-spécification : ordre $p$, transformations, choix des variables, traitement saisonnalité/ruptures."
            ),
            "",
        ]

    if tbl_pvals:
        lines += [
            r"\paragraph{Tableau 5 — Significativité des coefficients (p-values)}",
            md_basic_to_tex(
                "Lecture : ce tableau indique quels retards/relations sont statistiquement robustes dans l’estimation. "
                "Il ne remplace pas IRF/FEVD, mais il sert de contrôle : un VAR sur-paramétré produit des coefficients non significatifs en masse."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_pvals, caption="tbl.var.params_pvalues", label="tab:tbl-var-params-pvalues"),
            narr_call("tbl.var.params_pvalues"),
            "",
        ]
    
    # Granger
    if tbl_granger:
        lines += [
            r"\paragraph{Tableau 6 — Causalité de Granger (pairwise)}",
            md_basic_to_tex(
                "Lecture : un rejet signifie un gain de prévision conditionnel à $p$. "
                "Ce résultat est sensible au choix de $p$, à la stationnarité, et aux ruptures. "
                "Une interprétation robuste croise Granger avec stabilité, whiteness et IRF."
            ),
            "",
            include_table_tex(
                run_root=run_root,
                tbl_rel=tbl_granger,
                caption="tbl.var.granger",
                label="tab:tbl-var-granger",
            ),
            narr_call("tbl.var.granger"),
            "",
        ]

    # Sims
    if tbl_sims:
        sims_txt = ""
        if isinstance(sims_q, list) and sims_q:
            sims_txt = f"q testés={sims_q}"
        if sims_err is not None:
            sims_txt += (", " if sims_txt else "") + f"erreurs={sims_err}"
        if sims_min_n is not None:
            sims_txt += (", " if sims_txt else "") + f"min nobs_used={sims_min_n}"

        if sims_txt:
            lines += [md_basic_to_tex(f"Audit Sims : {sims_txt}."), narr_call("m.var.sims"), ""]

        lines += [
            r"\paragraph{Tableau 7 — Causalité à la Sims (leads)}",
            md_basic_to_tex(
                "Lecture : la significativité des leads est un signal d’incohérence temporelle (anticipation) "
                "ou de mauvaise spécification (retards insuffisants, variables inadaptées, effets de calendrier). "
                "Un signal Sims systématique impose une re-spécification avant toute conclusion."
            ),
            "",
            include_table_tex(
                run_root=run_root,
                tbl_rel=tbl_sims,
                caption="tbl.var.sims",
                label="tab:tbl-var-sims",
            ),
            narr_call("tbl.var.sims"),
            "",
        ]


    # IRF (uniquement si stable)
    if fig_irf and stable is not False:
        lines += [
            r"\paragraph{Figure 2 — Fonctions de réponse impulsionnelle (IRF)}",
            md_basic_to_tex(
                "Lecture : les IRF décrivent la trajectoire dynamique après un choc unitaire. "
                "L’interprétation économique requiert un VAR stable. "
                "Si une orthogonalisation impose un ordre causal, cet ordre doit être justifié."
            ),
            "",
            include_figure(fig_rel=fig_irf, caption="Fonctions de réponse impulsionnelle (IRF)", label="fig:fig-var-irf"),
            narr_call("fig.var.irf"),
            "",
        ]

    # FEVD (uniquement si stable)
    if tbl_fevd and stable is not False:
        lines += [
            r"\paragraph{Tableau 8 — Décomposition de variance des erreurs de prévision (FEVD)}",
            md_basic_to_tex(
                "Lecture : mesure la contribution relative de chaque choc à différents horizons. "
                "Une part dominante propre suggère une dynamique auto-entretenue ; "
                "des contributions croisées croissantes suggèrent des canaux de transmission internes."
            ),
            "",
            include_table_tex(
                run_root=run_root,
                tbl_rel=tbl_fevd,
                caption="tbl.var.fevd",
                label="tab:tbl-var-fevd",
            ),
            narr_call("tbl.var.fevd"),
            "",
        ]
    
    # --- Annexes techniques (non insérées dans le corps)
    if tbl_lag_grid or tbl_const or tbl_A1 or tbl_stationary_data:
        lines += [
            md_basic_to_tex(
                "**Annexes techniques (VAR)** : grilles de sélection, constantes, matrices $A_i$ et données stationnarisées "
                "sont disponibles dans les artefacts (non insérées dans le corps)."
            ),
            "",
        ]
        if tbl_lag_grid:        lines += [narr_call("tbl.var.lag_grid"), ""]
        if tbl_const:           lines += [narr_call("tbl.var.const"), ""]
        if tbl_A1:              lines += [narr_call("tbl.var.A1"), ""]
        if tbl_A2:              lines += [narr_call("tbl.var.A2"), ""]
        if tbl_A3:              lines += [narr_call("tbl.var.A3"), ""]
        if tbl_A4:              lines += [narr_call("tbl.var.A4"), ""]
        if tbl_A5:              lines += [narr_call("tbl.var.A5"), ""]
        if tbl_stationary_data: lines += [narr_call("tbl.var.stationary_data"), ""]

    # Note step5 : optionnel
    if note_md.strip():
        lines += [
            md_basic_to_tex("**Note d’interprétation automatisée (Step5)**"),
            md_basic_to_tex(
                "Cette note doit rester strictement cohérente avec : sélection de $p$, stabilité, whiteness, "
                "tests Granger/Sims, IRF et FEVD. Aucune conclusion dynamique sans VAR stable."
            ),
            "",
            md_basic_to_tex(note_md),
            narr_call("m.note.step5"),
            "",
        ]

    # Conclusion
    lines += [
        md_basic_to_tex("**Conclusion**"),
        md_basic_to_tex(
            "Le VAR documente les dépendances dynamiques (prévisionnelles) entre composantes/variables. "
            "La priorité méthodologique est : (i) stabilité, (ii) whiteness, (iii) cohérence de $p$, "
            "puis seulement (iv) IRF/FEVD et lecture des causalités Granger/Sims."
        ),
        narr_call("m.var.audit"),
        "",
    ]

    return "\n".join(lines).strip() + "\n"
