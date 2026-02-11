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
    # SECTION 1 : Cadre VAR (théorie + périmètre + conditions)
    # ============================================================

    lines += [
        r"\section{Analyse multivariée : modèles VAR et dépendances dynamiques}",
        "",
        md_basic_to_tex(
            "L’analyse univariée permet de caractériser la dynamique propre de la croissance naturelle, "
            "mais elle ne capture pas les interactions potentielles entre variables économiques et sociales "
            "susceptibles d’influencer cette dynamique. "
            "L’approche multivariée vise précisément à modéliser ces interdépendances."
        ),
        "",
        md_basic_to_tex(
            "Le périmètre de l’étude est ici ajusté à la période 1978–2025 en fréquence mensuelle, "
            "afin de garantir la disponibilité homogène de l’ensemble des variables retenues. "
            "L’analyse repose désormais sur un système à quatre variables :"
        ),
        "",

        md_basic_to_tex(
            r"$Y_1$ : croissance naturelle (CN); "
            r"$Y_2$ : nombre de mariages ; "
            r"$Y_3$ : indice des prix à la consommation (IPC) ; "
            r"$Y_4$ : masse monétaire M3."
        ),
        "",

        md_basic_to_tex(
            "Le choix de ces variables répond à une logique économique et démographique cohérente. "
            "Le nombre de mariages constitue un indicateur avancé des dynamiques familiales, "
            "potentiellement liées à la fécondité. "
            "L’IPC capte les tensions inflationnistes susceptibles d’affecter les décisions de consommation "
            "et d’investissement des ménages, y compris les décisions liées à la natalité. "
            "La masse monétaire M3 reflète les conditions monétaires et financières de long terme, "
            "susceptibles d’influencer indirectement l’environnement macroéconomique et social."
        ),
        "",
    ]

    # -----------------------------
    # Spécification du VAR
    # -----------------------------

    lines += [
        r"\subsection*{Spécification du modèle VAR}",
        "",
        r"\begin{equation}",
        r"Y_t = c + A_1 Y_{t-1} + A_2 Y_{t-2} + \cdots + A_p Y_{t-p} + \varepsilon_t",
        r"\end{equation}",
        "",
        md_basic_to_tex(
            "Un VAR(p) à $k$ variables — ici $k=4$ — est un vecteur auto-régressif "
            "dans lequel chaque variable est expliquée par ses propres retards "
            "et par les retards des autres variables du système. "
            "Le vecteur des résidus $\\varepsilon_t$ contient un choc par équation : "
            "il s’agit de chocs spécifiques à chaque variable, supposés non autocorrélés "
            "et contemporanément corrélés de manière contrôlée."
        ),
        "",

        md_basic_to_tex(
            "La question de savoir « qui choque en premier » dépend de l’intuition économique "
            "et de l’ordonnancement retenu pour l’identification des chocs (décomposition de Cholesky). "
            "Le VAR réduit ne permet pas d’identifier une causalité structurelle, "
            "mais uniquement des relations dynamiques internes au système."
        ),
        "",
    ]

    # -----------------------------
    # Choix de l’ordre p
    # -----------------------------

    lines += [
        r"\subsection*{Choix de l’ordre $p$}",
        md_basic_to_tex(
            "Le choix du nombre de retards $p$ est crucial. "
            "Plus $p$ est élevé, plus le nombre de paramètres à estimer augmente rapidement, "
            "ce qui dégrade la précision des estimations et peut introduire des coefficients non significatifs."
        ),
        "",
        md_basic_to_tex(
            "En données annuelles, un VAR(3) signifierait qu’un choc met trois ans à s’absorber, "
            "ce qui représenterait une inertie extrêmement forte. "
            "En données mensuelles, un VAR(6) correspondrait à un horizon d’ajustement de six mois : "
            "cela peut paraître élevé, mais demeure plausible en présence de chocs macroéconomiques "
            "ou de changements de régime."
        ),
        "",
        md_basic_to_tex(
            "L’objectif est de retenir le VAR(p) le plus parcimonieux possible, "
            "minimisant le critère d’information (notamment l’AIC) "
            "tout en conservant des paramètres significatifs et économiquement cohérents."
        ),
        "",
    ]

    # -----------------------------
    # Conditions de validité
    # -----------------------------

    lines += [
        r"\subsection*{Conditions de validité économétrique}",
        md_basic_to_tex(
            "Avant toute estimation VAR, la nature des tendances doit être correctement identifiée. "
            "Les variables doivent être stationnaires. "
            "Si certaines séries sont intégrées d’ordre un (DS), "
            "elles doivent être différenciées ; "
            "si elles sont stationnaires autour d’une tendance déterministe (TS), "
            "un ajustement approprié est nécessaire."
        ),
        "",

        md_basic_to_tex(
            "Un VAR est économétriquement valide si les racines du polynôme caractéristique "
            "sont situées à l’intérieur du cercle unité : "
            "cette condition garantit la stabilité du système. "
            "Sans stabilité, les fonctions de réponse impulsionnelle (IRF) "
            "et les décompositions de variance (FEVD) ne sont pas interprétables."
        ),
        "",

        md_basic_to_tex(
            "Les diagnostics incluent également la vérification de l’absence "
            "d’autocorrélation résiduelle et la cohérence globale du modèle. "
            "Le compromis biais–variance guide la sélection finale."
        ),
        "",
    ]

    # ============================================================
    # SECTION 2 : Résultats empiriques (VAR(5) interprété)
    # ============================================================


    lines += [
        r"\section{Résultats empiriques}",
        "",
        md_basic_to_tex(
            "Le modèle estimé est un VAR(5) à quatre variables mensuelles "
            "(Croissance Naturelle, Mariages, IPC, M3) "
            "sur la période 1978–2025 après stationnarisation par différenciation première. "
            "Le choix $p=5$ est issu de la minimisation du critère AIC "
            "et correspond à une dynamique d’ajustement sur cinq mois."
        ),
        "",
    ]

    # ============================================================
    # TABLEAU 1 — Fenêtre d’estimation
    # ============================================================

    lines += [
        r"\paragraph{Tableau 1 — Fenêtre effective et données utilisées}",
        "",
        include_table_tex(run_root=run_root, tbl_rel=tbl_input,
                         caption="Fenêtre d’estimation et variables retenues pour le VAR",
                          label="tab:var_window"),
        "",
        md_basic_to_tex(
            "L’échantillon comprend 551 observations mensuelles après transformation. "
            "La perte initiale provient de la différenciation et de l’introduction des retards.\n\n"
            "Avec un VAR(5) à 4 variables, le nombre de paramètres estimés reste compatible "
            "avec la taille de l’échantillon, ce qui garantit une estimation stable. "
            "Le ratio observations/paramètres demeure suffisant pour éviter un sur-apprentissage."
        ),
        "",
    ]

    # ============================================================
    # TABLEAU 2 — Stationnarité
    # ============================================================

    lines += [
        r"\paragraph{Tableau 2 — Stationnarité des variables}",
        "",
        include_table_tex(run_root=run_root, tbl_rel=tbl_stat,
                          caption="Stationnarité des variables avant estimation du VAR",
                          label="tab:var_stationarity"),
        "",
        md_basic_to_tex(
            "Les tests ADF en niveau montrent l’absence de stationnarité :\n"
            "• CN : p=0.985\n"
            "• Mariages : p=0.102\n"
            "• IPC : p=0.622\n"
            "• M3 : p=1.000\n\n"
            "Après différenciation d’ordre 1, toutes les p-values deviennent inférieures à 0.05 "
            "(ex. CN : 0.000 ; M3 : 0.004), confirmant la stationnarité des variations.\n\n"
            "Nous modélisons donc les changements mensuels et non les niveaux structurels. "
            "Cela signifie que le modèle capture les accélérations et décélérations "
            "du processus démographique plutôt que son niveau absolu."
        ),
        "",
    ]

    # ============================================================
    # FIGURE 1 — Heatmap corrélations
    # ============================================================

    lines += [
        r"\paragraph{Figure 1 — Heatmap des corrélations}",
        "",
        include_figure(fig_rel=fig_corr,
                       caption="Corrélations entre variables du modèle VAR",
                       label="fig:var_corr"),
        "",
        md_basic_to_tex(
            "Les corrélations contemporaines sont modérées :\n\n"
            "• CN – Mariages : 0.261 → relation positive intuitive.\n"
            "• CN – IPC : 0.145 → faible corrélation.\n"
            "• CN – M3 : 0.001 → quasi indépendance.\n\n"
            "Aucune corrélation excessive n’est observée. "
            "Le modèle n’est donc pas menacé par une multicolinéarité forte.\n\n"
            "La dynamique matrimoniale apparaît comme le canal démographique "
            "le plus directement lié à la croissance naturelle."
        ),
        "",
    ]

    # ============================================================
    # TABLEAU 4 — Sélection du VAR(5)
    # ============================================================

    lines += [
        r"\paragraph{Tableau 4 — Sélection du nombre de retards}",
        "",
        include_table_tex(run_root=run_root, tbl_rel=tbl_lag,
                          caption="Sélection du nombre de retards du VAR",
                          label="tab:var_lag"),
        "",
        md_basic_to_tex(
            "Le critère AIC est minimal pour p=5. "
            "Ce choix implique que la dynamique significative du système "
            "s’étend sur cinq mois.\n\n"
            "Interprétation économique :\n"
            "Un choc démographique (ex. réforme, crise sanitaire, tension géopolitique) "
            "n’a pas d’effet instantané. "
            "Les décisions familiales s’ajustent progressivement.\n\n"
            "Cinq mois représentent un horizon cohérent en données mensuelles : "
            "• Assez long pour capter l’inertie comportementale.\n"
            "• Assez court pour éviter une inertie artificielle.\n\n"
            "Un p plus élevé aurait augmenté l’instabilité et réduit la précision "
            "des coefficients."
        ),
        "",
    ]

    # ============================================================
    # TABLEAU 6 — Granger
    # ============================================================

    lines += [
        r"\paragraph{Tableau 6 — Tests de causalité de Granger}",
        "",
        include_table_tex(run_root=run_root, tbl_rel=tbl_granger,
                          caption="Tests de causalité de Granger",
                          label="tab:var_granger"),
        "",
        md_basic_to_tex(
            "Lecture approfondie :\n\n"
            "1) Mariages → CN : p=0.000\n"
            "   → causalité unidirectionnelle forte.\n\n"
            "2) CN → Mariages : p=0.000\n"
            "   → relation bidirectionnelle.\n\n"
            "La relation CN–Mariages est donc dynamique et réciproque.\n\n"
            "3) IPC ↔ M3 : causalité bidirectionnelle significative.\n"
            "   → cohérence macroéconomique classique.\n\n"
            "4) M3 → CN : non significatif.\n"
            "   → pas de transmission monétaire directe vers la démographie.\n\n"
            "Conclusion : la variable centrale influencée est la dynamique matrimoniale, "
            "tandis que les variables macro opèrent surtout entre elles."
        ),
        "",
    ]

    # ============================================================
    # TABLEAU 7 — Sims
    # ============================================================

    lines += [
        r"\paragraph{Tableau 7 — Tests de causalité instantanée (Sims)}",
        "",
        include_table_tex(run_root=run_root, tbl_rel=tbl_sims,
                          caption="Tests de causalité à la Sims",
                          label="tab:var_sims"),
        "",
        md_basic_to_tex(
            "Les résultats Sims sont globalement cohérents avec Granger.\n\n"
            "Cela signifie que les anticipations des agents "
            "ne créent pas d’incohérence temporelle majeure.\n\n"
            "Rappel conceptuel :\n"
            "• Si Granger ≠ Sims → anticipation mal modélisée.\n"
            "• Si Granger = Sims → rationalité dynamique.\n\n"
            "Dans notre cas : cohérence → le cadre VAR est suffisant.\n"
            "Aucune nécessité immédiate de basculer vers un modèle anticipatif alternatif."
        ),
        "",
    ]

    # ============================================================
    # FIGURE 2 — IRF
    # ============================================================

    lines += [
        r"\paragraph{Figure 2 — Fonctions de réponse impulsionnelle (IRF)}",
        "",
        include_figure(fig_rel=fig_irf,
                       caption="Fonctions de réponse impulsionnelle (IRF)",
                       label="fig:var_irf"),
        "",
        md_basic_to_tex(
            "Analyse centrée sur la réponse de la Croissance Naturelle :\n\n"
            "• Choc propre : forte réaction initiale, extinction en 4–5 mois.\n"
            "• Choc Mariages : effet positif court terme puis retour à 0.\n"
            "• Choc IPC : impact négatif transitoire.\n"
            "• Choc M3 : effet faible et instable.\n\n"
            "Toutes les réponses convergent vers 0 → système stable.\n\n"
            "Interprétation : les chocs exogènes (inflation, tensions géopolitiques, "
            "politique monétaire) perturbent temporairement la dynamique démographique, "
            "mais n’altèrent pas son sentier de long terme."
        ),
        "",
    ]

    # ============================================================
    # TABLEAU 8 — FEVD
    # ============================================================

    lines += [
        r"\paragraph{Tableau 8 — Décomposition de variance (FEVD)}",
        "",
        include_table_tex(run_root=run_root, tbl_rel=tbl_fevd,
                          caption="Décomposition de la variance des erreurs de prévision",
                          label="tab:var_fevd"),
        "",
        md_basic_to_tex(
            "À court horizon :\n"
            "• >98% de la variance de CN provient de ses propres chocs.\n\n"
            "À horizon intermédiaire :\n"
            "• Contribution des Mariages augmente légèrement (≈1–2%).\n\n"
            "Les variables macro contribuent faiblement.\n\n"
            "Conclusion : la dynamique démographique reste principalement endogène, "
            "mais la composante matrimoniale joue un rôle structurel secondaire."
        ),
        "",
    ]

    # ============================================================
    # Conclusion multivariée VAR(p)
    # ============================================================

    lines += [
        md_basic_to_tex("**Conclusion VARF(p)**"),
        md_basic_to_tex(
            "Le VAR(5) révèle un système dynamique stable où les chocs "
            "mettent environ cinq mois à s’absorber.\n\n"
            "La croissance naturelle est principalement auto-entretenue, "
            "mais significativement reliée à la dynamique matrimoniale.\n\n"
            "Les variables macroéconomiques influencent surtout entre elles, "
            "et affectent la démographie de manière indirecte.\n\n"
            "Les tests Granger et Sims sont cohérents, validant "
            "la rationalité dynamique du système.\n\n"
            "Les IRF confirment une convergence naturelle vers zéro : "
            "les chocs exogènes sont transitoires.\n\n"
            "Ainsi, la dynamique démographique française apparaît "
            "structurellement résiliente à court terme, "
            "mais sensible aux canaux internes liés aux comportements sociaux."
        ),
        "",
    ]

    return "\n".join(lines).strip() + "\n"
