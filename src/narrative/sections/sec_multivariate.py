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


def render_sec_multivariate(
    *,
    run_root: Path,
    manifest: Dict[str, Any],
    sec: SectionSpec,
    metrics_cache: Dict[str, Dict[str, Any]],
) -> str:
    # metrics
    m_var = metrics_cache.get("m.var.meta") or {}
    m_sims = metrics_cache.get("m.var.sims") or {}
    m_audit = metrics_cache.get("m.var.audit") or {}

    note = metrics_cache.get("m.note.step5") or {}
    note_md = ""
    if isinstance(note, dict):
        note_md = note.get("markdown") or note.get("text") or note.get("summary") or ""
    elif isinstance(note, str):
        note_md = note
    note_md = note_md.replace("−", "-")

    # artefacts (tables)
    tbl_lag = lookup(manifest, "tables", "tbl.var.lag_selection")
    tbl_granger = lookup(manifest, "tables", "tbl.var.granger")
    tbl_sims = lookup(manifest, "tables", "tbl.var.sims")
    tbl_fevd = lookup(manifest, "tables", "tbl.var.fevd")

    # artefacts (figures)
    fig_irf = lookup(manifest, "figures", "fig.var.irf")

    # meta fields (robust)
    k = m_var.get("k") or m_var.get("n_vars") or "NA"
    p = m_var.get("p") or m_var.get("lag_order") or m_var.get("selected_lag") or "NA"
    nobs = m_var.get("nobs") or m_var.get("n") or "NA"
    stable = m_audit.get("stable")
    stable_txt = "Oui" if stable is True else "Non" if stable is False else "NA"

    lines: list[str] = []

    # ============================================================
    # SECTION 1 : Mémoire longue (cadre)
    # ============================================================
    lines += [
        r"\section{Analyse de la mémoire longue}",
        "",
        md_basic_to_tex(
            "L’analyse de la mémoire longue vise à caractériser la persistance temporelle des chocs au-delà du cadre classique des modèles ARMA. "
            "Dans de nombreuses séries macroéconomiques et démographiques, la dépendance temporelle ne décroît pas de manière exponentielle "
            "mais selon une loi hyperbolique, traduisant une inertie structurelle profonde. Cette propriété remet en cause l’hypothèse de mémoire "
            "courte implicite des modèles ARMA et justifie le recours à des outils spécifiques."
        ),
        "",
        r"\subsection*{5.1 Mémoire courte versus mémoire longue}",
        md_basic_to_tex(
            "Mémoire courte : la somme des autocorrélations est absolument convergente. "
            "Mémoire longue : la somme diverge et l’autocorrélation décroît selon une loi hyperbolique."
        ),
        "",
        r"\begin{equation}",
        r"\sum_{h=0}^{\infty}|\rho(h)| < \infty",
        r"\end{equation}",
        r"\begin{equation}",
        r"\sum_{h=0}^{\infty}|\rho(h)| = \infty",
        r"\end{equation}",
        "",
        r"\begin{equation}",
        r"\rho(h) \sim C h^{2H-2}\quad \text{lorsque } h\to\infty",
        r"\end{equation}",
        "",
        r"\subsection*{5.2 Fondements théoriques de la mémoire longue}",
        md_basic_to_tex(
            "La mémoire longue traduit des mécanismes d’agrégation et des rigidités institutionnelles générant une persistance de long terme. "
            "En démographie : structures familiales stables, politiques publiques durables, inerties biologiques et sociales. "
            "Économétriquement, elle peut être confondue avec une racine unitaire, d’où la nécessité de diagnostics robustes."
        ),
        "",
        r"\subsection*{5.3 Statistique du Rescaled Range (R/S)}",
        md_basic_to_tex(
            "L’approche R/S (Hurst) examine la croissance de l’amplitude cumulée des écarts à la moyenne."
        ),
        "",
        r"\begin{equation}",
        r"X_k=\sum_{t=1}^{k}(Y_t-\bar{Y})",
        r"\end{equation}",
        r"\begin{equation}",
        r"R(n)=\max_{1\le k\le n}X_k-\min_{1\le k\le n}X_k",
        r"\end{equation}",
        r"\begin{equation}",
        r"\frac{R(n)}{S(n)}",
        r"\end{equation}",
        "",
        r"\begin{equation}",
        r"\mathbb{E}\left[\frac{R(n)}{S(n)}\right]=C n^{H}",
        r"\end{equation}",
        "",
        r"\subsection*{5.4 Exposant de Hurst : interprétation}",
        md_basic_to_tex(
            "H=0,5 : absence de mémoire longue (bruit blanc/ARMA). "
            "H>0,5 : persistance (les chocs tendent à se prolonger). "
            "H<0,5 : antipersistence (retour rapide vers la moyenne)."
        ),
        "",
        r"\subsection*{5.5 Lien entre Hurst et intégration fractionnaire}",
        "",
        r"\begin{equation}",
        r"H=d+\frac{1}{2}",
        r"\end{equation}",
        "",
        md_basic_to_tex(
            "d=0 : ARMA ; 0<d<0,5 : mémoire longue stationnaire ; d≥0,5 : non-stationnarité. "
            "Ce lien conduit naturellement aux modèles ARFIMA."
        ),
        "",
        r"\subsection*{5.6 Modèles ARFIMA(p,d,q)}",
        "",
        r"\begin{equation}",
        r"\Phi(L)(1-L)^{d}Y_t=\Theta(L)\varepsilon_t",
        r"\end{equation}",
        "",
        md_basic_to_tex(
            "Le paramètre d capture l’intensité de la mémoire longue : persistance intermédiaire entre stationnarité et non-stationnarité."
        ),
        "",
        r"\subsection*{5.8 Pièges et limites}",
        md_basic_to_tex(
            "Risques : confusion entre mémoire longue et ruptures, sensibilité aux erreurs de mesure, instabilité en échantillon fini. "
            "Les diagnostics doivent être confrontés aux tests de racine unitaire et à l’analyse des ruptures."
        ),
        "",
        r"\subsection*{5.9 Implications économétriques et économiques}",
        md_basic_to_tex(
            "Une mémoire longue implique des effets persistants à très long horizon : "
            "les modèles ARMA sous-estiment la persistance, et les politiques publiques peuvent produire des effets différés durables. "
            "Cela justifie une extension multivariée pour analyser les canaux dynamiques."
        ),
        "",
    ]

    # ============================================================
    # SECTION 2 : Analyse multivariée (VAR) (résultats Step5)
    # ============================================================
    lines += [
        r"\section{Analyse multivariée : modèles VAR et dépendances dynamiques}",
        "",
        md_basic_to_tex(
            "L’analyse univariée reste limitée pour expliquer les mécanismes sous-jacents. "
            "Un cadre multivarié permet d’étudier les interactions dynamiques sans imposer a priori une causalité structurelle. "
            "Le VAR est l’outil standard pour caractériser ces dépendances."
        ),
        "",
        r"\subsection*{6.1 Fondements du modèle VAR}",
        "",
        r"\begin{equation}",
        r"Y_t=c+A_1Y_{t-1}+A_2Y_{t-2}+\cdots+A_pY_{t-p}+\varepsilon_t",
        r"\end{equation}",
        "",
        md_basic_to_tex(
            "Validité : stationnarité (ou intégration homogène), stabilité (racines hors cercle unité), résidus sans autocorrélation. "
            "Les coefficients individuels s’interprètent rarement ; l’analyse porte sur causalité, IRF et FEVD."
        ),
        "",
        md_basic_to_tex(
            f"**Synthèse quantitative (Step5)** — Dimension $k$={k}, ordre $p$={p}, observations $n$={nobs}, stabilité={stable_txt}."
        ),
        narr_call("m.var.meta"),
        "",
    ]

    # Lag selection (table) + analyse
    if tbl_lag:
        lines += [
            r"\paragraph{Tableau 1 — Sélection du nombre de retards}",
            md_basic_to_tex(
                "Lecture : le choix de $p$ est un arbitrage biais–variance. "
                "Un $p$ trop faible omet de la dynamique (résidus autocorrélés) ; un $p$ trop élevé dégrade la précision. "
                "La décision doit rester cohérente avec les critères d’information et les diagnostics."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_lag, caption="tbl.var.lag_selection", label="tab:tbl-var-lag-selection"),
            narr_call("tbl.var.lag_selection"),
            "",
        ]

    # Granger (table) + analyse
    if tbl_granger:
        lines += [
            r"\paragraph{Tableau 2 — Causalité de Granger}",
            md_basic_to_tex(
                "Lecture : une causalité de Granger signifie un gain de prévision, pas une causalité structurelle. "
                "Les résultats sont sensibles au choix de $p$, à la stationnarité et aux ruptures. "
                "Les conclusions doivent être croisées avec l’IRF et la stabilité du VAR."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_granger, caption="tbl.var.granger", label="tab:tbl-var-granger"),
            narr_call("tbl.var.granger"),
            "",
        ]

    # Sims (table) + analyse
    if tbl_sims:
        lines += [
            r"\paragraph{Tableau 3 — Causalité à la Sims}",
            md_basic_to_tex(
                "Lecture : la significativité des valeurs futures remet en cause l’hypothèse de causalité temporelle stricte "
                "ou révèle une mauvaise spécification. "
                "Un signal Sims fort impose une re-spécification (variables, retards, transformations)."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_sims, caption="tbl.var.sims", label="tab:tbl-var-sims"),
            narr_call("tbl.var.sims"),
            "",
        ]

    # IRF (figure) + analyse
    if fig_irf:
        lines += [
            r"\paragraph{Figure 1 — Fonctions de réponse impulsionnelle (IRF)}",
            md_basic_to_tex(
                "Lecture : les IRF décrivent le profil dynamique d’un choc unitaire. "
                "Une interprétation économique exige un VAR stable ; sinon, les réponses sont non interprétables. "
                "L’orthogonalisation (type Cholesky) implique un ordre causal : il doit être explicitement justifié."
            ),
            "",
            include_figure(fig_rel=fig_irf, caption="fig.var.irf", label="fig:fig-var-irf"),
            narr_call("fig.var.irf"),
            "",
        ]

    # FEVD (table) + analyse
    if tbl_fevd:
        lines += [
            r"\paragraph{Tableau 4 — Décomposition de la variance des erreurs de prévision (FEVD)}",
            md_basic_to_tex(
                "Lecture : mesurer la contribution relative de chaque choc à différents horizons. "
                "Une dominance endogène indique une dynamique principalement auto-entretenue ; "
                "une contribution externe croissante à long horizon suggère des canaux de transmission macroéconomiques."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_fevd, caption="tbl.var.fevd", label="tab:tbl-var-fevd"),
            narr_call("tbl.var.fevd"),
            "",
        ]

    # Audits / note
    if note_md.strip():
        lines += [
            md_basic_to_tex("**Note d’interprétation automatisée (Step5)**"),
            md_basic_to_tex(
                "Cette note doit rester strictement cohérente avec : sélection de $p$, stabilité, causalités (Granger/Sims), IRF et FEVD. "
                "Toute conclusion dynamique sans VAR stable est invalide."
            ),
            "",
            md_basic_to_tex(note_md),
            narr_call("m.note.step5"),
            "",
        ]

    lines += [
        md_basic_to_tex("**Conclusion**"),
        md_basic_to_tex(
            "Le diagnostic multivarié complète l’univarié : il identifie direction, délais et persistance des interactions. "
            "La priorité méthodologique est la stabilité et la blancheur des résidus ; ensuite seulement viennent IRF et FEVD. "
            "Ces résultats cadrent l’analyse de long terme (cointégration/VECM) de la section suivante."
        ),
        narr_call("m.var.audit"),
        "",
    ]

    return "\n".join(lines).strip() + "\n"
