# src/narrative/sections/sec_descriptive.py
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

def _looks_too_large_table(tbl_rel: str) -> bool:
    # Heuristique simple : si le fichier contient "longtable" ou si c’est la table STL détaillée connue.
    # On évite d’ouvrir/compter les lignes (coût/IO) : règle métier stable.
    return "tbl.desc.decomp_components" in (tbl_rel or "")

def render_sec_descriptive(
    *,
    run_root: Path,
    manifest: Dict[str, Any],
    sec: SectionSpec,
    metrics_cache: Dict[str, Dict[str, Any]],
) -> str:
    m_strength = (metrics_cache.get("m.desc.seasonal_strength") or {}).get("value")
    m_type = (metrics_cache.get("m.desc.seasonality_type") or {}).get("value")

    note = (metrics_cache.get("m.note.step2") or {})
    note_md = ""
    if isinstance(note, dict):
        note_md = note.get("markdown") or note.get("text") or note.get("summary") or ""
    elif isinstance(note, str):
        note_md = note
    note_md = note_md.replace("−", "-")

    # artefacts
    fig_level = lookup(manifest, "figures", "fig.desc.level")
    fig_decomp = lookup(manifest, "figures", "fig.desc.decomp")

    tbl_summary = lookup(manifest, "tables", "tbl.desc.summary")
    tbl_season = lookup(manifest, "tables", "tbl.desc.seasonality")
    tbl_decomp = lookup(manifest, "tables", "tbl.desc.decomp_components")

    lines: list[str] = []

    # ============================================================
    # SECTION : Analyse descriptive (texte structurant)
    # ============================================================
    lines += [
        r"\section{Analyse descriptive et décomposition de la série}",
        "",
        md_basic_to_tex(
            "Avant toute modélisation stochastique formelle, l’analyse descriptive constitue une étape économétrique fondamentale. "
            "Elle permet d’identifier la structure globale de la série, de formuler des hypothèses plausibles sur le processus générateur "
            "des données (DGP) et d’éviter les erreurs de spécification ultérieures. Cette étape n’est pas purement exploratoire : "
            "elle conditionne directement le choix des tests de stationnarité et des modèles dynamiques."
        ),
        "",
        r"\subsection*{Représentation graphique et analyse qualitative}",
        md_basic_to_tex(
            "La représentation graphique de la croissance naturelle en niveau permet une première lecture de la dynamique temporelle. "
            "La visualisation vise à identifier : (i) une tendance de long terme, (ii) d’éventuels cycles de moyen terme, "
            "(iii) des ruptures structurelles potentielles, et (iv) des périodes de volatilité accrue. "
            "Une tendance apparente suggère une non-stationnarité possible en niveau, alors que des fluctuations autour d’une moyenne "
            "constante orientent vers un processus stationnaire. Cette lecture n’est pas concluante en soi, mais elle cadre l’interprétation "
            "des tests formels."
        ),
        "",
    ]

    if fig_level:
        lines += [
            r"\paragraph{Figure 1 — Série en niveau}",
            md_basic_to_tex(
                "Lecture : repérer pente de long terme, phases de retournement, et épisodes de volatilité atypique. "
                "Toute rupture visuelle doit être gardée en mémoire : les tests de racine unitaire y sont sensibles si la rupture n’est pas modélisée."
            ),
            "",
            include_figure(fig_rel=fig_level, caption="fig.desc.level", label="fig:fig-desc-level"),
            narr_call("fig.desc.level"),
            "",
        ]

    lines += [
        r"\subsection*{Décomposition formelle de la série temporelle}",
        md_basic_to_tex(
            "Afin de structurer l’analyse, la série est décomposée selon un schéma additif classique :"
        ),
        "",
        r"\begin{equation}",
        r"Y_t = T_t + S_t + \varepsilon_t",
        r"\end{equation}",
        "",
        md_basic_to_tex(
            "où $T_t$ représente la tendance, $S_t$ la composante saisonnière, et $\\varepsilon_t$ le résidu supposé de moyenne nulle. "
            "Cette décomposition suppose une séparation raisonnable des composantes ; elle doit être explicitée car une saisonnalité mal traitée "
            "déforme ACF/PACF et les diagnostics ultérieurs."
        ),
        "",
        r"\subsection*{Analyse de la tendance}",
        md_basic_to_tex(
            "La tendance traduit l’évolution structurelle de long terme. Elle peut être déterministe (fonction explicite du temps) "
            "ou stochastique (accumulation de chocs permanents). Cette distinction est critique : elle conditionne TS/DS et le traitement "
            "requis pour stationnariser la série. Une mauvaise identification de la tendance entraîne des erreurs sévères de spécification, "
            "notamment dans les tests de racine unitaire."
        ),
        "",
        r"\subsection*{Analyse de la saisonnalité : cadre théorique}",
        md_basic_to_tex(
            "Une saisonnalité mal spécifiée a des conséquences directes : distorsion des autocorrélations, perte de puissance des tests de "
            "stationnarité, et identification erronée des ordres AR/MA. Quatre configurations doivent être envisagées : "
            "absence de saisonnalité, saisonnalité déterministe (stable), saisonnalité stochastique (affectée par chocs), "
            "et saisonnalité évolutive (amplitude/forme variable)."
        ),
        "",
        r"\subsection*{Conséquences économétriques de la saisonnalité}",
        md_basic_to_tex(
            "Conclusion opérationnelle : l’analyse de la saisonnalité doit précéder toute estimation ARMA/ARIMA. "
            "C’est un prérequis pour interpréter correctement ACF/PACF et calibrer les différenciations éventuelles."
        ),
        "",
        r"\subsection*{Analyse descriptive de la volatilité}",
        md_basic_to_tex(
            "La volatilité descriptive correspond à l’étude de la dispersion dans le temps. En démographie, une variance relativement stable "
            "est attendue hors chocs exceptionnels (crises sanitaires, épisodes extrêmes). Une hausse persistante de variance peut signaler "
            "une rupture, un changement de régime, ou une transformation du DGP, et prépare la discussion sur les diagnostics résiduels."
        ),
        "",
        r"\subsection*{Détection visuelle des ruptures structurelles}",
        md_basic_to_tex(
            "L’analyse graphique permet d’identifier des dates candidates de rupture, potentiellement liées à politiques familiales, chocs "
            "macro-économiques, ou événements sanitaires. Toute rupture repérée ici doit être considérée lors de l’interprétation des tests "
            "de stationnarité et des modèles, car les ruptures non modélisées biaisent les inférences."
        ),
        "",
        r"\subsection*{Rôle de l’analyse descriptive dans la stratégie globale}",
        md_basic_to_tex(
            "Cette étape joue un rôle structurant : elle formalise des hypothèses plausibles sur la stationnarité, oriente les tests, "
            "et limite les erreurs de spécification dans les modèles dynamiques. Elle sert de pont entre la construction des données "
            "et la stationnarité, traitée dans la section suivante."
        ),
        "",
    ]

    # ============================================================
    # SECTION 2 : Résultats STL / artefacts (figures + tables + analyses)
    # ============================================================
    lines += [
        r"\section{Résultats de la décomposition et diagnostics}",
        "",
        md_basic_to_tex(
            f"Synthèse quantitative : **force saisonnière (STL) = {_fmt2(m_strength)}** ; "
            f"**qualification = {m_type}**. Ces sorties doivent être cohérentes avec la figure de décomposition et les tableaux associés."
        ),
        narr_call("m.desc.seasonal_strength"),
        narr_call("m.desc.seasonality_type"),
        "",
    ]
    # --- Analyse économétrique guidée par les métriques STL
    strength_txt = _fmt2(m_strength)
    qual_txt = str(m_type or "NA")

    lines += [
        r"\paragraph{Analyse — saisonnalité et implications de spécification}",
        md_basic_to_tex(
            "La force saisonnière issue de la STL synthétise le poids de la composante saisonnière relativement à la variance totale. "
            "Elle sert de critère opérationnel pour décider si la saisonnalité doit être traitée explicitement (différence saisonnière, "
            "dummies mensuelles, ou composante saisonnière dans un modèle)."
        ),
        "",
        md_basic_to_tex(
            f"Dans les résultats présents : **force = {strength_txt}** et **qualification = {qual_txt}**. "
            "Interprétation : si la saisonnalité est faible/absente, la dynamique est dominée par la tendance et/ou des ruptures ; "
            "si elle est forte et stable, un traitement saisonnier explicite est requis avant l’identification ARMA/ARIMA afin d’éviter "
            "des autocorrélations artificielles et un mauvais choix d’ordres."
        ),
        "",
        md_basic_to_tex(
            "Conséquence directe pour la section suivante (stationnarité) : une saisonnalité non traitée peut faire rejeter à tort "
            "une hypothèse de stationnarité, ou dégrader la puissance des tests (ADF/PP), car les corrélations périodiques contaminent "
            "les résidus des régressions de test."
        ),
        "",
    ]

    if fig_decomp:
        lines += [
            r"\paragraph{Figure 1 — Décomposition (STL)}",
            md_basic_to_tex(
                "Lecture : la tendance isole le mouvement structurel ; la composante saisonnière capture les cycles réguliers ; "
                "le résidu doit être centré et dépourvu de structure évidente. Un résidu encore autocorrélé signale une dynamique "
                "non expliquée et impose prudence sur la spécification ultérieure."
            ),
            "",
            include_figure(fig_rel=fig_decomp, caption="fig.desc.decomp", label="fig:fig-desc-decomp"),
            narr_call("fig.desc.decomp"),
            "",
        ]

        lines += [
            md_basic_to_tex(
                "Contrôle de cohérence : la composante saisonnière visualisée doit être compatible avec la force saisonnière reportée ci-dessus ; "
                "si divergence apparente, cela suggère une hétérogénéité temporelle de la saisonnalité (amplitude variable) et impose prudence."
            ),
            "",
        ]
    if tbl_summary:
        lines += [
            r"\paragraph{Tableau 1 — Synthèse descriptive}",
            md_basic_to_tex(
                "Lecture : valider l’ordre de grandeur, dispersion, et asymétrie. Ce tableau sert de contrôle de cohérence "
                "avec la figure de niveau et les diagnostics de tendance."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_summary, caption="tbl.desc.summary", label="tab:tbl-desc-summary"),
            narr_call("tbl.desc.summary"),
            "",
        ]

    if tbl_season:
        lines += [
            r"\paragraph{Tableau 2 — Indicateurs de saisonnalité}",
            md_basic_to_tex(
                "Lecture : vérifier si l’amplitude et la stabilité saisonnière justifient un traitement explicite. "
                "Si la saisonnalité est qualifiée de faible/absente, la priorité se déplace vers la tendance et les ruptures ; "
                "si elle est forte, elle doit être traitée avant toute identification ARMA/ARIMA."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_season, caption="tbl.desc.seasonality", label="tab:tbl-desc-seasonality"),
            narr_call("tbl.desc.seasonality"),
            "",
        ]

    # --- Table détaillée des composantes (trop volumineuse -> annexe contrôlée)
    if tbl_decomp:
        if _looks_too_large_table(tbl_decomp):
            lines += [
                md_basic_to_tex(
                    "**Table détaillée des composantes (STL)** : disponible en annexe/artefacts (non insérée dans le corps du rapport "
                    "pour préserver la lisibilité)."
                ),
                narr_call("tbl.desc.decomp_components"),
                "",
            ]
        else:
            lines += [
                r"\paragraph{Tableau — Détail des composantes (STL)}",
                md_basic_to_tex(
                    "Lecture : contrôle fin des composantes (tendance, saisonnalité, résidu) au niveau observationnel. "
                    "Cette table sert d’audit, pas de synthèse."
                ),
                "",
                include_table_tex(
                    run_root=run_root,
                    tbl_rel=tbl_decomp,
                    caption="tbl.desc.decomp_components",
                    label="tab:tbl-desc-decomp-components",
                ),
                narr_call("tbl.desc.decomp_components"),
                "",
            ]

    if note_md.strip():
        lines += [
            md_basic_to_tex("**Note d’interprétation automatisée**"),
            md_basic_to_tex(
                "La table détaillée des composantes est disponible dans les artefacts (tables) ; le rapport présente une synthèse et les graphiques. "
                "Toute affirmation sur tendance/saisonnalité doit être justifiée par les artefacts ci-dessus."
            ),
            "",
            md_basic_to_tex(note_md),
            narr_call("m.note.step2"),
            "",
        ]

    lines += [
        md_basic_to_tex("**Conclusion**"),
        md_basic_to_tex(
            "La décomposition et les diagnostics descriptifs cadrent la stationnarité et la spécification. "
            "Ils déterminent si la série doit être traitée (tendance/saisonnalité) avant l’identification des modèles dynamiques."
        ),
        "",
    ]

    return "\n".join(lines).strip() + "\n"
