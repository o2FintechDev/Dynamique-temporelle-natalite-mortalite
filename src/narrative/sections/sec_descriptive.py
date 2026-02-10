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

    # ===============================
    # SECTION 1 : Analyse descriptive
    # ===============================
    lines += [
        r"\section{Analyse descriptive de la croissance naturelle}",
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
            "",
            include_figure(
                fig_rel=fig_level,
                caption="Évolution de la croissance naturelle en France (1975--2025)",
                label="fig:desc-level"
            ),
            narr_call("fig.desc.level"),
            "",
            md_basic_to_tex(
                "La figure met en évidence une dynamique de long terme marquée par une tendance globalement décroissante "
                "de la croissance naturelle sur l’ensemble de la période étudiée. "
                "Jusqu’au milieu des années 2010, la série oscille autour de valeurs positives, "
                "avec une saisonnalité prononcée et relativement stable.\n\n"

                "À partir de la fin des années 2010, une inflexion nette apparaît, "
                "conduisant à des valeurs de plus en plus faibles, puis négatives. "
                "Cette évolution suggère l’existence d’une rupture structurelle récente, "
                "cohérente avec l’inversion observée entre les taux de natalité et de mortalité.\n\n"

                "D’un point de vue économétrique, cette lecture visuelle oriente vers une possible "
                "non-stationnarité en niveau et justifie le recours à des tests formels de racine unitaire "
                "dans la section suivante."
            ),
            "",
        ]


    lines += [
        r"\subsection*{Décomposition formelle de la série temporelle}",
        md_basic_to_tex(
            "Afin de structurer l’analyse et de distinguer les différentes composantes de la dynamique observée, "
            "la série de croissance naturelle est décomposée selon un schéma additif classique."
        ),
        "",
        r"\begin{equation}",
        r"Y_t = T_t + S_t + \varepsilon_t",
        r"\end{equation}",
        "",
        md_basic_to_tex(
            "où $T_t$ désigne la composante de tendance de long terme, $S_t$ la composante saisonnière, "
            "et $\\varepsilon_t$ un terme résiduel supposé de moyenne nulle.\n\n"

            "Le choix d’une décomposition additive est justifié par l’observation graphique préalable : "
            "l’amplitude des fluctuations saisonnières demeure globalement stable dans le temps "
            "et ne semble pas proportionnelle au niveau de la série. "
            "De plus, la croissance naturelle prenant des valeurs négatives sur la période récente, "
            "une décomposition multiplicative serait économétriquement inappropriée.\n\n"

            "Cette hypothèse additive constitue un cadre de travail initial. "
            "Elle permet d’isoler les composantes structurelles de la série et prépare l’analyse formelle "
            "de la stationnarité et des propriétés dynamiques étudiées dans les sections suivantes."
        ),
        "",

        r"\subsection*{Synthèse descriptive et implications pour l’analyse économétrique}",
        md_basic_to_tex(
            "L’analyse descriptive met en évidence plusieurs faits stylisés majeurs. "
            "La série de croissance naturelle présente une dynamique de long terme marquée, "
            "une saisonnalité régulière et persistante, ainsi que des phases de rupture visuelle "
            "associées à des chocs macro-sociaux majeurs.\n\n"

            "Ces éléments suggèrent que la série ne peut être assimilée à un simple processus aléatoire stationnaire. "
            "La présence conjointe d’une tendance apparente, d’une composante saisonnière structurée "
            "et de fluctuations résiduelles hétérogènes impose une analyse économétrique rigoureuse "
            "des propriétés de stationnarité et de persistance.\n\n"

            "L’analyse descriptive ne permet pas, à elle seule, de trancher sur la nature exacte "
            "du processus générateur des données. "
            "Elle fournit toutefois un cadre interprétatif indispensable pour guider "
            "les tests formels de racine unitaire, le traitement de la saisonnalité "
            "et la spécification des modèles dynamiques.\n\n"

            "La section suivante est ainsi consacrée à l’étude formelle de la stationnarité "
            "et à l’identification de la nature temporelle de la croissance naturelle française."
        ),
        "",
    ]

    # =============================
    # SECTION 2 : Décomposition STL
    # =============================

    lines += [
        r"\section{Décomposition STL de la croissance naturelle}",
        "",
        md_basic_to_tex(
            "Afin d’analyser finement la dynamique temporelle de la croissance naturelle, "
            "la série est décomposée à l’aide de la méthode STL (Seasonal and Trend decomposition using Loess). "
            "Cette méthode permet de séparer explicitement trois composantes : "
            "une tendance de long terme, une saisonnalité potentiellement évolutive, "
            "et un résidu capturant les chocs irréguliers.\n\n"

            "Contrairement aux décompositions classiques, la STL autorise une saisonnalité non strictement stable dans le temps, "
            "ce qui est particulièrement adapté aux données démographiques mensuelles. "
            "Les comportements de fécondité et de mortalité peuvent en effet conserver une structure saisonnière "
            "tout en voyant leur amplitude ou leur régularité évoluer sous l’effet de facteurs sociaux, sanitaires ou institutionnels."
        ),
        "",
    ]

    # ----------------------------
    # Figure STL (sans paragraphe avant)
    # ----------------------------
    if fig_decomp:
        lines += [
            include_figure(
                fig_rel=fig_decomp,
                caption="Décomposition STL de la croissance naturelle en France (tendance, saisonnalité, résidu)",
                label="fig:fig-desc-decomp",
            ),
            narr_call("fig.desc.decomp"),
            "",
        ]

    # ----------------------------
    # Interprétation de la figure
    # ----------------------------
    lines += [
        md_basic_to_tex(
            "La décomposition met en évidence plusieurs faits stylisés majeurs.\n\n"

            "La composante de tendance révèle une dynamique de long terme marquée par un infléchissement progressif "
            "de la croissance naturelle, cohérent avec le vieillissement démographique et la baisse structurelle de la fécondité. "
            "Cette évolution suggère que les chocs observés ne sont pas purement transitoires, mais s’inscrivent dans une trajectoire durable.\n\n"

            "La composante saisonnière présente une structure régulière, traduisant la persistance de comportements saisonniers "
            "dans les naissances et les décès. Toutefois, son amplitude n’est pas strictement constante dans le temps, "
            "ce qui indique une saisonnalité évolutive plutôt que parfaitement déterministe.\n\n"

            "Enfin, le résidu est globalement centré autour de zéro, mais laisse apparaître des épisodes de chocs marqués, "
            "notamment lors d’événements sanitaires ou climatiques exceptionnels. "
            "Ces chocs ponctuels n’altèrent pas la structure globale mais doivent être pris en compte "
            "dans les diagnostics économétriques ultérieurs."
        ),
        "",
    ]

    # ----------------------------
    # Métriques STL : force et qualification
    # ----------------------------
    strength_txt = _fmt2(m_strength)
    qual_txt = str(m_type or "NA")

    lines += [
        md_basic_to_tex(
            f"D’un point de vue quantitatif, la STL fournit deux indicateurs synthétiques essentiels. "
            f"La **force saisonnière** est estimée à **{strength_txt}**, indiquant que la saisonnalité explique "
            "une part substantielle de la variance totale de la série. "
            f"La **qualification** de la saisonnalité est **{qual_txt}**, ce qui confirme "
            "que son intensité n’est pas strictement stable dans le temps.\n\n"

            "Ces résultats sont particulièrement cohérents avec une lecture démographique : "
            "les cycles saisonniers liés aux comportements reproductifs persistent, "
            "mais leur intensité peut être modifiée par des changements sociaux profonds "
            "(évolution des normes familiales, politiques publiques, crises sanitaires)."
        ),
        narr_call("m.desc.seasonal_strength"),
        narr_call("m.desc.seasonality_type"),
        "",
    ]

    # ----------------------------
    # Tables associées
    # ----------------------------
    if tbl_summary:
        lines += [
            r"\paragraph{Tableau — Synthèse statistique}",
            md_basic_to_tex(
                "Ce tableau fournit un contrôle de cohérence numérique des ordres de grandeur, "
                "de la dispersion et de l’asymétrie de la croissance naturelle."
            ),
            "",
            include_table_tex(
                run_root=run_root,
                tbl_rel=tbl_summary,
                caption="Résumé statistique de la croissance naturelle",
                label="tab:tbl-desc-summary",
            ),
            narr_call("tbl.desc.summary"),
            "",
        ]

    if tbl_season:
        lines += [
            r"\paragraph{Tableau — Indicateurs de saisonnalité}",
            md_basic_to_tex(
                "Ces indicateurs complètent la lecture graphique en quantifiant la présence "
                "et la nature de la saisonnalité."
            ),
            "",
            include_table_tex(
                run_root=run_root,
                tbl_rel=tbl_season,
                caption="Indicateurs de saisonnalité de la croissance naturelle",
                label="tab:tbl-desc-seasonality",
            ),
            narr_call("tbl.desc.seasonality"),
            "",
        ]

    # ----------------------------
    # Implications pour la suite
    # ----------------------------
    lines += [
        md_basic_to_tex("**Implications économétriques**"),
        md_basic_to_tex(
            "La présence d’une saisonnalité forte et évolutive implique que les tests de stationnarité "
            "et l’identification des modèles dynamiques doivent être conduits avec prudence. "
            "Une saisonnalité non traitée peut biaiser les tests de racine unitaire "
            "et fausser l’identification des ordres ARMA/ARIMA.\n\n"

            "La décomposition STL constitue ainsi une étape structurante : "
            "elle justifie le traitement explicite de la saisonnalité et prépare l’analyse formelle "
            "de la stationnarité développée dans la section suivante."
        ),
        "",
    ]

    return "\n".join(lines).strip() + "\n"
