# src/narrative/sections/sec_data.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from src.narrative.sections.base import (
    SectionSpec,
    lookup,
    md_basic_to_tex,
    include_table_tex,
    narr_call,
)

# ---------- formatting ----------
def _as_percent(x: Any) -> str:
    try:
        v = float(x)
        return f"{100.0 * v:.2f}\\%"
    except Exception:
        return "NA"


def render_sec_data(
    *,
    run_root: Path,
    manifest: Dict[str, Any],
    sec: SectionSpec,
    metrics_cache: Dict[str, Dict[str, Any]],
) -> str:
    # ---------- Step1 metrics ----------
    meta = metrics_cache.get("m.data.dataset_meta") or {}
    note = metrics_cache.get("m.note.step1") or {}

    freq = meta.get("freq") or meta.get("frequency") or "NA"
    start = meta.get("start") or meta.get("start_date") or "NA"
    end = meta.get("end") or meta.get("end_date") or "NA"
    nobs = meta.get("nobs") or meta.get("n") or "NA"
    miss_rate = meta.get("missing_rate")
    miss_txt = _as_percent(miss_rate)

    # Note Step1 (markdown)
    note_md = ""
    if isinstance(note, dict):
        note_md = note.get("markdown") or note.get("text") or note.get("summary") or ""
    elif isinstance(note, str):
        note_md = note
    note_md = note_md.replace("−", "-")

    # ---------- Step1 artefacts (tables) ----------
    t_desc = lookup(manifest, "tables", "tbl.data.desc_stats")
    t_miss = lookup(manifest, "tables", "tbl.data.missing_report")
    t_cov = lookup(manifest, "tables", "tbl.data.coverage_report")

    lines: list[str] = []


    # Introduction

    # ============================================================
    # SECTION 1 : Coeur de la recherche
    # ============================================================
    lines += [
    r"\section{Cœur de la recherche}",
    "",
    md_basic_to_tex(
    "Cette recherche s’inscrit dans l’analyse des dynamiques démographiques de long terme et prend pour objet central "
    "la croissance naturelle de la population française sur la période 1975–2025. "
    "Au-delà d’un indicateur statistique, la croissance naturelle constitue un révélateur profond des transformations "
    "sociales, économiques et sanitaires qui traversent une société. "
    "Définie comme la différence entre le taux de natalité et le taux de mortalité, "
    "elle permet d’isoler la dynamique interne de la population, indépendamment des mouvements migratoires.\n\n"

    "Pendant plusieurs décennies, la France s’est distinguée par une vitalité démographique relative au sein des pays développés, "
    "souvent présentée comme une exception européenne. "
    "Toutefois, l’inversion récente entre les courbes de natalité et de mortalité, "
    "avec un solde naturel mensuel devenu négatif à partir de 2023, "
    "marque une rupture historique majeure. "
    "Ce basculement interroge directement la trajectoire démographique française et soulève des enjeux majeurs "
    "en matière de financement des systèmes sociaux, de marché du travail et de politiques publiques.\n\n"

    "L’enjeu scientifique de ce travail est de dépasser le simple constat statistique pour analyser la nature profonde "
    "de cette trajectoire démographique. "
    "Il s’agit de déterminer si la croissance naturelle française suit une dynamique stable ou instable, "
    "si les chocs démographiques observés ont des effets transitoires ou persistants, "
    "et si les ruptures récentes traduisent un changement de régime durable. "
    "Cette analyse mobilise l’économétrie des séries temporelles afin de relier les évolutions observées "
    "aux mécanismes structurels sous-jacents.\n\n"

    "L’utilisation de données mensuelles sur un horizon long permet de capter finement les phénomènes de saisonnalité, "
    "les effets de mémoire et les dynamiques de long terme. "
    "Cette granularité temporelle est essentielle pour analyser l’impact des chocs exogènes majeurs — "
    "crises économiques, pandémies ou événements climatiques extrêmes — "
    "sur la démographie française. "
    "Elle autorise également l’estimation de modèles économétriques avancés, "
    "tels que les modèles ARIMA, ARFIMA, VAR et VECM, adaptés à l’étude conjointe du court et du long terme.\n\n"

    "Enfin, ce travail s’inscrit dans une démarche méthodologique innovante reposant sur un automate économétrique déterministe. "
    "Cet outil permet de produire de manière reproductible l’ensemble des analyses, graphiques et interprétations du rapport. "
    "L’intelligence artificielle y est mobilisée comme outil d’assistance à la structuration du raisonnement et à l’interprétation, "
    "sans jamais se substituer aux choix économétriques fondamentaux. "
    "Cette articulation maîtrisée entre automatisation, rigueur statistique et lecture interprétative "
    "constitue le cœur scientifique et méthodologique de la recherche."
    ),
    "",
        r"\subsection*{Choix des données}",
        md_basic_to_tex(
            "Les données utilisées proviennent exclusivement de l’INSEE (Institut national de la statistique et des études économiques), garantissant une homogénéité institutionnelle "
            "et une comparabilité temporelle sur l’ensemble de la période étudiée. "
            "Le recours à une source statistique unique permet de limiter les biais liés aux ruptures de méthode ou aux changements de définition. "
            "Les séries mobilisées sont :\n\n"
            "— le nombre mensuel de naissances,\n"
            "— le nombre mensuel de décès,\n"
            "— la population totale moyenne mensuelle.\n\n"
            "Le choix d’une fréquence mensuelle s’impose afin de capturer à la fois les dynamiques de long terme "
            "et les variations saisonnières propres aux phénomènes démographiques."
        ),
        "",
        r"\subsection*{Problématique de l’échelle et choix des taux}",
        md_basic_to_tex(
            "Les flux démographiques exprimés en niveau (naissances et décès) sont mécaniquement liés à la taille de la population, "
            "ce qui induit une hétéroscédasticité structurelle et complique l’analyse économétrique. "
            "Une hausse ou une baisse des flux peut ainsi refléter une simple variation de la population totale, "
            "sans traduire une modification réelle des comportements démographiques.\n\n"

            "La normalisation en taux permet de neutraliser cet effet d’échelle, "
            "de stabiliser la variance et de rendre les séries comparables dans le temps. "
            "Au-delà de cet intérêt statistique, l’analyse en taux offre une lecture plus pertinente sur le plan interprétatif : "
            "elle permet d’appréhender les comportements démographiques relatifs, "
            "indépendamment de la taille absolue de la population.\n\n"

            "Interpréter les évolutions en taux plutôt qu’en niveau revient ainsi à raisonner en termes d’intensité démographique "
            "plutôt qu’en volumes bruts. "
            "Cette approche facilite les comparaisons temporelles et met en évidence des transformations structurelles "
            "des comportements de fécondité et de mortalité, "
            "qui pourraient être masquées par une analyse exclusivement fondée sur les niveaux."
        ),
        "",
        r"\begin{equation}",
        r"\text{Taux de natalité}_t = \frac{\text{Naissances}_t}{\text{Population}_t} \times 1000",
        r"\end{equation}",
        r"\begin{equation}",
        r"\text{Taux de mortalité}_t = \frac{\text{Décès}_t}{\text{Population}_t} \times 1000",
        r"\end{equation}",
        "",
        r"\subsection*{Définition formelle de la croissance naturelle}",
        "",
        r"\begin{equation}",
        r"\text{Croissance naturelle}_t = \text{Taux de natalité}_t - \text{Taux de mortalité}_t",
        r"\end{equation}",
        "",
        md_basic_to_tex(
            "La croissance naturelle résulte de l’interaction entre deux processus démographiques fondamentaux. "
            "Cette variable peut présenter des phénomènes de persistance, de saisonnalité et de ruptures structurelles. "
            "Une croissance durablement négative constitue un signal fort de vieillissement démographique, "
            "susceptible d’affecter en profondeur l’équilibre économique et social."
        ),
        "",

    ]

    # ============================================================
    # SECTION 2 : Préparation des données
    # ============================================================
    lines += [
    r"\section{Préparation des données}",
    "",
    md_basic_to_tex(
        f"La présente section décrit les étapes de préparation des données ayant conduit à la constitution "
        f"de l’échantillon exploitable pour l’analyse économétrique. "
        f"L’étude repose sur des données mensuelles couvrant la période **{start} -> {end}**, "
        f"à la fréquence **{freq}**, pour un total de **n = {nobs}** observations.\n\n"

        "Avant toute analyse statistique, un soin particulier a été apporté à la qualité et à la fiabilité des données. "
        "L’ensemble des séries a été préparé selon un processus structuré de type ETL "
        "(extraction, transformation et chargement), "
        "visant à garantir la cohérence temporelle, l’homogénéité des définitions et la traçabilité complète des traitements appliqués.\n\n"

        "Les données utilisées proviennent exclusivement de sources institutionnelles (INSEE) "
        "et ne présentent aucune valeur manquante sur la période considérée. "
        f"Le taux de données manquantes observé est ainsi de **{miss_txt}**, "
        "ce qui assure une continuité temporelle complète des séries et renforce la robustesse "
        "des tests économétriques ultérieurs.\n\n"

        "Le processus de préparation a permis de vérifier l’intégrité des séries, "
        "d’identifier d’éventuelles incohérences ou anomalies ponctuelles, "
        "et d’harmoniser les formats et unités de mesure. "
        "La stratégie retenue est volontairement conservatrice : "
        "aucune interpolation ou correction artificielle n’a été appliquée, "
        "afin de préserver la dynamique propre des séries temporelles et d’éviter l’introduction de biais exogènes.\n\n"

        "Les diagnostics présentés dans les tableaux suivants constituent une étape de validation empirique essentielle. "
        "Ils permettent de confirmer la qualité des données avant d’engager l’analyse de stationnarité, "
        "la modélisation économétrique et l’étude des dynamiques de court et de long terme."
    ),
    narr_call("m.data.dataset_meta"),
    "",
    ]

    # --- Table 1: descriptives + analyse
    if t_desc:
        lines += [
            r"\subsection*{Tableau 1 — Statistiques descriptives}",
            md_basic_to_tex(
                "Lecture : contrôler l’ordre de grandeur, l’asymétrie et la dispersion. "
                "Des extrêmes prononcés ou une distribution très dissymétrique sont cohérents avec des chocs (épidémiques, caniculaires) "
                "et imposent de vérifier la robustesse des tests et diagnostics."
            ),
            "",
            include_table_tex(
                run_root=run_root,
                tbl_rel=t_desc,
                caption="tbl.data.desc_stats",
                label="tab:tbl-data-desc-stats",
            ),
            narr_call("tbl.data.desc_stats"),
            "",
        ]

    # --- Table 2: missing report + analyse
    if t_miss:
        lines += [
            r"\subsection*{Tableau 2 — Valeurs manquantes}",
            md_basic_to_tex(
                "Lecture : même un faible taux de manquants peut biaiser ADF/Ljung–Box si les trous sont concentrés temporellement "
                "(rupture de collecte, anomalies de source). La règle est : documenter et éviter de lisser artificiellement."
            ),
            "",
            include_table_tex(
                run_root=run_root,
                tbl_rel=t_miss,
                caption="tbl.data.missing_report",
                label="tab:tbl-data-missing-report",
            ),
            narr_call("tbl.data.missing_report"),
            "",
        ]

    # --- Table 3: coverage report + analyse
    if t_cov:
        lines += [
            r"\subsection*{Tableau 3 — Couverture temporelle}",
            md_basic_to_tex(
                "Lecture : valider la continuité de l’index, la présence éventuelle de périodes incomplètes, et la cohérence du début/fin d’échantillon. "
                "Toute discontinuité non traitée se répercute sur la dynamique (ACF/PACF), les résidus et la détection de ruptures."
            ),
            "",
            include_table_tex(
                run_root=run_root,
                tbl_rel=t_cov,
                caption="tbl.data.coverage_report",
                label="tab:tbl-data-coverage-report",
            ),
            narr_call("tbl.data.coverage_report"),
            "",
        ]

    # --- Note Step1 (optionnelle)
    if note_md.strip():
        lines += [
            r"\textbf{Synthèse automatisée}",
            md_basic_to_tex(
                "Cette note sert d’audit : elle doit rester cohérente avec les trois diagnostics ci-dessus. "
                "Toute mention de correction/interpolation doit être explicitée et traçable."
            ),
            "",
            md_basic_to_tex(note_md),
            narr_call("m.note.step1"),
            "",
        ]

    # Conclusion
    lines += [
        r"\textbf{Conclusion}",
        md_basic_to_tex(
            "Les tables de préparation déterminent la qualité du signal exploitable. "
            "Elles bornent les choix méthodologiques des sections suivantes (stationnarité, ARIMA, VAR/VECM) "
            "et cadrent l’interprétation des ruptures et chocs."
        ),
        "",
    ]

    return "\n".join(lines).strip() + "\n"
