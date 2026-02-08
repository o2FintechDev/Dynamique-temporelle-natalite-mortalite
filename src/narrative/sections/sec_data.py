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

    # ============================================================
    # SECTION 1 : Contexte & construction (texte long)
    # ============================================================
    lines += [
        r"\section{Contexte et construction de la variable}",
        "",
        md_basic_to_tex(
            "Toute analyse économétrique rigoureuse repose sur une étape préliminaire fondamentale : "
            "la construction correcte des variables et la maîtrise des propriétés statistiques des données mobilisées. "
            "En démographie, cette exigence est renforcée par la nature institutionnelle des sources, "
            "la dépendance aux méthodes de recensement et la présence potentielle de ruptures méthodologiques non économiques.\n\n"
            "Dans ce projet, l’objet d’étude est la croissance naturelle de la population française sur la période 1975–2025. "
            "Cette variable constitue un solde démographique fondamental, reflétant la dynamique interne de la population "
            "indépendamment des flux migratoires. Son analyse permet d’isoler des mécanismes structurels de long terme, "
            "fortement liés aux comportements familiaux, aux politiques publiques et aux conditions socio-économiques générales."
        ),
        "",
        r"\subsection*{1.1 Choix des données et justification des sources}",
        md_basic_to_tex(
            "Les données utilisées proviennent exclusivement de l’INSEE, garantissant une homogénéité institutionnelle et une comparabilité "
            "temporelle sur l’ensemble de la période étudiée. Les séries mobilisées sont :\n\n"
            "— le nombre mensuel de naissances,\n"
            "— le nombre mensuel de décès,\n"
            "— la population totale moyenne mensuelle.\n\n"
            "Le choix d’une fréquence mensuelle s’impose pour capturer les dynamiques de moyen/long terme tout en conservant la saisonnalité."
        ),
        "",
        r"\subsection*{1.2 Problématique de l’échelle et choix des taux}",
        md_basic_to_tex(
            "Les flux en niveau (naissances/décès) sont mécaniquement corrélés à la taille de la population, induisant une hétéroscédasticité "
            "structurelle et un risque de conclusions biaisées en stationnarité. La normalisation en taux vise à stabiliser la variance et à rendre "
            "les séries comparables."
        ),
        "",
        r"\begin{equation}",
        r"\text{Taux de natalité}_t = \frac{\text{Naissances}_t}{\text{Population}_t} \times 1000",
        r"\end{equation}",
        r"\begin{equation}",
        r"\text{Taux de mortalité}_t = \frac{\text{Décès}_t}{\text{Population}_t} \times 1000",
        r"\end{equation}",
        "",
        r"\subsection*{1.3 Définition formelle de la croissance naturelle}",
        "",
        r"\begin{equation}",
        r"\text{Croissance naturelle}_t = \text{Taux de natalité}_t - \text{Taux de mortalité}_t",
        r"\end{equation}",
        "",
        md_basic_to_tex(
            "La variable synthétique combine deux processus interdépendants et peut présenter persistance, saisonnalité et ruptures. "
            "Une croissance négative signale un vieillissement structurel susceptible d’affecter la soutenabilité des systèmes sociaux."
        ),
        "",
        r"\subsection*{1.4 Cohérence temporelle et continuité méthodologique}",
        md_basic_to_tex(
            "Toute rupture méthodologique non identifiée peut être confondue avec une rupture structurelle. "
            "L’hypothèse retenue est une continuité suffisante des méthodes INSEE sur la période, à expliciter pour cadrer l’interprétation des tests."
        ),
        "",
        r"\subsection*{1.5 Gestion des valeurs manquantes et anomalies}",
        md_basic_to_tex(
            "Les manquants/anomalies créent des sauts artificiels et perturbent tests de racine unitaire et diagnostics résiduels. "
            "La stratégie est conservatrice : identification, traçabilité, refus d’interpolations agressives."
        ),
        "",
        r"\subsection*{1.6 Implications économétriques pour la suite}",
        md_basic_to_tex(
            "Une variable mal construite entraîne erreurs de stationnarité, persistance surévaluée et cointégrations fallacieuses. "
            "Le reste du rapport est conditionné par la qualité de cette étape."
        ),
        "",
    ]

    # ============================================================
    # SECTION 2 : Préparation des données (Step1) + tables sans subsection
    # ============================================================
    lines += [
        r"\section{Préparation des données}",
        "",
        md_basic_to_tex(
            f"Les diagnostics de préparation résument l’échantillon effectivement exploitable : "
            f"**{start} -> {end}**, fréquence **{freq}**, **n={nobs}**, taux de manquants **{miss_txt}**."
        ),
        narr_call("m.data.dataset_meta"),
        "",
    ]

    # --- Table 1: descriptives + analyse
    if t_desc:
        lines += [
            r"\paragraph{Tableau 1 — Statistiques descriptives}",
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
            r"\paragraph{Tableau 2 — Valeurs manquantes}",
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
            r"\paragraph{Tableau 3 — Couverture temporelle}",
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
            md_basic_to_tex("**Synthèse automatisée (Step1)**"),
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
        md_basic_to_tex("**Conclusion**"),
        md_basic_to_tex(
            "Les tables de préparation déterminent la qualité du signal exploitable. "
            "Elles bornent les choix méthodologiques des sections suivantes (stationnarité, ARIMA, VAR/VECM) "
            "et cadrent l’interprétation des ruptures et chocs."
        ),
        "",
    ]

    return "\n".join(lines).strip() + "\n"
