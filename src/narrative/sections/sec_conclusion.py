# src/narrative/sections/sec_conclusion.py

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from src.narrative.sections.base import SectionSpec, md_basic_to_tex, narr_call


def render_sec_conclusion(
    *,
    run_root: Path,
    manifest: Dict[str, Any],
    sec: SectionSpec,
    metrics_cache: Dict[str, Dict[str, Any]],
) -> str:
    # ---------- métriques clés (si disponibles, pour cohérence) ----------
    tsds = metrics_cache.get("m.diag.ts_vs_ds") or {}
    uni = metrics_cache.get("m.uni.best") or {}
    var_meta = metrics_cache.get("m.var.meta") or {}
    coint = metrics_cache.get("m.coint.meta") or {}

    verdict = tsds.get("verdict", "NA")
    uni_order = (
        (uni.get("key_points") or {}).get("order")
        or (uni.get("best") or {}).get("order")
        or "NA"
    )
    coint_choice = coint.get("choice", "NA")
    coint_rank = coint.get("rank")
    coint_rank = coint_rank if coint_rank is not None else "NA"

    lines: list[str] = []

    # ============================================================
    # CONCLUSION GÉNÉRALE
    # ============================================================
    lines += [
        r"\section*{Conclusion}",
        "",
        md_basic_to_tex(
            "Ce rapport a proposé une analyse économétrique approfondie de la croissance naturelle de la population française "
            "sur la période 1975–2025, en mobilisant un cadre méthodologique entièrement automatisé. "
            "L’objectif principal était double : caractériser rigoureusement la dynamique démographique de long terme "
            "et démontrer l’intérêt d’un automate économétrique déterministe pour la production de résultats reproductibles."
        ),
        "",
        md_basic_to_tex(
            "L’analyse descriptive et les diagnostics statistiques ont mis en évidence une trajectoire marquée par des tendances persistantes, "
            "une saisonnalité significative et des ruptures potentielles, incompatibles avec une lecture purement conjoncturelle. "
            "Les tests de stationnarité et les modèles dynamiques ont confirmé que la croissance naturelle française "
            "est gouvernée par des mécanismes structurels de long terme, sensibles aux chocs mais dotés d’une inertie élevée."
        ),
        "",
        md_basic_to_tex(
            "La modélisation univariée et multivariée a permis d’évaluer la persistance des chocs démographiques "
            "et d’explorer les relations dynamiques entre les composantes du système. "
            "Ces résultats suggèrent que les évolutions récentes observées — notamment l’inversion du solde naturel — "
            "ne relèvent pas de fluctuations transitoires, mais traduisent un changement profond du processus générateur des données."
        ),
        "",
        md_basic_to_tex(
            "Sur le plan méthodologique, ce travail illustre l’apport d’un pipeline économétrique automatisé, "
            "capable de structurer l’ensemble de la chaîne d’analyse : préparation des données, diagnostics, modélisation, "
            "production d’artefacts et interprétation. "
            "L’automatisation renforce la traçabilité des choix, limite les erreurs manuelles "
            "et garantit la reproductibilité complète des résultats."
        ),
        "",
        md_basic_to_tex(
            "L’intégration contrôlée de l’intelligence artificielle s’est limitée à un rôle d’assistance à l’interprétation "
            "et à la structuration du raisonnement, sans jamais se substituer aux décisions économétriques fondamentales. "
            "Cette approche permet de concilier innovation technologique et rigueur scientifique."
        ),
        "",
        md_basic_to_tex(
            "Enfin, la mise en perspective anthropologique a montré que les résultats économétriques peuvent être éclairés "
            "par une lecture qualitative des transformations sociales et institutionnelles, "
            "à condition de maintenir une séparation stricte entre faits statistiques et interprétation."
        ),
        "",
        md_basic_to_tex(
            "Les limites du travail tiennent principalement au périmètre des variables explicatives retenues "
            "et à l’analyse centrée sur un seul pays. "
            "Des prolongements naturels consisteraient à intégrer davantage de déterminants économiques et sociaux, "
            "ou à étendre l’approche à une comparaison internationale."
        ),
        "",
        md_basic_to_tex(
            "En définitive, ce projet démontre que l’automatisation raisonnée de l’économétrie "
            "constitue un outil puissant pour l’analyse des dynamiques démographiques de long terme, "
            "à condition qu’elle reste guidée par une réflexion méthodologique explicite et une interprétation critique des résultats."
        ),
        "",
        md_basic_to_tex(
            f"**Repères de cohérence (issus des métriques)** : "
            f"stationnarité **{verdict}**, "
            f"modèle univarié de référence **ARIMA{uni_order}**, "
            f"analyse de long terme **{coint_choice}** (rang = {coint_rank}). "
            "Ces éléments synthétisent les résultats sans en constituer une nouvelle analyse."
        ),
        "",
    ]

    return "\n".join(lines).strip() + "\n"
