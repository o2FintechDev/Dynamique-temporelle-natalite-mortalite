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
        
        md_basic_to_tex(
            "Ce travail avait pour objectif d’analyser la dynamique de la croissance naturelle de la population française "
            "sur la période 1975–2025, en mobilisant un cadre économétrique rigoureux, reproductible et automatisé. "
            "L’ambition n’était pas de produire une prévision démographique, "
            "mais de qualifier la nature du processus démographique observé et d’en apprécier la profondeur temporelle."
        ),
        "",
        md_basic_to_tex(
            "La problématique centrale posée en introduction était la suivante : "
            "la dégradation récente de la croissance naturelle en France relève-t-elle "
            "d’un phénomène conjoncturel et transitoire, ou traduit-elle une transformation "
            "structurelle profonde et durable des dynamiques démographiques et sociales ? "
            "L’ensemble des analyses conduites permet désormais d’apporter une réponse argumentée à cette question."
        ),
        "",
        md_basic_to_tex(
            "Les résultats obtenus montrent que l’évolution récente de la croissance naturelle "
            "ne peut pas être interprétée comme une simple fluctuation de court terme. "
            "La dynamique observée s’inscrit dans un mouvement de long terme, "
            "caractérisé par une forte inertie et par des ruptures persistantes. "
            "Les chocs démographiques identifiés ne disparaissent pas rapidement, "
            "mais s’intègrent durablement dans la trajectoire de la série."
        ),
        "",
        md_basic_to_tex(
            "Autrement dit, la dégradation de la croissance naturelle apparaît moins comme un accident ponctuel "
            "que comme l’expression d’une transformation plus profonde du processus démographique français. "
            "Cette transformation ne signifie pas une rupture brutale ou irréversible, "
            "mais elle indique que les mécanismes sous-jacents à la dynamique démographique "
            "ont évolué de manière durable au cours de la période étudiée."
        ),
        "",
        md_basic_to_tex(
            "Sur le plan méthodologique, ce travail met en évidence l’intérêt d’un automate économétrique déterministe "
            "pour structurer l’analyse des séries temporelles de long terme. "
            "L’automatisation n’a pas pour fonction de remplacer l’analyse économique, "
            "mais de garantir la cohérence des diagnostics, la traçabilité des choix méthodologiques "
            "et la reproductibilité complète des résultats."
        ),
        "",
        md_basic_to_tex(
            "L’intelligence artificielle a été mobilisée de manière encadrée et prudente, "
            "exclusivement comme outil d’assistance à la structuration du raisonnement "
            "et à la mise en cohérence des interprétations. "
            "Elle n’a joué aucun rôle décisionnel et n’a introduit ni hypothèse ni résultat "
            "qui ne soit déjà fondé sur l’analyse économétrique."
        ),
        "",
        md_basic_to_tex(
            "La lecture anthropologique proposée en fin de rapport ne constitue pas une explication alternative, "
            "mais un prolongement interprétatif des résultats obtenus. "
            "Elle permet de replacer la dynamique de la croissance naturelle "
            "dans un cadre plus large, en lien avec des transformations sociales, "
            "institutionnelles et culturelles de long terme, "
            "tout en respectant strictement les faits établis par l’analyse statistique."
        ),
        "",
        md_basic_to_tex(
            "Les limites de ce travail tiennent principalement au périmètre retenu : "
            "l’analyse est centrée sur un seul pays et sur un nombre restreint de variables explicatives. "
            "Des prolongements naturels consisteraient à enrichir le cadre multivarié "
            "ou à étendre l’approche à une comparaison internationale, "
            "afin de mieux situer la trajectoire française dans un contexte plus large."
        ),
        "",
        md_basic_to_tex(
            "En définitive, ce rapport montre que la croissance naturelle constitue un indicateur central "
            "pour comprendre les dynamiques démographiques de long terme. "
            "L’approche proposée, combinant économétrie des séries temporelles, automatisation raisonnée "
            "et lecture interprétative encadrée, "
            "offre un cadre robuste pour analyser des phénomènes démographiques complexes "
            "sans céder ni au déterminisme statistique ni à la sur-interprétation."
        ),
        "",
    ]

    return "\n".join(lines).strip() + "\n"
