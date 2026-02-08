from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from src.narrative.sections.base import SectionSpec, md_basic_to_tex, narr_call


def render_sec_anthropology(
    *,
    run_root: Path,
    manifest: Dict[str, Any],
    sec: SectionSpec,
    metrics_cache: Dict[str, Dict[str, Any]],
) -> str:
    # ---------- metrics ----------
    m = metrics_cache.get("m.anthro.todd_analysis") or {}
    md = ""
    if isinstance(m, dict):
        md = m.get("markdown") or m.get("text") or m.get("summary") or ""
    elif isinstance(m, str):
        md = m
    md = (md or "").replace("−", "-").strip()

    # optional: pull some upstream metrics if present (for coherence hints)
    tsds = metrics_cache.get("m.diag.ts_vs_ds") or {}
    uni = metrics_cache.get("m.uni.best") or {}
    coint = metrics_cache.get("m.coint.meta") or {}
    vecm_meta = metrics_cache.get("m.vecm.meta") or {}

    verdict = tsds.get("verdict", "NA")
    uni_kp = (uni.get("key_points") or {})
    uni_order = uni_kp.get("order") or (uni.get("best") or {}).get("order") or "NA"
    coint_choice = coint.get("choice") or "NA"
    coint_rank = coint.get("rank")
    coint_rank = coint_rank if coint_rank is not None else "NA"

    lines: list[str] = []

    # ============================================================
    # SECTION 1 : Cadre (hors-économétrie, lecture augmentée)
    # ============================================================
    lines += [
        r"\section{Analyse anthropologique augmentée par l’IA}",
        "",
        md_basic_to_tex(
            "L’économétrie fournit un cadre rigoureux d’identification des régularités statistiques et des relations dynamiques entre variables. "
            "Toutefois, elle demeure volontairement silencieuse sur les mécanismes sociaux profonds qui sous-tendent ces dynamiques. "
            "L’analyse anthropologique vise à combler cet espace interprétatif en proposant une lecture qualitative, non quantifiable, "
            "des résultats économétriques obtenus, sans jamais s’y substituer. "
            "Cette section s’inscrit explicitement hors du champ économétrique : aucun test, aucun modèle supplémentaire n’y est introduit. "
            "Elle repose sur l’exploitation raisonnée des résultats précédents, éclairés par les travaux d’Emmanuel Todd et par l’usage d’un agent IA "
            "comme outil d’exploration conceptuelle."
        ),
        "",
        r"\subsection*{8.1 Exploitation raisonnée des résultats économétriques}",
        md_basic_to_tex(
            "Les sections précédentes produisent des faits stylisés (inertie, persistance, relations de long terme, mécanismes d’ajustement). "
            "D’un point de vue strictement économétrique, ces éléments indiquent une dynamique structurée dans le temps long, "
            "et non une simple réaction conjoncturelle. "
            "La lecture anthropologique part de ces résultats (et seulement d’eux) pour proposer des hypothèses interprétatives."
        ),
        "",
        r"\subsection*{8.2 Identification des ruptures démographiques majeures}",
        md_basic_to_tex(
            "Les ruptures détectées (chocs persistants, changements de pente, modifications de trajectoire) ne sont pas uniquement des événements statistiques. "
            "Leur persistance peut refléter des facteurs sociaux et institutionnels profonds : transformations du modèle familial, normes de fécondité, "
            "politiques publiques, conditions sanitaires. "
            "L’économétrie identifie quand et comment ces ruptures se manifestent ; elle n’explique pas, à elle seule, leur nature sociologique."
        ),
        "",
        r"\subsection*{8.3 Mise en perspective avec les travaux d’Emmanuel Todd}",
        md_basic_to_tex(
            "Les travaux d’Emmanuel Todd soutiennent que certains indicateurs démographiques (ex. mortalité infantile) jouent le rôle de signaux sociaux latents, "
            "révélant des tensions profondes avant leur expression politique ou économique. "
            "Dans cette perspective, la démographie n’est pas un simple sous-produit de la conjoncture : elle devient un miroir anthropologique de l’état social."
        ),
        "",
        r"\subsection*{8.4 Application au cas français}",
        md_basic_to_tex(
            "Transposée au cas français, cette grille invite à interpréter la persistance de la croissance naturelle comme un reflet de structures sociales stables "
            "(modèle familial, inertie institutionnelle, État-providence). "
            "Inversement, une dégradation durable pourrait signaler une fragilisation latente du contrat social, même en l’absence de signaux macroéconomiques immédiats."
        ),
        "",
        r"\subsection*{8.5 Rôle de l’IA dans l’analyse anthropologique}",
        md_basic_to_tex(
            "L’agent IA n’est pas un producteur de vérité sociologique. Il sert d’outil d’exploration : mise en relation des résultats économétriques "
            "avec un corpus de connaissances, repérage de coïncidences temporelles entre ruptures et événements, formulation d’hypothèses non testables économétriquement. "
            "L’IA amplifie la réflexion ; elle ne remplace ni l’analyse scientifique ni la critique méthodologique."
        ),
        "",
        r"\subsection*{8.6 Hypothèses interprétatives non quantifiables}",
        md_basic_to_tex(
            "À partir de cette lecture croisée, on peut formuler des hypothèses : stabilité anthropologique relativement élevée, "
            "transmission des chocs économiques avec retard, rôle de politiques publiques durables, ou encore caractère précoce de certaines ruptures démographiques "
            "comme signaux sociaux. Ces hypothèses relèvent d’une démarche interprétative assumée."
        ),
        "",
        r"\subsection*{8.7 Séparation explicite entre économétrie et anthropologie}",
        md_basic_to_tex(
            "Les résultats économétriques sont falsifiables et reproductibles ; l’analyse anthropologique est qualitative et non falsifiable. "
            "Cette séparation garantit la rigueur : la lecture anthropologique n’a pas le droit de contredire les artefacts économétriques."
        ),
        "",
        r"\subsection*{8.8 Apport de l’approche augmentée}",
        md_basic_to_tex(
            "L’articulation économétrie–anthropologie évite une lecture techniciste, donne un sens social aux dynamiques de long terme, "
            "et justifie une intégration raisonnée de l’IA dans un projet académique."
        ),
        "",
    ]

    # ============================================================
    # SECTION 2 : Synthèse produite (métrique IA)
    # ============================================================
    lines += [
        r"\section{Synthèse anthropologique (Step7)}",
        "",
        md_basic_to_tex(
            f"Repères de cohérence (issus des métriques) : verdict stationnarité **{verdict}** ; "
            f"benchmark univarié **ARIMA{uni_order}** ; module long terme **{coint_choice}** (rang={coint_rank}). "
            "Ces repères ne sont pas des conclusions anthropologiques : ils cadrent la compatibilité avec les résultats économétriques."
        ),
        "",
    ]

    if md:
        lines += [
            md_basic_to_tex(md),
            narr_call("m.anthro.todd_analysis"),
            "",
        ]
    else:
        lines += [
            md_basic_to_tex("Aucune synthèse anthropologique disponible (métrique absente)."),
            "",
        ]

    # optional: reference vecm meta if present
    if vecm_meta:
        lines += [narr_call("m.vecm.meta"), ""]

    lines += [
        md_basic_to_tex("**Conclusion**"),
        md_basic_to_tex(
            "Cette section enrichit l’interprétation sans sur-interprétation : elle reste strictement contrainte par les faits économétriques "
            "matérialisés dans les figures, tables et métriques."
        ),
        "",
    ]

    return "\n".join(lines).strip() + "\n"
