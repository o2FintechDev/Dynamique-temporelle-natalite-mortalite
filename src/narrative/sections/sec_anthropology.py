# src/narrative/sections/sec_anthropology.py
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

    # =======================================================
    # SECTION 1 : Cadre (hors-économétrie, lecture augmentée)
    # =======================================================
    lines += [
        r"\section{Analyse anthropologique augmentée par l’IA}",
        "",
        md_basic_to_tex(
            "L’économétrie permet d’identifier avec rigueur des régularités statistiques, des ruptures "
            "et des dynamiques de long terme à partir de données observables. "
            "Elle ne vise cependant pas à épuiser l’interprétation des phénomènes sociaux sous-jacents. "
            "La présente section s’inscrit volontairement hors du champ économétrique et propose "
            "une lecture anthropologique augmentée des résultats obtenus, sans jamais s’y substituer. "
            "Aucun test supplémentaire n’est introduit, aucun paramètre n’est estimé, "
        "et aucune causalité n’est postulée."
        ),
        "",
        r"\subsection*{Définition du cadre anthropologique}",
        md_basic_to_tex(
            "Le terme « anthropologique » est employé ici au sens des sciences sociales, "
            "désignant l’étude des structures profondes et relativement stables "
            "qui organisent les comportements collectifs : normes familiales, "
            "rapport à la reproduction, projection intergénérationnelle, "
            "relation au temps long et aux institutions. "
            "L’anthropologie, dans ce cadre, ne cherche pas à mesurer directement, "
            "mais à interpréter des régularités sociales durables."
        ),
        "",

        md_basic_to_tex(
            "Appliquée à ce projet, l’anthropologie ne constitue pas une discipline concurrente "
            "de l’économétrie, mais un niveau de lecture complémentaire. "
            "Elle vise à donner sens aux dynamiques démographiques observées "
            "en les replaçant dans un contexte social, culturel et institutionnel "
            "qui dépasse la seule logique économique de court terme."
        ),
        "",
        r"\subsection*{Inspiration méthodologique des travaux d’Emmanuel Todd}",
        md_basic_to_tex(
            "Cette démarche s’inspire méthodologiquement des travaux d’Emmanuel Todd, "
            "notamment de ses analyses démographiques menées à partir de la fin des années 1970. "
            "L’apport central de ces travaux réside dans l’utilisation d’indicateurs démographiques "
            "comme signaux sociaux latents, révélateurs de déséquilibres profonds "
            "avant leur expression politique ou économique. "
            "Il ne s’agit pas de transposer des contextes historiques, "
            "mais d’adopter une logique d’analyse fondée sur la démographie comme révélateur structurel."
        ),
        "",
        r"\subsection*{Exploitation raisonnée des résultats économétriques}",
        md_basic_to_tex(
            "Les résultats économétriques précédents mettent en évidence "
            "une persistance temporelle élevée, des ruptures durables "
            "et une dynamique de long terme structurée de la croissance naturelle. "
            "Ces faits stylisés indiquent que la série ne réagit pas uniquement "
            "à des chocs conjoncturels, mais s’inscrit dans une trajectoire profonde. "
            "La lecture anthropologique prend ces résultats comme point de départ exclusif "
            "et ne mobilise aucune information incompatible avec les diagnostics établis."
        ),
        "",
        r"\subsection*{Variables non quantifiables et signaux sociaux latents}",
        md_basic_to_tex(
            "L’analyse anthropologique introduit explicitement des dimensions "
            "non quantifiables au sens économétrique : confiance collective dans l’avenir, "
            "stabilité perçue du modèle familial, normes de fécondité, "
            "rapport à la transmission intergénérationnelle, "
            "ou encore légitimité ressentie des institutions. "
            "Ces variables ne sont pas observables directement, "
            "mais peuvent se manifester indirectement à travers "
            "des évolutions démographiques persistantes."
        ),
        "",

        md_basic_to_tex(
            "Dans cette perspective, la croissance naturelle est interprétée "
            "comme un indicateur synthétique de l’état anthropologique d’une société. "
            "Une dégradation durable peut signaler une fragilisation latente, "
            "indépendante de la performance économique immédiate, "
            "tandis qu’une inertie relative peut refléter "
            "la stabilité de structures sociales profondes."
        ),
        "",
        r"\subsection*{Rôle et encadrement méthodologique de l’agent IA (ChatGPT 5.2)}",
        md_basic_to_tex(
            "Le terme d’« agent IA » est employé au sens fonctionnel et non autonome. "
            "ChatGPT 5.2 est utilisé comme un agent d’assistance interprétative, "
            "destiné à structurer et enrichir la réflexion anthropologique "
            "à partir des résultats économétriques produits en amont."
        ),
        "",

        md_basic_to_tex(
            "Afin de limiter tout biais ou hallucination, son utilisation est strictement encadrée "
            "par une méthode scientifique explicite : "
            "les entrées de l’agent sont exclusivement constituées de résultats économétriques validés, "
            "les sorties sont systématiquement contraintes par ces résultats, "
            "et toute interprétation proposée est évaluée au regard de sa compatibilité "
            "avec les faits stylisés observés. "
            "Aucune information non corroborée par le cadre du projet n’est intégrée."
        ),
        "",

        md_basic_to_tex(
            "L’agent IA agit ainsi comme un amplificateur cognitif contrôlé : "
            "il facilite l’exploration conceptuelle des liens possibles "
            "entre dynamiques démographiques et structures sociales, "
            "sans jamais produire de validation empirique ni d’assertion causale."
        ),
        "",
        r"\subsection*{Séparation épistémologique et limites assumées}",
        md_basic_to_tex(
            "Une séparation stricte est maintenue entre résultats économétriques et lecture anthropologique. "
            "Les premiers sont falsifiables, reproductibles et quantifiables ; "
            "la seconde est interprétative, qualitative et non falsifiable. "
            "Cette distinction garantit la rigueur scientifique du projet "
            "et interdit toute contradiction entre interprétation anthropologique "
            "et faits économétriques établis."
        ),
        "",
    ]

    # ===========================================
    # SECTION 2 : Synthèse produite (métrique IA)
    # ===========================================
    lines += [
        r"\section{Interprétation anthropologique des dynamiques démographiques}",
        "",
        md_basic_to_tex(
            f"Les repères économétriques suivants cadrent strictement la lecture anthropologique : "
            f"verdict de stationnarité **{verdict}**, "
            f"benchmark univarié **ARIMA (1,1,4)**, "
            f"et structure de long terme **{coint_choice}** (rang={coint_rank}). "
            "Ces éléments ne constituent pas des conclusions anthropologiques, "
            "mais définissent l’espace interprétatif admissible."
        ),
        "",
        md_basic_to_tex(
            "À partir de ces contraintes, la synthèse anthropologique vise à qualifier "
            "l’état profond de la dynamique démographique française. "
            "Elle ne cherche ni à prévoir quantitativement, ni à expliquer causalement, "
            "mais à identifier des configurations sociales cohérentes "
            "avec les régularités observées."
        ),
        "",
    ]

    
    lines += [
        md_basic_to_tex(
            "En l’absence de synthèse automatisée, la lecture anthropologique "
            "repose exclusivement sur l’analyse conceptuelle développée dans cette section, "
            "contrainte par les résultats économétriques et par le cadre méthodologique défini."
        ),
        "",
    ]

    lines += [
        md_basic_to_tex("**Conclusion**"),
        md_basic_to_tex(
            "Cette synthèse anthropologique augmentée apporte une plus-value interprétative "
            "au projet économétrique en intégrant explicitement des dimensions non quantifiables "
            "habituellement absentes des modèles statistiques. "
            "Elle assume l’usage encadré de l’IA comme outil méthodologique, "
            "sans remettre en cause la primauté des résultats économétriques. "
            "L’approche proposée ne vise pas la prédiction, "
            "mais la qualification rigoureuse des fragilités et inerties sociales "
            "associées à la dynamique de la croissance naturelle française."
        ),
        "",
    ]

    return "\n".join(lines).strip() + "\n"
