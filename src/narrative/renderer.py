# src/narrative/renderer.py
from __future__ import annotations

from typing import Any


def render_anthropology(*, facts: dict[str, Any]) -> dict[str, Any]:
    """
    Offline, cadré, auditable.
    Entrée: facts = faits économétriques extraits depuis metrics de la run.
    Sortie: {"markdown": "...", "refs": [...], "meta": {...}}
      - refs: liste de labels d'artefacts/metrics qui justifient les faits
    """
    y = facts.get("y", "NA")

    # Faits stylisés (strict): uniquement keys présentes dans facts
    refs = facts.get("_refs", [])
    meta = {
        "y": y,
        "contract": "facts_only_from_metrics",
        "note": "Interprétation anthropologique séparée des faits économétriques.",
    }

    md: list[str] = []
    md.append("# Analyse anthropologique (cadrée, offline)")
    md.append("")
    md.append("## Faits économétriques (issus des métriques du run)")
    md.append("Ces points reprennent uniquement des valeurs présentes dans les métriques persistées.")
    md.append("")

    # Affichage stable, sans bruit interne
    for k, v in facts.items():
        if k.startswith("_"):
            continue
        md.append(f"- **{k}** : `{v}`")

    md.append("")
    md.append("## Hypothèses anthropologiques (cadre Todd, non-testé économétriquement)")
    md.append("- Les ruptures/variations mises en évidence sont traitées comme **faits stylisés** (pas de causalité démontrée).")
    md.append("- Cadre Todd (lecture longue) : normes familiales, structures de reproduction sociale, cycles longs comme **grille** d’interprétation.")
    md.append("- Toute hypothèse reste **conditionnelle** aux artefacts et ne remplace pas une identification causale.")
    md.append("")
    md.append("## Traçabilité")
    if refs:
        md.append("Références (labels d’artefacts/metrics) :")
        for r in refs:
            md.append(f"- `{r}`")
    else:
        md.append("Aucune référence explicite trouvée : la run ne contient pas (encore) de métriques exploitables pour l’analyse.")

    return {"markdown": "\n".join(md), "refs": refs, "meta": meta}
