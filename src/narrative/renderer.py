from __future__ import annotations
from typing import Iterable
from src.agent.schemas import Artefact
from src.narrative.rules import Sentence, render_sentence, audit_narrative

def build_mvp_narrative(artefacts: list[Artefact]) -> tuple[str, dict]:
    ids = {a.artefact_id for a in artefacts}

    # MVP: synthèse factuelle sur coverage + harmonisation + présence figures/stats
    has_cov = next((a for a in artefacts if a.name == "coverage_report"), None)
    has_desc = next((a for a in artefacts if a.name == "describe_stats"), None)
    has_hmeta = next((a for a in artefacts if a.name == "harmonize_meta"), None)
    figs = [a for a in artefacts if a.kind == "figure"]

    sentences: list[Sentence] = []
    if has_hmeta:
        sentences.append(Sentence(
            text="Les dates ont été normalisées en index mensuel month-start (MS) et la grille temporelle complète a été construite.",
            artefact_ids=[has_hmeta.artefact_id],
        ))
    if has_cov:
        sentences.append(Sentence(
            text="Le Data Coverage Report est disponible (périodes par variable, volumes et taux de valeurs manquantes).",
            artefact_ids=[has_cov.artefact_id],
        ))
    if figs:
        sentences.append(Sentence(
            text=f"{len(figs)} graphiques de séries en niveau ont été générés pour l’exploration initiale.",
            artefact_ids=[figs[0].artefact_id],
        ))
    if has_desc:
        sentences.append(Sentence(
            text="Les statistiques descriptives de base (describe) ont été calculées pour les variables sélectionnées.",
            artefact_ids=[has_desc.artefact_id],
        ))

    text = "\n".join(render_sentence(s) for s in sentences) if sentences else ""
    ok, problems = audit_narrative(text, ids)
    audit = {"ok": ok, "problems": problems, "n_sentences": len(sentences)}
    return text, audit
