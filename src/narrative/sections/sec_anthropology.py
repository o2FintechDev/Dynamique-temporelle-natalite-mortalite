from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from src.narrative.sections.base import SectionSpec, md_basic_to_tex, escape_tex, narr_call

def render_sec_anthropology(*, run_root: Path, manifest: Dict[str, Any], sec: SectionSpec, metrics_cache: Dict[str, Dict[str, Any]]) -> str:
    m = metrics_cache.get("m.anthro.todd_analysis") or {}
    md = ""
    if isinstance(m, dict):
        md = m.get("markdown") or m.get("text") or ""

    lines: list[str] = [
        r"\section{" + escape_tex(sec.title) + r"}", "",
        md_basic_to_tex(sec.intro_md), "",
        r"\subsection{Synthèse interprétative}",
    ]

    if md.strip():
        lines += [md_basic_to_tex(md), narr_call("m.anthro.todd_analysis"), ""]
    else:
        lines += [md_basic_to_tex("Aucune synthèse anthropologique disponible (métrique absente)."), ""]

    lines += [
        r"\subsection{Conclusion de section}",
        md_basic_to_tex(
            "Cette lecture contextualise les résultats sans sur-interprétation : les faits retenus doivent rester cohérents avec les artefacts économétriques."
        ),
        "",
    ]
    return "\n".join(lines).strip() + "\n"
