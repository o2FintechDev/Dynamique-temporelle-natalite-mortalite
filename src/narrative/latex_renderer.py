# src/narrative/latex_renderer.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.narrative.sections.base import SectionSpec, lookup, read_json, md_basic_to_tex
from src.narrative.sections.spec import default_spec

from src.narrative.sections.sec_data import render_sec_data
from src.narrative.sections.sec_descriptive import render_sec_descriptive
from src.narrative.sections.sec_stationarity import render_sec_stationarity
from src.narrative.sections.sec_univariate import render_sec_univariate
from src.narrative.sections.sec_multivariate import render_sec_multivariate
from src.narrative.sections.sec_cointegration import render_sec_cointegration
from src.narrative.sections.sec_anthropology import render_sec_anthropology


SECTION_RENDERERS = {
    "sec_data": render_sec_data,
    "sec_descriptive": render_sec_descriptive,
    "sec_stationarity": render_sec_stationarity,
    "sec_univariate": render_sec_univariate,
    "sec_multivariate": render_sec_multivariate,
    "sec_cointegration": render_sec_cointegration,
    "sec_anthropology": render_sec_anthropology,
}


def render_all_section_blocks(
    run_root: Path,
    manifest: Dict[str, Any],
    *,
    spec: Optional[List[SectionSpec]] = None,
) -> Dict[str, str]:
    spec = spec or default_spec()

    blocks_dir = run_root / "latex" / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)

    # preload metrics_cache (1 seule lecture disque par métrique)
    metrics_cache: Dict[str, Dict[str, Any]] = {}
    for s in spec:
        for mk in s.metric_keys:
            rel = lookup(manifest, "metrics", mk)
            if rel and mk not in metrics_cache:
                metrics_cache[mk] = read_json((run_root / rel).resolve())

    out_map: Dict[str, str] = {}

    for sec in spec:
        renderer = SECTION_RENDERERS.get(sec.key)

        if renderer is None:
            tex: Any = (
                r"\section{" + sec.title + r"}" + "\n\n" +
                r"\textit{Section non implémentée.}" + "\n"
            )
        else:
            tex = renderer(run_root=run_root, manifest=manifest, sec=sec, metrics_cache=metrics_cache)

        # HARDEN: jamais de list dans write_text
        if isinstance(tex, list):
            tex_str = "\n".join(str(x) for x in tex)
        else:
            tex_str = str(tex)

        out_path = blocks_dir / f"{sec.key}.tex"
        out_path.write_text(tex_str, encoding="utf-8")
        out_map[sec.key] = f"latex/blocks/{sec.key}.tex"

    return out_map


def build_section_blocks_from_manifest(run_root: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    out_map = render_all_section_blocks(run_root, manifest)
    audit = {
        "blocks_written": sorted(out_map.keys()),
        "blocks_paths": out_map,
        "blocks_dir": "latex/blocks",
    }
    return {"metrics": {"m.report.blocks": audit}}
