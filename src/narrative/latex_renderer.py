# src/narrative/latex_renderer.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.narrative.sections.base import SectionSpec, lookup, read_json
from src.narrative.sections.spec import default_spec

from src.narrative.sections.sec_data import render_sec_data
from src.narrative.sections.sec_descriptive import render_sec_descriptive
from src.narrative.sections.sec_stationarity import render_sec_stationarity
from src.narrative.sections.sec_univariate import render_sec_univariate
from src.narrative.sections.sec_multivariate import render_sec_multivariate
from src.narrative.sections.sec_cointegration import render_sec_cointegration
from src.narrative.sections.sec_anthropology import render_sec_anthropology
from src.narrative.sections.sec_conclusion import render_sec_conclusion
import re

SECTION_RENDERERS = {
    "sec_data": render_sec_data,
    "sec_descriptive": render_sec_descriptive,
    "sec_stationarity": render_sec_stationarity,
    "sec_univariate": render_sec_univariate,
    "sec_multivariate": render_sec_multivariate,
    "sec_cointegration": render_sec_cointegration,
    "sec_anthropology": render_sec_anthropology,
    "sec_conclusion": render_sec_conclusion,
}


_CTRL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def _sanitize_tex(s: str) -> str:
    if s is None:
        return ""

    s = s.replace("\ufeff", "")
    s = _CTRL.sub("", s)

    # stop $$ accidents
    while "$$" in s:
        s = s.replace("$$", "$")
    
    # unicode grecs -> latex cmds
    s = s.replace("α", r"\alpha").replace("β", r"\beta")

    # 1) remplacer les patterns dangereux "$\alpha" ou "\alpha$" isolés
    s = s.replace(r"$\alpha", r"\alpha").replace(r"\alpha$", r"\alpha")
    s = s.replace(r"$\beta", r"\beta").replace(r"\beta$", r"\beta")

    # 2) corriger les cas "=$\alpha $$\beta$'" -> "=\alpha\beta'"
    s = s.replace(r"= $\\alpha $$\\beta$'", r"= \\alpha\\beta'")
    s = s.replace(r"= $\\alpha $$\\beta$' ", r"= \\alpha\\beta' ")

    s = (s.replace("Ã©","é").replace("Ã¨","è").replace("Ãª","ê").replace("Ã ","à")
       .replace("Ã´","ô").replace("Ã¹","ù").replace("Ã¢","â").replace("Ã®","î")
       .replace("Ã§","ç").replace("Â",""))
    

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
    # Génère les blocks (fichiers latex/blocks/sec_*.tex)
    out_map = render_all_section_blocks(run_root, manifest)

    # --- NEW: sanitation post-write des fichiers générés ---
    sanitized = []
    sanitize_errors = {}

    for key, rel_path in (out_map or {}).items():
        try:
            p = (run_root / rel_path).resolve()
            if not p.exists():
                continue
            raw = p.read_text(encoding="utf-8", errors="replace")
            clean = _sanitize_tex(raw)
            if clean != raw:
                p.write_text(clean, encoding="utf-8")
                sanitized.append(rel_path)
        except Exception as e:
            sanitize_errors[key] = f"{type(e).__name__}: {e}"

    audit = {
        "blocks_written": sorted(out_map.keys()),
        "blocks_paths": out_map,
        "blocks_dir": "latex/blocks",
        "sanitized_blocks": sanitized,
        "sanitize_errors": sanitize_errors,
        "sanitize_note": "Suppression des caractères de contrôle (U+0000..U+001F hors \\t\\n\\r) + α/β -> \\alpha/\\beta.",
    }
    return {"metrics": {"m.report.blocks": audit}}
