# src/narrative/tex_snippets.py
from __future__ import annotations
from pathlib import Path
from typing import Dict
import re

# On évite underscore pour ne pas devoir l'échapper dans \input (macro Narrative)
SAFE = re.compile(r"[^A-Za-z0-9\-]+")

def normalize_key(key: str) -> str:
    """
    Convertit une key logique (ex: tbl.diag.adf) en nom de fichier safe:
    - remplace tout non [A-Za-z0-9-] par '-'
    - pas de underscore
    """
    k = (key or "").strip()
    k = k.replace(" ", "-").replace("_", "-").replace(".", "-")
    k = SAFE.sub("-", k)
    k = re.sub(r"-{2,}", "-", k).strip("-")
    return k[:120] if k else "snippet"

def write_snippets(run_root: Path, snippets: Dict[str, str]) -> Dict[str, str]:
    """
    Écrit artefacts/text/<normalized>.tex et retourne un audit key->relpath.
    """
    outdir = run_root / "artefacts" / "text"
    outdir.mkdir(parents=True, exist_ok=True)

    audit: Dict[str, str] = {}
    for raw_key, tex in (snippets or {}).items():
        nk = normalize_key(raw_key)
        p = outdir / f"{nk}.tex"
        content = (tex or "").strip()
        if content and not content.endswith("\n"):
            content += "\n"
        p.write_text(content, encoding="utf-8")
        audit[raw_key] = str(p.relative_to(run_root)).replace("\\", "/")
    return audit
