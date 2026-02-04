# src/narrative/latex_report.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import re

from src.utils.run_writer import RunWriter


CANONICAL_ORDER = [
    "01_methodologie",
    "02_donnees_preparation",
    "03_tests_stationnarite",
    "04_modelisation",
    "05_validation_residus",
    "06_regimes_ruptures",
    "07_interpretation_anthropologique",
    "08_conclusion",
]


def render_step_block(rw: RunWriter, step_name: str, latex_body: str, *, title: str) -> str:
    """
    Écrit un bloc LaTeX autonome (hors préambule) dans latex/blocks/<step_name>.tex
    """
    blocks_dir = rw.paths.latex_dir / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    block_path = blocks_dir / f"{step_name}.tex"

    content = rf"""
\section{{{title}}}
{latex_body}
"""
    block_path.write_text(content.strip() + "\n", encoding="utf-8")

    rel_path = str(block_path.relative_to(rw.paths.run_dir)).replace("\\", "/")
    rw.register_artefact("latex_blocks", lookup_key=step_name, rel_path=rel_path, meta={"title": title})
    return rel_path


def rebuild_master(rw: RunWriter) -> None:
    """
    Rebuild master.tex en injectant \input{blocks/<step>.tex} pour les blocs présents,
    en respectant CANONICAL_ORDER.
    """
    manifest = rw.read_manifest()
    present = set((manifest.get("lookup", {}).get("latex_blocks", {}) or {}).keys())

    inputs: List[str] = []
    for s in CANONICAL_ORDER:
        if s in present:
            inputs.append(rf"\input{{blocks/{s}.tex}}")

    master = rw.paths.latex_master_path.read_text(encoding="utf-8")

    # Remplace la zone BLOCKS (auto)
    master_new = re.sub(
        r"% --- BLOCKS \(auto\) ---.*?\\end\{document\}",
        "% --- BLOCKS (auto) ---\n" + "\n".join(inputs) + "\n\n\\end{document}",
        master,
        flags=re.S,
    )
    rw.paths.latex_master_path.write_text(master_new, encoding="utf-8")
