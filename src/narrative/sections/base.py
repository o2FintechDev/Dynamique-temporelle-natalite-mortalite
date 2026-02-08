from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

from src.narrative.tex_snippets import normalize_key


# ---------- unicode normalization ----------
def normalize_unicode(s: str) -> str:
    if not s:
        return ""
    return (
        s.replace("−", "-")
         .replace("–", "-")
         .replace("—", "-")
         .replace("→", "->")
         .replace("\u00a0", " ")  # nbsp
    )


# ---------- core helpers ----------
def escape_tex(s: Any) -> str:
    """
    Escape texte LaTeX pour contenu "normal".
    IMPORTANT: ne pas échapper "\" sinon tu casses tes macros.
    """
    if s is None:
        return ""
    s = normalize_unicode(str(s))
    return (
        s.replace("&", r"\&")
         .replace("%", r"\%")
         .replace("$", r"\$")
         .replace("#", r"\#")
         .replace("{", r"\{")
         .replace("}", r"\}")
         .replace("_", r"\_")
         .replace("~", r"\textasciitilde{}")
         .replace("^", r"\textasciicircum{}")
    )


def label_tex_safe(s: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9:\-]+", "-", str(s or "na"))
    out = re.sub(r"-{2,}", "-", out).strip("-")
    return out[:80] if out else "na"


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def lookup(manifest: Dict[str, Any], kind: str, key: str) -> Optional[str]:
    lu = (manifest.get("lookup") or {}).get(kind) or {}
    if key in lu:
        return lu[key]
    for a in ((manifest.get("artefacts") or {}).get(kind) or []):
        if a.get("key") == key:
            return a.get("path")
    return None


# ---------- markdown -> latex (minimal, safe) ----------
def md_basic_to_tex(md: str) -> str:
    s = normalize_unicode((md or "").strip())

    # stash maths
    math_tokens: list[str] = []

    def _stash(m: re.Match) -> str:
        math_tokens.append(m.group(0))
        return f"@@MATH{len(math_tokens)-1}@@"

    s = re.sub(r"\$\$.*?\$\$", _stash, s, flags=re.DOTALL)
    s = re.sub(r"\$.*?\$", _stash, s, flags=re.DOTALL)

    s = escape_tex(s)

    # bold / ital
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\emph{\1}", s)

    # inline code -> \texttt{}
    s = re.sub(r"`([^`]+)`", r"\\texttt{\1}", s)

    for i, tok in enumerate(math_tokens):
        s = s.replace(f"@@MATH{i}@@", tok)

    return s


def narr_call(raw_key: str) -> str:
    return r"\Narrative{" + normalize_key(raw_key) + r"}"


# ---------- include artefacts ----------
def include_table_tex(*, run_root: Path, tbl_rel: str, caption: str, label: str) -> str:
    """
    Tables exportées via save_table_tex(wrap_table_env=False) => fragment tabular.
    On wrap ici (caption/label/adjustbox) de façon stable.
    """
    fname = Path(tbl_rel).name.replace(" ", "_")
    return "\n".join([
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{" + escape_tex(caption) + r"}",
        r"\label{" + label_tex_safe(label) + r"}",
        r"\begin{adjustbox}{max width=\linewidth,center}",
        r"\input{../artefacts/tables/" + fname + r"}",
        r"\end{adjustbox}",
        r"\end{table}",
        "",
    ])


def include_figure(*, fig_rel: str, caption: str, label: str) -> str:
    fname = Path(fig_rel).name.replace(" ", "_")
    return "\n".join([
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{../artefacts/figures/" + fname + r"}",
        r"\caption{" + escape_tex(caption) + r"}",
        r"\label{" + label_tex_safe(label) + r"}",
        r"\end{figure}",
        "",
    ])


# ---------- spec shared ----------
@dataclass(frozen=True)
class SectionSpec:
    key: str
    title: str
    intro_md: str
    figure_keys: List[str]
    table_keys: List[str]
    metric_keys: List[str]
