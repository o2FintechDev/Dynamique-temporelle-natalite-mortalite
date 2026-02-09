# src/narrative/sections/base.py
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

    if s is None:
        return ""

    s = normalize_unicode(str(s))

    # Normalisation typographique (safe pdflatex)
    s = (
        s.replace("−", "-")
         .replace("–", "--")
         .replace("—", "---")
         .replace("’", "'")
         .replace("“", '"')
         .replace("”", '"')
    )

    # Escape LaTeX (hors backslash)
    s = (
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
    return s

def label_tex_safe(s: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9:\-]+", "-", str(s or "na"))
    out = re.sub(r"-{2,}", "-", out).strip("-")
    return out[:80] if out else "na"


def read_json(path: Path) -> Dict[str, Any]:
    try:
        txt = path.read_text(encoding="utf-8")
    except Exception:
        try:
            txt = path.read_text(encoding="utf-8-sig")
        except Exception:
            return {}
    try:
        return json.loads(txt)
    except Exception:
        return {}

def looks_like_full_table(tex: str) -> bool:
    t = (tex or "").lower()
    return (
        ("\\begin{table" in t) or ("\\end{table" in t)
        or ("\\begin{longtable" in t) or ("\\end{longtable" in t)
        or ("\\begin{threeparttable" in t) or ("\\end{threeparttable" in t)
    )

def read_text_safe(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    
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
def include_table_tex(
    *,
    run_root: Path,
    tbl_rel: str,
    caption: str,
    label: str,
    mode: str = "fit",
    font_size: str = r"\small",
    tabcolsep_pt: int = 3,
    arraystretch: float = 0.95,
    max_height_ratio: float = 0.85,
) -> str:
    
    fname = Path(tbl_rel).name.replace(" ", "_")

    cap = escape_tex(caption)
    lab = label_tex_safe(label)

    inp = r"\input{../artefacts/tables/" + fname + r"}"

    # --- auto-detect full environment if mode != raw ---
    tbl_path = (run_root / tbl_rel).resolve()
    content = read_text_safe(tbl_path) if tbl_path.exists() else ""
    if mode != "raw" and looks_like_full_table(content):
        mode = "raw"
        
    # Compactage local (évite d'impacter tout le document)
    compact_block = [
        r"\begingroup",
        font_size,
        rf"\setlength{{\tabcolsep}}{{{int(tabcolsep_pt)}pt}}",
        rf"\renewcommand{{\arraystretch}}{{{arraystretch}}}",
    ]
    compact_end = [r"\endgroup"]

    adjust = (
        r"\begin{adjustbox}{"
        + rf"max width=\textwidth, max totalheight={max_height_ratio:.2f}\textheight, keepaspectratio, center"
        + r"}"
    )

    if mode == "raw":
        # Cas: le fichier est déjà un environnement complet (ex: longtable)
        return "\n".join([inp, ""])

    if mode == "landscape":
        # Nécessite dans le preamble: \usepackage{pdflscape}
        return "\n".join([
            r"\begin{landscape}",
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{" + cap + r"}",
            r"\label{" + lab + r"}",
            *compact_block,
            adjust,
            inp,
            r"\end{adjustbox}",
            *compact_end,
            r"\end{table}",
            r"\end{landscape}",
            "",
        ])

    # mode == "fit" (default)
    return "\n".join([
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{" + cap + r"}",
        r"\label{" + lab + r"}",
        *compact_block,
        adjust,
        inp,
        r"\end{adjustbox}",
        *compact_end,
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
