from __future__ import annotations

import re
from pathlib import Path
from .schema import NarrativePacket

def _escape_tex(s: str) -> str:
    if s is None:
        return ""
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )

def _md_to_tex(md: str) -> str:
    s = md.strip()
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\emph{\1}", s)
    s = _escape_tex(s)
    return s.replace("\n\n", "\n\n") + "\n"

def render_report_tex(run_root: Path, packet: NarrativePacket, *, tex_name: str = "report.tex") -> Path:
    latex_dir = run_root / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)
    out = latex_dir / tex_name

    lines = [
        r"\documentclass[11pt,a4paper]{report}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[french]{babel}",
        r"\usepackage{geometry}\geometry{margin=2.5cm}",
        r"\usepackage{setspace}\onehalfspacing",
        r"\usepackage{amsmath, amssymb}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\usepackage{booktabs,longtable,adjustbox,xcolor,hyperref}",
        r"\usepackage{etoolbox}",
        r"\AtBeginEnvironment{figure}{\centering}",
        r"\AtBeginEnvironment{table}{\centering}",
        r"\graphicspath{{./../artefacts/figures/}{./}}",
        r"\title{Dynamique temporelle de la natalité et de la mortalité en France (1975--2025)}",
        r"\author{AnthroDem Lab}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\tableofcontents",
        r"\listoffigures",
        r"\listoftables",
        r"\clearpage",
    ]

    for ch in packet.chapters:
        lines.append(r"\chapter{" + _escape_tex(ch.title) + "}")
        for sec in ch.sections:
            lines.append(r"\section{" + _escape_tex(sec.title) + "}")
            for p in sec.paragraphs:
                lines.append(_md_to_tex(p.text_md))
                if p.evidence:
                    refs = ", ".join([f"{e.kind}:{e.path}" for e in p.evidence])
                    lines.append(r"\begin{quote}\small\textit{Sources: " + _escape_tex(refs) + r"}\end{quote}")

    lines.append(r"\end{document}")
    out.write_text("\n\n".join(lines), encoding="utf-8")
    return out
