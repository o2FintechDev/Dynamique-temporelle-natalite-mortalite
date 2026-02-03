from __future__ import annotations
from pathlib import Path
import pandas as pd

def save_table_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8")

def save_table_latex(df: pd.DataFrame, path: Path, caption: str = "", label: str = "") -> None:
    latex = df.to_latex(index=False, escape=True)
    if caption or label:
        cap = f"\\caption{{{caption}}}\n" if caption else ""
        lab = f"\\label{{{label}}}\n" if label else ""
        latex = "\\begin{table}[ht]\n\\centering\n" + cap + lab + latex + "\n\\end{table}\n"
    path.write_text(latex, encoding="utf-8")
