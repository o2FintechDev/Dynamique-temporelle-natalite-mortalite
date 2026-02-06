# src/visualization/tables.py
from __future__ import annotations

from pathlib import Path
import pandas as pd


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_table_csv(df: pd.DataFrame, path: Path) -> Path:
    """
    Sauve le CSV (utile pour debug/data). Retourne le path.
    """
    path = Path(path)
    _ensure_parent(path)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def save_table_tex(
    df: pd.DataFrame,
    path: Path,
    *,
    caption: str = "",
    label: str = "",
    index: bool = False,
    float_format: str = "{:.3f}",
    wrap_table_env: bool = False,
) -> Path:
    """
    Exporte un tableau LaTeX prêt à être \input{}.
    Compatible pandas anciens: booktabs peut être absent.
    """
    path = Path(path)
    _ensure_parent(path)

    fmt = (lambda x: float_format.format(x)) if float_format else None

    # Essai pandas "récent"
    try:
        latex = df.to_latex(
            index=index,
            escape=True,
            longtable=False,
            float_format=fmt,
            booktabs=True,
        )
    except TypeError:
        # Fallback pandas "ancien"
        latex = df.to_latex(
            index=index,
            escape=True,
            longtable=False,
            float_format=fmt,
        )

        # Remplacement des \hline par booktabs si présents
        # (évite un rendu "grille" et fonctionne avec \usepackage{booktabs} côté LaTeX)
        if r"\hline" in latex:
            # remplace 1er et 2e \hline
            latex = latex.replace(r"\hline", r"\toprule", 1)
            latex = latex.replace(r"\hline", r"\midrule", 1)

            # remplace le DERNIER \hline restant (s'il existe) par \bottomrule
            pos = latex.rfind(r"\hline")
            if pos != -1:
                latex = latex[:pos] + r"\bottomrule" + latex[pos + len(r"\hline"):]

    if wrap_table_env:
        cap = f"\\caption{{{caption}}}\n" if caption else ""
        lab = f"\\label{{{label}}}\n" if label else ""
        latex = "\\begin{table}[H]\n\\centering\n" + cap + lab + latex + "\n\\end{table}\n"

    path.write_text(latex, encoding="utf-8")
    return path


def save_table_csv_and_tex(
    df: pd.DataFrame,
    csv_path: Path,
    *,
    caption: str = "",
    label: str = "",
    float_format: str = "{:.3f}",) -> tuple[Path, Path]:
    """
    Sauve CSV + TEX en parallèle.
    Convention: même nom, extension changée .csv -> .tex
    """
    csv_path = Path(csv_path)
    tex_path = csv_path.with_suffix(".tex")

    save_table_csv(df, csv_path)

    # Recommandé: pas de wrapper table ici (latex_report.py gère caption/label)
    save_table_tex(
        df,
        tex_path,
        caption=caption,
        label=label,
        wrap_table_env=False,
        float_format=float_format,
        index=False,
    )
    return csv_path, tex_path
