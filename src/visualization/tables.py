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
    Exporte un tableau LaTeX 'booktabs' prêt à être \input{}.

    - wrap_table_env=False (recommandé) :
        génère uniquement le tabular, car ton latex_report.py enveloppe déjà avec table/caption/label.
    - wrap_table_env=True :
        génère un environnement table complet (utile si tu veux le rendre standalone).
    """
    path = Path(path)
    _ensure_parent(path)

    latex = df.to_latex(
        index=index,
        escape=True,
        booktabs=True,
        longtable=False,
        float_format=(lambda x: float_format.format(x)) if float_format else None,
    )

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
    float_format: str = "{:.3f}",
) -> tuple[Path, Path]:
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
