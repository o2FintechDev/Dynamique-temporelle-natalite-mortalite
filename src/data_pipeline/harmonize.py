# src/data_pipeline/harmonize.py
from __future__ import annotations
import pandas as pd
import numpy as np

from src.utils.settings import settings
from src.utils.logger import get_logger

log = get_logger("data_pipeline.harmonize")

Y = "Croissance_Naturelle"

def _pick_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    raise KeyError(f"Colonne introuvable. Candidates={candidates}. Colonnes={list(df.columns)}")

def harmonize(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    c_date = _pick_col(df, settings.col_date_candidates)
    c_b = _pick_col(df, settings.col_birth_rate_candidates)
    c_d = _pick_col(df, settings.col_death_rate_candidates)

    df[c_date] = pd.to_datetime(df[c_date], errors="coerce")
    df = df.dropna(subset=[c_date]).sort_values(c_date)

    # nettoyage numeric robuste
    def to_num(s: pd.Series) -> pd.Series:
        if s.dtype.kind in "biufc":
            return s.astype(float)
        x = s.astype(str).str.replace("\u202f", " ", regex=False).str.replace(" ", "", regex=False)
        x = x.str.replace(",", ".", regex=False)
        x = x.replace({"NA": np.nan, "nan": np.nan, "None": np.nan, "": np.nan})
        return pd.to_numeric(x, errors="coerce")

    df[c_b] = to_num(df[c_b])
    df[c_d] = to_num(df[c_d])

    df[Y] = df[c_b] - df[c_d]

    df = df.rename(columns={c_date: "date", c_b: "taux_naissances", c_d: "taux_deces"})
    df = df[["date", "taux_naissances", "taux_deces", Y]]

    # mensuel : normalise au 1er du mois
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()

    # filtre 1975–2025
    df = df[(df["date"] >= "1975-01-01") & (df["date"] <= "2025-12-01")]

    # index date
    df = df.drop_duplicates(subset=["date"]).set_index("date").sort_index()
    full_idx = pd.date_range("1975-01-01", "2025-12-01", freq="MS")
    df = df.reindex(full_idx)
    df.index.name = "date"
    df = df.asfreq("MS")
    log.info(f"Harmonized: n={len(df)} from {df.index.min()} to {df.index.max()}")
    return df

