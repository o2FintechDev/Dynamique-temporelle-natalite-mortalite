from __future__ import annotations
import pandas as pd
import numpy as np

def harmonize_monthly_index(df: pd.DataFrame, date_col: str = "Date") -> tuple[pd.DataFrame, dict]:
    out = df.copy()

    # Date parsing défensif
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    bad_dates = int(out[date_col].isna().sum())

    out = out.dropna(subset=[date_col]).sort_values(date_col)

    # Index mensuel month-start (MS)
    out = out.set_index(date_col)
    out.index = out.index.to_period("M").to_timestamp()
    # OUTRE possibilité:
    # out.index = out.index.to_period("M").to_timestamp(how='start')

    # Déduire fréquence + construire grille complète mensuelle
    start = out.index.min()
    end = out.index.max()
    full_idx = pd.date_range(start=start, end=end, freq="MS")

    # Reindex (introduit trous explicites)
    out = out.reindex(full_idx)

    meta = {
        "date_parse_bad_rows": bad_dates,
        "index_start": str(start.date()) if pd.notna(start) else None,
        "index_end": str(end.date()) if pd.notna(end) else None,
        "n_rows_original": int(df.shape[0]),
        "n_rows_after_drop_bad_dates": int(df.dropna(subset=[date_col]).shape[0]),
        "n_rows_full_index": int(len(full_idx)),
        "n_missing_index_rows": int(len(full_idx) - df.dropna(subset=[date_col]).shape[0]),
        "freq_assumed": "MS",
    }
    return out, meta
