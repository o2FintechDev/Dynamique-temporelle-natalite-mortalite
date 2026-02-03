from __future__ import annotations
import pandas as pd
import numpy as np

def coverage_report(df_ms: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # df_ms index = MS
    rep_rows = []
    for col in df_ms.columns:
        s = df_ms[col]
        non_na = s.dropna()
        rep_rows.append({
            "variable": col,
            "start": non_na.index.min().date().isoformat() if len(non_na) else None,
            "end": non_na.index.max().date().isoformat() if len(non_na) else None,
            "n_obs": int(len(non_na)),
            "n_missing": int(s.isna().sum()),
            "missing_rate": float(s.isna().mean()) if len(s) else 1.0,
        })

    table = pd.DataFrame(rep_rows).sort_values("variable")
    holes = int(df_ms.isna().any(axis=1).sum())
    meta = {
        "n_rows": int(len(df_ms)),
        "n_cols": int(df_ms.shape[1]),
        "n_rows_with_any_missing": holes,
        "period_start": df_ms.index.min().date().isoformat() if len(df_ms) else None,
        "period_end": df_ms.index.max().date().isoformat() if len(df_ms) else None,
        "freq": "MS",
    }
    return table, meta
