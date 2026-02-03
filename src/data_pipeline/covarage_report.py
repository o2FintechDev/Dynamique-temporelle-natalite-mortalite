from __future__ import annotations
import pandas as pd

def missing_time_gaps(df: pd.DataFrame) -> pd.DataFrame:
    full = pd.date_range(df.index.min(), df.index.max(), freq="MS")
    missing_dates = full.difference(df.index)
    return pd.DataFrame({"missing_date": missing_dates})

def data_coverage_report(df: pd.DataFrame) -> dict:
    full = pd.date_range(df.index.min(), df.index.max(), freq="MS")
    holes = full.difference(df.index)

    by_var = []
    for c in df.columns:
        s = df[c]
        by_var.append({
            "variable": c,
            "n_total_rows": int(len(df)),
            "n_non_null": int(s.notna().sum()),
            "null_rate": float(s.isna().mean()),
            "start": (s.dropna().index.min().date().isoformat() if s.notna().any() else None),
            "end": (s.dropna().index.max().date().isoformat() if s.notna().any() else None),
        })

    return {
        "dataset": {
            "start": df.index.min().date().isoformat(),
            "end": df.index.max().date().isoformat(),
            "n_rows": int(len(df)),
            "n_missing_dates_in_index": int(len(holes)),
        },
        "missing_dates": [d.date().isoformat() for d in holes[:200]],
        "variables": by_var,
    }
