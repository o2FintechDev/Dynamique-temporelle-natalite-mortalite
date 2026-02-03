from __future__ import annotations
import pandas as pd

def per_column_missingness(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    out = []
    for c in df.columns:
        n_missing = int(df[c].isna().sum())
        out.append({"variable": c, "missing": n_missing, "missing_rate": n_missing / total if total else 0.0})
    return pd.DataFrame(out).sort_values("missing_rate", ascending=False)

def temporal_coverage(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for c in df.columns:
        s = df[c].dropna()
        if s.empty:
            out.append({"variable": c, "start": None, "end": None, "n_obs": 0})
        else:
            out.append({"variable": c, "start": s.index.min().date(), "end": s.index.max().date(), "n_obs": int(s.shape[0])})
    return pd.DataFrame(out).sort_values(["start", "variable"], na_position="last")
