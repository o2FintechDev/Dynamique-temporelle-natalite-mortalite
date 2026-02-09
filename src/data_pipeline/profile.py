# src/data_pipeline/profile.py
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass
class Profile:
    desc: pd.DataFrame
    missing: pd.DataFrame
    meta: dict

def profile_dataset(df: pd.DataFrame, *, variables: list[str]) -> Profile:
    vars_ok = [v for v in variables if v in df.columns]
    vars_missing = [v for v in variables if v not in df.columns]

    sub = df[vars_ok].copy()

    desc = sub.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T if not sub.empty else pd.DataFrame([])
    missing = pd.DataFrame({
        "n": [len(df)] * len(vars_ok),
        "missing": [int(sub[v].isna().sum()) for v in vars_ok],
        "missing_rate": [float(sub[v].isna().mean()) for v in vars_ok],
    }, index=vars_ok)

    meta = {
        "nobs": int(len(df)),
        "start": str(df.index.min()) if len(df) else None,
        "end": str(df.index.max()) if len(df) else None,
        "vars_requested": variables,
        "vars_used": vars_ok,
        "vars_missing": vars_missing,
    }

    return Profile(desc=desc, missing=missing, meta=meta)