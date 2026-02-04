from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass
class Profile:
    desc: pd.DataFrame
    missing: pd.DataFrame
    meta: dict

def profile_dataset(df: pd.DataFrame, *, variables: list[str]) -> Profile:
    sub = df[variables].copy()
    desc = sub.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    missing = pd.DataFrame({
        "n": [len(sub)] * len(variables),
        "missing": [sub[v].isna().sum() for v in variables],
        "missing_rate": [float(sub[v].isna().mean()) for v in variables],
    }, index=variables)
    meta = {
        "nobs": int(len(sub)),
        "start": str(sub.index.min()),
        "end": str(sub.index.max()),
        "variables": variables,
    }
    return Profile(desc=desc, missing=missing, meta=meta)
