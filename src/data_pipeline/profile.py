from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "Date",
    "taux_naissances",
    "taux_décès",
    "Croissance_Naturelle",
    "Nb_mariages",
    "IPC",
    "Masse_Monétaire",
]

@dataclass(frozen=True)
class ProfileOutputs:
    desc: pd.DataFrame
    missing: pd.DataFrame
    coverage: pd.DataFrame
    meta: dict[str, Any]

def _to_datetime_month(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    return out

def profile_dataset(df: pd.DataFrame, *, variables: list[str]) -> ProfileOutputs:
    df = _to_datetime_month(df)
    df = df.sort_values("Date").reset_index(drop=True)

    cols = ["Date"] + variables
    d = df[cols].copy()

    # Stats descriptives robustes
    desc = d[variables].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    desc["missing_n"] = d[variables].isna().sum().values
    desc["missing_pct"] = (d[variables].isna().mean().values * 100.0)

    # Manquants par colonne
    missing = pd.DataFrame({
        "missing_n": d[variables].isna().sum(),
        "missing_pct": d[variables].isna().mean() * 100.0,
    }).sort_values("missing_pct", ascending=False)

    # Couverture temporelle par variable (première/dernière obs non-na)
    cov_rows = []
    for v in variables:
        s = d[["Date", v]].dropna()
        cov_rows.append({
            "variable": v,
            "start": s["Date"].min(),
            "end": s["Date"].max(),
            "n_obs": int(s.shape[0]),
        })
    coverage = pd.DataFrame(cov_rows).sort_values("variable")

    meta = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "min_date": None if df["Date"].isna().all() else df["Date"].min().isoformat(),
        "max_date": None if df["Date"].isna().all() else df["Date"].max().isoformat(),
        "variables": variables,
    }
    return ProfileOutputs(desc=desc, missing=missing, coverage=coverage, meta=meta)
