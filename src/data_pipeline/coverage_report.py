from __future__ import annotations

import pandas as pd
from src.utils.logger import get_logger


log = get_logger("data_pipeline.coverage_report")

def coverage_report(df: pd.DataFrame, *, variables: list[str]) -> pd.DataFrame:
    """
    Rapport de couverture simple:
      variable | start | end | n_obs | missing_n | missing_pct
    """
    if "Date" not in df.columns:
        raise ValueError("Colonne 'Date' absente.")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date")

    rows = []
    n_total = df.shape[0]
    for v in variables:
        if v not in df.columns:
            continue
        s = df[["Date", v]]
        s_non = s.dropna()
        missing_n = int(s[v].isna().sum())
        rows.append({
            "variable": v,
            "start": None if s_non.empty else s_non["Date"].min(),
            "end": None if s_non.empty else s_non["Date"].max(),
            "n_obs": int(s_non.shape[0]),
            "missing_n": missing_n,
            "missing_pct": (missing_n / n_total * 100.0) if n_total else None,
        })

    rep = pd.DataFrame(rows).sort_values("variable")
    log.info(f"Coverage report created: rows={rep.shape[0]}")
    return rep
