from __future__ import annotations
import pandas as pd

def coverage_report(df: pd.DataFrame, *, variables: list[str]) -> pd.DataFrame:
    out = []
    for v in variables:
        s = df[v]
        out.append({
            "variable": v,
            "nobs": int(s.notna().sum()),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
            "missing": int(s.isna().sum()),
        })
    return pd.DataFrame(out).set_index("variable")