from __future__ import annotations
import pandas as pd

def head_table(df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df.head(n)
