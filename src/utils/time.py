from __future__ import annotations
import pandas as pd

def to_month_start_index(dt: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.to_datetime(dt)
    return pd.DatetimeIndex(idx.to_period("M").to_timestamp("MS"))

def parse_date_like(value: str) -> pd.Timestamp:
    # accepte "YYYY-MM", "YYYY-MM-DD", etc.
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Date invalide: {value}")
    return ts

def monthly_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    start = pd.Timestamp(start).to_period("M").to_timestamp("MS")
    end = pd.Timestamp(end).to_period("M").to_timestamp("MS")
    return pd.date_range(start=start, end=end, freq="MS")
