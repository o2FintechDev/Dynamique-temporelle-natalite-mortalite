#fred.py

from __future__ import annotations
from typing import Any
import pandas as pd

from src.utils import settings, get_logger
from .base import CachedHttpClient, ApiError

log = get_logger("data_api.fred")

class FredClient:
    BASE = "https://api.stlouisfed.org/fred"

    def __init__(self, http: CachedHttpClient) -> None:
        self.http = http

    def get_series_observations(self, series_id: str, *, frequency: str = "m") -> pd.DataFrame:
        """
        Retourne un DataFrame indexé par date, colonne 'value' (float, NaN si manquant).
        """
        url = f"{self.BASE}/series/observations"
        params: dict[str, Any] = {
            "series_id": series_id,
            "file_type": "json",
            "frequency": frequency,  # 'm' = monthly
        }
        if not settings.fred_api_key:
            raise ApiError(
                "FRED_API_KEY manquante. Définis-la dans l'environnement (FRED_API_KEY) "
                "ou dans ton fichier de config (.env) selon src.utils.settings."
            )

        params["api_key"] = settings.fred_api_key

        r = self.http.request("GET", url, params=params)
        if not r.json or "observations" not in r.json:
            raise ApiError(f"Réponse FRED invalide pour {series_id} (url={r.url})")

        obs = r.json["observations"]
        df = pd.DataFrame(obs)
        if df.empty:
            return pd.DataFrame(columns=["value"], index=pd.DatetimeIndex([], name="date"))

        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["date", "value"]].set_index("date").sort_index()
        df.index.name = "date"
        df.attrs["source"] = "FRED"
        df.attrs["series_id"] = series_id
        df.attrs["fetched_url"] = r.url
        df.attrs["from_cache"] = r.from_cache
        return df
