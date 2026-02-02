from __future__ import annotations
from typing import Any
import base64
import pandas as pd

from src.utils import settings, get_logger
from .base import CachedHttpClient, ApiError

log = get_logger("data_api.insee")

class InseeClient:
    TOKEN_URL = "https://api.insee.fr/token"
    BASE = "https://api.insee.fr"

    def __init__(self, http: CachedHttpClient) -> None:
        self.http = http
        self._token: str | None = None

    def _get_token(self) -> str:
        if self._token:
            return self._token

        if not settings.insee_client_id or not settings.insee_client_secret:
            raise ApiError("INSEE_CLIENT_ID / INSEE_CLIENT_SECRET manquants (OAuth requis).")

        basic = f"{settings.insee_client_id}:{settings.insee_client_secret}".encode("utf-8")
        auth = base64.b64encode(basic).decode("ascii")
        headers = {"Authorization": f"Basic {auth}", "Content-Type": "application/x-www-form-urlencoded"}
        data = "grant_type=client_credentials"

        r = self.http.request("POST", self.TOKEN_URL, headers=headers, data=data)
        if not r.json or "access_token" not in r.json:
            raise ApiError("Token INSEE invalide.")

        self._token = r.json["access_token"]
        return self._token

    def get_bdm_series(self, series_id: str) -> pd.DataFrame:
        """
        Connecteur générique pour séries INSEE via endpoint BDM (format JSON variable selon séries).
        Endpoint couramment utilisé:
          /series/BDM/V1/data/SERIES_BDM/{series_id}
        Le parsing est défensif: on extrait date + valeur si présents.
        """
        token = self._get_token()
        url = f"{self.BASE}/series/BDM/V1/data/SERIES_BDM/{series_id}"
        headers = {"Authorization": f"Bearer {token}"}

        r = self.http.request("GET", url, headers=headers)
        j = r.json
        if not j:
            raise ApiError(f"Réponse INSEE vide pour {series_id}")

        # Parsing défensif (structures INSEE variables)
        # On cherche des paires (date/valeur) dans des champs typiques.
        rows: list[dict[str, Any]] = []

        def walk(obj: Any) -> None:
            if isinstance(obj, dict):
                # patterns fréquents
                if "TIME_PERIOD" in obj and ("OBS_VALUE" in obj or "OBS_VALUE" in obj.keys()):
                    rows.append({"date": obj.get("TIME_PERIOD"), "value": obj.get("OBS_VALUE")})
                if "date" in obj and ("value" in obj or "valeur" in obj):
                    rows.append({"date": obj.get("date"), "value": obj.get("value", obj.get("valeur"))})
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for it in obj:
                    walk(it)

        walk(j)

        df = pd.DataFrame(rows).dropna(subset=["date"])
        if df.empty:
            return pd.DataFrame(columns=["value"], index=pd.DatetimeIndex([], name="date"))

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        df.index.name = "date"

        df.attrs["source"] = "INSEE"
        df.attrs["series_id"] = series_id
        df.attrs["fetched_url"] = r.url
        df.attrs["from_cache"] = r.from_cache
        return df
