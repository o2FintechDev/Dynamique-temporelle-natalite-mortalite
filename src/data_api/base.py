# base.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import requests
import requests_cache

from src.utils import settings, get_logger

log = get_logger("data_api.base")

class ApiError(RuntimeError):
    pass

@dataclass(frozen=True)
class ApiResponse:
    url: str
    status_code: int
    from_cache: bool
    json: dict[str, Any] | None = None
    text: str | None = None

class CachedHttpClient:
    """
    Cache disque + stratégie offline:
    - online: requête normale (réponse cachée)
    - offline: lecture cache uniquement
    - si online échoue: tentative de lecture cache
    """
    def __init__(self) -> None:
        self.session = requests_cache.CachedSession(
            cache_name=settings.http_cache_path,
            backend="sqlite",
            expire_after=settings.http_cache_expire_seconds,
            allowable_methods=("GET", "POST"),
        )

    def request(self, method: str, url: str, *, headers: dict[str, str] | None = None,
                params: dict[str, Any] | None = None, data: Any = None, timeout: int = 30) -> ApiResponse:
        method = method.upper()
        headers = headers or {}
        params = params or {}

        if settings.offline:
            # cache only
            self.session.settings.only_if_cached = True
        else:
            self.session.settings.only_if_cached = False

        try:
            resp = self.session.request(method, url, headers=headers, params=params, data=data, timeout=timeout)
            from_cache = getattr(resp, "from_cache", False)
            content_type = resp.headers.get("Content-Type", "")

            if resp.status_code >= 400:
                body = (resp.text or "").strip()
                raise ApiError(
                    f"HTTP {resp.status_code} sur {resp.url}\n"
                    f"Params: {params}\n"
                    f"Body: {body[:4000]}"
    )

            is_json = "application/json" in content_type or text.lstrip().startswith("{")
            if is_json:
                return ApiResponse(url=str(resp.url), status_code=resp.status_code, from_cache=from_cache, json=resp.json(), text=text)
            return ApiResponse(url=str(resp.url), status_code=resp.status_code, from_cache=from_cache, text=text)

        except requests.exceptions.RequestException as e:
            # fallback: tenter cache
            if settings.offline:
                raise ApiError(f"Offline et cache indisponible pour {url}: {e}") from e

            log.info(f"Echec réseau, tentative fallback cache: {e}")
            try:
                self.session.settings.only_if_cached = True
                resp = self.session.request(method, url, headers=headers, params=params, data=data, timeout=timeout)
                from_cache = getattr(resp, "from_cache", False)
                if not from_cache:
                    raise ApiError(f"Aucune réponse cache disponible pour {url}")
                content_type = resp.headers.get("Content-Type", "")
                if "application/json" in content_type or resp.text.strip().startswith("{"):
                    return ApiResponse(url=str(resp.url), status_code=resp.status_code, from_cache=True, json=resp.json())
                return ApiResponse(url=str(resp.url), status_code=resp.status_code, from_cache=True, text=resp.text)
            except Exception as e2:
                raise ApiError(f"Fallback cache impossible pour {url}: {e2}") from e2
            finally:
                self.session.settings.only_if_cached = False
