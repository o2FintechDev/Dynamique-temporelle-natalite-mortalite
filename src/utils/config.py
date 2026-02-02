from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(override=False)

@dataclass(frozen=True)
class Settings:
    app_name: str = "AnthroDem Lab"
    offline: bool = os.getenv("ANTHRODEM_OFFLINE", "0").strip() == "1"

    # Cache
    cache_dir: str = os.getenv("ANTHRODEM_CACHE_DIR", "cache")
    outputs_dir: str = os.getenv("ANTHRODEM_OUTPUTS_DIR", "outputs")
    http_cache_path: str = os.getenv("ANTHRODEM_HTTP_CACHE_PATH", "cache/http_cache.sqlite")
    http_cache_expire_seconds: int = int(os.getenv("ANTHRODEM_HTTP_CACHE_EXPIRE_SECONDS", "86400"))

    # API keys
    fred_api_key: str | None = os.getenv("FRED_API_KEY")
    insee_client_id: str | None = os.getenv("INSEE_CLIENT_ID")
    insee_client_secret: str | None = os.getenv("INSEE_CLIENT_SECRET")

settings = Settings()
