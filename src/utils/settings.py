from __future__ import annotations
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Settings:
    # Données repo
    dataset_path: str = os.getenv("ANTHRODEM_DATASET_PATH", "data/bdd_natalité_mortalité_clean.xlsx")

    # Cache
    disk_cache_dir: str = os.getenv("ANTHRODEM_CACHE_DIR", "cache")
    offline_mode: bool = os.getenv("ANTHRODEM_OFFLINE_MODE", "1") == "1"

    # LLM (désactivé par défaut)
    llm_enabled: bool = os.getenv("ANTHRODEM_LLM_ENABLED", "0") == "1"

settings = Settings()
