# src/data_pipeline/loader.py
from __future__ import annotations

import io
from pathlib import Path
import pandas as pd
import requests

from src.utils.settings import settings
from src.utils.paths import data_dir
from src.utils.logger import get_logger

log = get_logger("data_pipeline.loader")

def load_clean_dataset() -> pd.DataFrame:
    if settings.data_url:
        log.info(f"Loading dataset from URL: {settings.data_url}")
        r = requests.get(settings.data_url, timeout=30)
        r.raise_for_status()
        content = r.content
        # Excel
        return pd.read_excel(io.BytesIO(content))
    # fallback local
    p = data_dir() / settings.local_xlsx
    log.info(f"Loading dataset from local file: {p}")
    if not p.exists():
        raise FileNotFoundError(f"Dataset introuvable: {p}. Configure ANTHRODEM_DATA_URL ou place le fichier en data/.")
    return pd.read_excel(p)
