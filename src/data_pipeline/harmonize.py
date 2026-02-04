from __future__ import annotations

import pandas as pd
from src.utils.logger import get_logger

log = get_logger("data_pipeline.harmonize")

REQUIRED_COLUMNS = [
    "Date",
    "taux_naissances",
    "taux_décès",
    "Croissance_Naturelle",
    "Nb_mariages",
    "IPC",
    "Masse_Monétaire",
]

def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonisation minimale et déterministe:
      - typage Date
      - tri par Date
      - garde les colonnes requises si présentes
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    out = df[REQUIRED_COLUMNS].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.sort_values("Date").reset_index(drop=True)
    log.info(f"Harmonized dataset: shape={out.shape}")
    return out
