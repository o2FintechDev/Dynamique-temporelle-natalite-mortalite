from __future__ import annotations
import pandas as pd
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.settings import settings

log = get_logger("data_pipeline.loader", settings.log_level)

EXPECTED_COLUMNS = [
    "Date",
    "taux_naissances",
    "taux_décès",
    "Croissance_Naturelle",
    "Nb_mariages",
    "IPC",
    "Masse_Monétaire",
]

class DataSchemaError(ValueError):
    pass

def load_local_excel(path: Path | None = None) -> pd.DataFrame:
    p = path or settings.data_path
    if not p.exists():
        raise FileNotFoundError(f"Fichier introuvable: {p}")

    df = pd.read_excel(p, engine="openpyxl")
    cols = list(df.columns)

    if cols != EXPECTED_COLUMNS:
        raise DataSchemaError(
            "Colonnes inattendues.\n"
            f"Attendu (ordre exact): {EXPECTED_COLUMNS}\n"
            f"Reçu: {cols}"
        )

    return df
