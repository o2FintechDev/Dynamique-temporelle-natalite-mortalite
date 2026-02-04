from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.utils import get_logger
from src.utils.paths import data_dir

log = get_logger("data_pipeline.loader")

DATA_FILE = "bdd_natalite_mortalite_clean.xlsx"

def load_clean_dataset() -> pd.DataFrame:
    """
    Charge la source unique locale:
      data/bdd_natalite_mortalite_clean.xlsx
    """
    path = data_dir() / DATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    df = pd.read_excel(path)
    log.info(f"Loaded dataset: shape={df.shape} path={path}")
    return df
