from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from joblib import Memory

from src.utils.settings import settings
from src.utils.paths import REPO_ROOT, CACHE_DIR
from src.utils.logger import get_logger

log = get_logger("data_pipeline.loader")
memory = Memory(location=str(CACHE_DIR / "joblib"), verbose=0)

EXPECTED_COLS = [
    "Date",
    "Nb_naissances",
    "Nb_décès",
    "solde_naturel",
    "taux_naissances",
    "taux_décès",
    "Croissance_Naturelle",
    "Nb_mariages",
    "IPC",
    "Taux_chômage",
    "Masse_Monétaire",
    "Population",
]

@dataclass(frozen=True)
class DatasetMeta:
    rows: int
    start: str
    end: str
    missing_dates: int
    columns: list[str]

def _resolve_dataset_path() -> Path:
    p = Path(settings.dataset_path)
    return p if p.is_absolute() else (REPO_ROOT / p)

@memory.cache
def load_dataset() -> tuple[pd.DataFrame, DatasetMeta]:
    path = _resolve_dataset_path()
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {path}")

    df = pd.read_excel(path, engine="openpyxl")
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    df = df[EXPECTED_COLS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if df["Date"].isna().any():
        bad = int(df["Date"].isna().sum())
        raise ValueError(f"{bad} lignes ont une Date invalide")

    # Index mensuel (MS = Month Start)
    df = df.sort_values("Date").set_index("Date")
    df.index = df.index.to_period("M").to_timestamp("MS")

    # Coercition numérique (tolérante)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="MS")
    missing_dates = int(len(full_index.difference(df.index)))

    meta = DatasetMeta(
        rows=int(df.shape[0]),
        start=str(df.index.min().date()),
        end=str(df.index.max().date()),
        missing_dates=missing_dates,
        columns=list(df.columns),
    )
    log.info(f"Dataset chargé: {meta.rows} lignes, {meta.start} -> {meta.end}, trous_dates={meta.missing_dates}")
    return df, meta
