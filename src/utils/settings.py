# src/utils/settings.py
from __future__ import annotations
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Settings:
    # URL "raw" GitHub recommandée (sinon fallback local data/)
    data_url: str | None = os.getenv("ANTHRODEM_DATA_URL")
    # Nom du fichier local fallback
    local_xlsx: str = os.getenv("ANTHRODEM_LOCAL_XLSX", "bdd_natalite_mortalite_clean.xlsx")

    # Colonnes candidates (tu ajustes si besoin)
    col_date_candidates: tuple[str, ...] = ("date", "Date", "DATE", "mois", "Mois")
    col_birth_rate_candidates: tuple[str, ...] = (
        "taux_naissances", "Taux_naissances", "taux_naissance", "naissances_taux", "birth_rate"
    )
    col_death_rate_candidates: tuple[str, ...] = (
        "taux_deces", "taux_décès", "Taux_deces", "Taux_décès", "death_rate"
    )

settings = Settings()
