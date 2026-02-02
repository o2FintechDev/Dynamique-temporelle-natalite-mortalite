from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

Provider = Literal["fred", "insee_bdm"]

@dataclass(frozen=True)
class VariableSpec:
    variable_id: str
    label: str
    provider: Provider
    provider_key: str   # ex: FRED series_id ou INSEE series_id
    frequency_hint: str = "MS"  # monthly start

# Catalogue minimal (extensible)
VARIABLES: dict[str, VariableSpec] = {
    # Exemples FRED (macro mensuel)
    "cpi_us": VariableSpec("cpi_us", "CPI (US, monthly) - demo", "fred", "CPIAUCSL"),
    "unrate_us": VariableSpec("unrate_us", "Unemployment Rate (US) - demo", "fred", "UNRATE"),

    # Exemples INSEE (à remplacer par tes séries BDM cibles)
    # Les IDs exacts dépendent de la série BDM; ce sont des placeholders structurés.
    "births_fr_insee": VariableSpec("births_fr_insee", "Naissances (FR) - INSEE BDM", "insee_bdm", "SERIE_BDM_ID_NAISSANCES"),
    "deaths_fr_insee": VariableSpec("deaths_fr_insee", "Décès (FR) - INSEE BDM", "insee_bdm", "SERIE_BDM_ID_DECES"),
}
