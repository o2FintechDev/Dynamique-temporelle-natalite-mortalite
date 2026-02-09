# app/data/cleaning_data.py
import re
import numpy as np
import pandas as pd
from pathlib import Path

# --------- PARAMS ----------
input_path = Path(r"C:\Users\audeb\Google Drive\M2\Econométrie_IA\bdd_natilté_mortalité.xlsx")
output_path = input_path.with_name(input_path.stem + "_clean.xlsx")

# Patterns "NA" à uniformiser vers null
NA_TOKENS = {
    "na", "n/a", "n.a", "nan", "null", "none", "", "-", "—", "nd", "n.d", "n/d"
}

def to_null_if_na(x):
    if x is None:
        return np.nan
    if isinstance(x, float) and np.isnan(x):
        return np.nan
    if isinstance(x, str):
        s = x.strip().lower()
        if s in NA_TOKENS:
            return np.nan
    return x

def normalize_text_series(s: pd.Series) -> pd.Series:
    # Trim + collapse multiple spaces
    return (s.astype("string")
             .str.replace(r"\s+", " ", regex=True)
             .str.strip())

def normalize_numeric_like_text(s: pd.Series) -> pd.Series:
    """
    Nettoie les nombres stockés comme texte avec séparateur de milliers (espaces),
    espace insécable, virgule décimale FR, etc.
    Ex: "6 028" -> 6028 ; "1 234,56" -> 1234.56
    """
    s = s.astype("string")

    # null tokens
    s = s.map(to_null_if_na)

    # Convertit espace insécable et autres blancs en rien pour gérer 6 028 / 6 028
    s = s.astype("string").str.replace(r"[\u00A0\u202F\s]", "", regex=True)

    # Gère la virgule décimale française
    s = s.str.replace(",", ".", regex=False)

    # Supprime tout ce qui n'est pas chiffre, signe, point (au cas où)
    s = s.str.replace(r"[^0-9\.\-\+]", "", regex=True)

    # Conversion
    return pd.to_numeric(s, errors="coerce")

def is_mostly_numeric(series: pd.Series, threshold: float = 0.85) -> bool:
    """
    Détermine si une colonne texte est majoritairement numérique.
    """
    temp = normalize_numeric_like_text(series)
    ratio = temp.notna().mean()
    return ratio >= threshold

# --------- LOAD ----------
# Lit toutes les feuilles
xls = pd.ExcelFile(input_path)
sheets = xls.sheet_names

cleaned = {}

for sh in sheets:
    df = pd.read_excel(input_path, sheet_name=sh)

    # 1) Uniformiser NA -> null
    df = df.map(to_null_if_na)

    # 2) Nettoyage colonnes
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            # Normalisation texte
            df[col] = normalize_text_series(df[col])

            # Tentative de conversion numérique si la colonne est "numeric-like"
            if is_mostly_numeric(df[col], threshold=0.80):
                df[col] = normalize_numeric_like_text(df[col])

        elif pd.api.types.is_numeric_dtype(df[col]):
            # Rien à faire, mais on garde les NaN
            pass
        else:
            # Dates, bool, etc.
            pass

    cleaned[sh] = df

# --------- SAVE ----------
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    for sh, df in cleaned.items():
        df.to_excel(writer, sheet_name=sh, index=False)

print("Fichier nettoyé généré :", output_path)
