from __future__ import annotations
from typing import Any
import pandas as pd

def narrative_from_artefacts(artefacts: dict[str, Any]) -> list[str]:
    """
    Retourne une liste de phrases.
    Règle: chaque phrase doit référencer un artefact_id via le préfixe [aXXXX].
    """
    lines: list[str] = []

    wide_id = artefacts.get("wide_id")
    cov_id = artefacts.get("coverage_id")
    desc_id = artefacts.get("describe_id")

    wide: pd.DataFrame | None = artefacts.get("wide")
    cov: pd.DataFrame | None = artefacts.get("coverage")
    desc: pd.DataFrame | None = artefacts.get("describe")

    if wide_id and isinstance(wide, pd.DataFrame) and not wide.empty:
        lines.append(f"[{wide_id}] Table harmonisée: {wide.shape[0]} mois, {wide.shape[1]} variables.")

    if cov_id and isinstance(cov, pd.DataFrame) and not cov.empty:
        # top trous
        tmp = cov.copy()
        if "missing_points" in tmp.columns:
            top = tmp.sort_values("missing_points", ascending=False).head(2)
            for var, row in top.iterrows():
                lines.append(f"[{cov_id}] Couverture {var}: start={row.get('start')}, end={row.get('end')}, manquants={int(row.get('missing_points',0))}.")

    if desc_id and isinstance(desc, pd.DataFrame) and not desc.empty:
        # missing rate
        if "missing_rate" in desc.columns:
            topm = desc["missing_rate"].sort_values(ascending=False).head(2)
            for var, rate in topm.items():
                lines.append(f"[{desc_id}] Qualité {var}: taux manquant={float(rate):.2%}.")

    if not lines:
        lines.append("[a0000] Aucun artefact exploitable pour produire une synthèse.")
    return lines
