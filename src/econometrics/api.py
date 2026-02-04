# econometrics/api.py
from __future__ import annotations
from __future__ import annotations

from typing import Any
import pandas as pd

from .diagnostics import run_diagnostics_pack
from .univariate import run_univariate_pack
from .multivariate import run_var_pack, run_granger_pack
from .cointegration import run_cointegration_pack
from .impulse import run_impulse_pack


Pack = dict[str, Any]


def _merge_pack(dst: Pack, src: Pack) -> Pack:
    """
    Merge profond standard du format pack:
      {"tables":{...}, "metrics":{...}, "figures":{...}, "models":{...}}
    - Concatène les sous-dicts.
    - En cas de collision de slug, suffixe _2, _3... (défensif, traçable).
    """
    for bucket in ["tables", "metrics", "figures", "models"]:
        if bucket not in src:
            continue
        dst.setdefault(bucket, {})
        if not isinstance(dst[bucket], dict) or not isinstance(src[bucket], dict):
            raise TypeError(f"Bucket '{bucket}' doit être un dict.")

        for k, v in src[bucket].items():
            if k not in dst[bucket]:
                dst[bucket][k] = v
            else:
                # collision: on suffixe
                i = 2
                nk = f"{k}_{i}"
                while nk in dst[bucket]:
                    i += 1
                    nk = f"{k}_{i}"
                dst[bucket][nk] = v
    return dst


def diagnostics_pack(df: pd.DataFrame, *, y: str) -> Pack:
    return run_diagnostics_pack(df, y=y)


def modelisation_pack(df: pd.DataFrame, *, y: str, x: list[str] | None = None, with_granger: bool = True) -> Pack:
    out: Pack = {}

    _merge_pack(out, run_univariate_pack(df, y=y))

    if x:
        vars_ = [y] + x
        _merge_pack(out, run_var_pack(df, vars=vars_))

        if with_granger:
            _merge_pack(out, run_granger_pack(df, vars=vars_, maxlag=6))

    return out


def resultats_pack(df: pd.DataFrame, *, vars: list[str]) -> Pack:
    out: Pack = {}
    _merge_pack(out, run_cointegration_pack(df, vars=vars))
    _merge_pack(out, run_impulse_pack(df, vars=vars))
    return out
