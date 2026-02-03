from __future__ import annotations
import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA

from src.agent.schemas import Artefact
from src.agent.tools import ToolContext, _next_id
from src.visualization.tables import save_table_csv

def fit_univariate_grid_artefacts(ctx: ToolContext, var: str, max_p: int = 3, max_q: int = 3, max_d: int = 1) -> list[Artefact]:
    df: pd.DataFrame = ctx.memory["df_ms"]
    y = df[var].dropna().astype(float)

    rows = []
    best = {"aic": np.inf, "bic": np.inf, "order": None}

    for d in range(max_d + 1):
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and d == 0 and q == 0:
                    continue
                order = (p, d, q)
                try:
                    m = ARIMA(y, order=order, trend="c").fit()
                    aic = float(m.aic)
                    bic = float(m.bic)
                    rows.append({"order": str(order), "aic": aic, "bic": bic, "nobs": int(m.nobs)})
                    if aic < best["aic"]:
                        best = {"aic": aic, "bic": bic, "order": order}
                except Exception as e:
                    rows.append({"order": str(order), "error": str(e)})

    tab = pd.DataFrame(rows).sort_values(["aic"], na_position="last")
    aid = _next_id(ctx, "table")
    p = ctx.run_dirs.tables_dir / f"{aid}_univariate_grid_{var}.csv"
    save_table_csv(tab, p)

    aid2 = _next_id(ctx, "metric")
    pm = ctx.run_dirs.metrics_dir / f"{aid2}_univariate_best_{var}.json"
    import json
    pm.write_text(json.dumps({"best": best}, ensure_ascii=False, indent=2), encoding="utf-8")

    return [
        Artefact(artefact_id=aid, kind="table", name=f"univariate_grid_{var}", path=str(p), meta={"max_p": max_p, "max_q": max_q, "max_d": max_d}),
        Artefact(artefact_id=aid2, kind="metric", name=f"univariate_best_{var}", path=str(pm), meta={"best": best}),
    ]
