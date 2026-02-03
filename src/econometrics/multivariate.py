from __future__ import annotations
import pandas as pd

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

from src.agent.schemas import Artefact
from src.agent.tools import ToolContext, _next_id
from src.visualization.tables import save_table_csv

def fit_var_artefacts(ctx: ToolContext, vars: list[str], max_lag: int = 6) -> list[Artefact]:
    df: pd.DataFrame = ctx.memory["df_ms"]
    data = df[vars].dropna()

    rows = []
    best = {"aic": None, "bic": None, "selected_lag": None}

    try:
        model = VAR(data)
        sel = model.select_order(maxlags=max_lag)
        best["selected_lag"] = int(sel.aic) if sel.aic is not None else None
        rows.append({"criterion": "aic", "lag": best["selected_lag"]})
        rows.append({"criterion": "bic", "lag": int(sel.bic) if sel.bic is not None else None})
    except Exception as e:
        rows.append({"error": str(e)})

    tab = pd.DataFrame(rows)
    aid = _next_id(ctx, "table")
    p = ctx.run_dirs.tables_dir / f"{aid}_var_selection.csv"
    save_table_csv(tab, p)

    return [Artefact(artefact_id=aid, kind="table", name="var_selection", path=str(p), meta={"vars": vars, "max_lag": max_lag})]

def granger_artefacts(ctx: ToolContext, vars: list[str], max_lag: int = 6) -> list[Artefact]:
    df: pd.DataFrame = ctx.memory["df_ms"]
    data = df[vars].dropna()

    out_rows = []
    for caused in vars:
        for causing in vars:
            if caused == causing:
                continue
            try:
                # granger tests expects [caused, causing]
                res = grangercausalitytests(data[[caused, causing]], maxlag=max_lag, verbose=False)
                # take smallest p-value over lags for ssr_ftest
                pvals = [res[l][0]["ssr_ftest"][1] for l in range(1, max_lag + 1)]
                out_rows.append({"caused": caused, "causing": causing, "min_pvalue_ssr_ftest": float(min(pvals))})
            except Exception as e:
                out_rows.append({"caused": caused, "causing": causing, "error": str(e)})

    tab = pd.DataFrame(out_rows).sort_values(["min_pvalue_ssr_ftest"], na_position="last")
    aid = _next_id(ctx, "table")
    p = ctx.run_dirs.tables_dir / f"{aid}_granger_pairwise.csv"
    save_table_csv(tab, p)

    return [Artefact(artefact_id=aid, kind="table", name="granger_pairwise", path=str(p), meta={"vars": vars, "max_lag": max_lag})]
