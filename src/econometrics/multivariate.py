# econometrics/multivariate.py
from __future__ import annotations

import pandas as pd

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

from src.agent.schemas import Artefact
from src.agent.tools import ToolContext, _next_id
from src.visualization.tables import save_table_csv


def _get_df(ctx: ToolContext, vars: list[str]) -> pd.DataFrame:
    df: pd.DataFrame = ctx.memory["df_ms"]
    cols = [v for v in vars if v in df.columns]
    tmp = df[cols].copy()
    if "Date" in df.columns:
        t = df[["Date"] + cols].copy()
        t["Date"] = pd.to_datetime(t["Date"], errors="coerce")
        t = t.dropna(subset=["Date"]).sort_values("Date")
        tmp = t[cols].copy()
    return tmp.apply(pd.to_numeric, errors="coerce")


def fit_var_artefacts(ctx: ToolContext, vars: list[str], max_lag: int = 6) -> list[Artefact]:
    data = _get_df(ctx, vars).dropna()

    if data.shape[1] < 2 or data.shape[0] < 50:
        tab = pd.DataFrame([{"error": "VAR nécessite >=2 variables et nobs>=50"}])
        aid = _next_id(ctx, "table")
        p = ctx.run_dirs.tables_dir / f"{aid}_var_selection.csv"
        save_table_csv(tab, p)
        return [Artefact(artefact_id=aid, kind="table", name="var_selection", path=str(p), meta={"vars": vars, "max_lag": max_lag})]

    try:
        model = VAR(data)
        sel = model.select_order(maxlags=max_lag)

        tab = pd.DataFrame([
            {"criterion": "aic", "lag": int(sel.aic) if sel.aic is not None else None},
            {"criterion": "bic", "lag": int(sel.bic) if sel.bic is not None else None},
            {"criterion": "hqic", "lag": int(sel.hqic) if getattr(sel, "hqic", None) is not None else None},
            {"criterion": "fpe", "lag": int(sel.fpe) if getattr(sel, "fpe", None) is not None else None},
        ])
    except Exception as e:
        tab = pd.DataFrame([{"error": str(e)}])

    aid = _next_id(ctx, "table")
    p = ctx.run_dirs.tables_dir / f"{aid}_var_selection.csv"
    save_table_csv(tab, p)

    return [Artefact(artefact_id=aid, kind="table", name="var_selection", path=str(p), meta={"vars": list(data.columns), "max_lag": max_lag})]


def granger_artefacts(ctx: ToolContext, vars: list[str], max_lag: int = 6) -> list[Artefact]:
    data = _get_df(ctx, vars).dropna()

    if data.shape[1] < 2 or data.shape[0] < 50:
        tab = pd.DataFrame([{"error": "Granger nécessite >=2 variables et nobs>=50"}])
        aid = _next_id(ctx, "table")
        p = ctx.run_dirs.tables_dir / f"{aid}_granger_pairwise.csv"
        save_table_csv(tab, p)
        return [Artefact(artefact_id=aid, kind="table", name="granger_pairwise", path=str(p), meta={"vars": vars, "max_lag": max_lag})]

    out_rows = []
    cols = list(data.columns)

    for caused in cols:
        for causing in cols:
            if caused == causing:
                continue
            try:
                res = grangercausalitytests(data[[caused, causing]], maxlag=max_lag, verbose=False)
                pvals = [res[l][0]["ssr_ftest"][1] for l in range(1, max_lag + 1)]
                out_rows.append({
                    "caused": caused,
                    "causing": causing,
                    "min_pvalue_ssr_ftest": float(min(pvals)),
                })
            except Exception as e:
                out_rows.append({"caused": caused, "causing": causing, "error": str(e)})

    tab = pd.DataFrame(out_rows).sort_values(["min_pvalue_ssr_ftest"], na_position="last")

    aid = _next_id(ctx, "table")
    p = ctx.run_dirs.tables_dir / f"{aid}_granger_pairwise.csv"
    save_table_csv(tab, p)

    return [Artefact(artefact_id=aid, kind="table", name="granger_pairwise", path=str(p), meta={"vars": cols, "max_lag": max_lag})]
