# economics/cointegration.py

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from src.agent.schemas import Artefact
from src.agent.tools import ToolContext, _next_id
from src.visualization.tables import save_table_csv


def _get_df(ctx: ToolContext, vars: list[str]) -> pd.DataFrame:
    df: pd.DataFrame = ctx.memory["df_ms"]
    cols = [v for v in vars if v in df.columns]
    d = df[cols].copy()
    if "Date" in df.columns:
        tmp = df[["Date"] + cols].copy()
        tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
        tmp = tmp.dropna(subset=["Date"]).sort_values("Date")
        d = tmp[cols].copy()
    return d.apply(pd.to_numeric, errors="coerce")


def engle_granger_artefacts(ctx: ToolContext, vars: list[str]) -> list[Artefact]:
    vars = [v for v in vars if v is not None]
    data = _get_df(ctx, vars).dropna()

    if data.shape[1] < 2 or data.shape[0] < 30:
        tab = pd.DataFrame([{"error": "Engle-Granger nécessite >=2 variables et nobs>=30"}])
        aid = _next_id(ctx, "table")
        p = ctx.run_dirs.tables_dir / f"{aid}_engle_granger.csv"
        save_table_csv(tab, p)
        return [Artefact(artefact_id=aid, kind="table", name="engle_granger", path=str(p), meta={"vars": vars})]

    # Pairwise (plus robuste que “base only”)
    rows: list[dict] = []
    cols = list(data.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            x, y = cols[i], cols[j]
            try:
                stat, pval, crit = coint(data[x].values, data[y].values)
                rows.append({
                    "x": x,
                    "y": y,
                    "coint_stat": float(stat),
                    "pvalue": float(pval),
                    "nobs": int(data.shape[0]),
                    "crit_1": float(crit[0]),
                    "crit_5": float(crit[1]),
                    "crit_10": float(crit[2]),
                })
            except Exception as e:
                rows.append({"x": x, "y": y, "error": str(e)})

    tab = pd.DataFrame(rows).sort_values(["pvalue"], na_position="last")

    aid = _next_id(ctx, "table")
    p = ctx.run_dirs.tables_dir / f"{aid}_engle_granger.csv"
    save_table_csv(tab, p)
    return [Artefact(artefact_id=aid, kind="table", name="engle_granger", path=str(p), meta={"vars": cols})]


def johansen_artefacts(ctx: ToolContext, vars: list[str], det_order: int = 0, k_ar_diff: int = 4) -> list[Artefact]:
    data = _get_df(ctx, vars).dropna()
    if data.shape[1] < 2 or data.shape[0] < 50:
        tab = pd.DataFrame([{"error": "Johansen nécessite >=2 variables et nobs>=50"}])
        aid = _next_id(ctx, "table")
        p = ctx.run_dirs.tables_dir / f"{aid}_johansen.csv"
        save_table_csv(tab, p)
        return [Artefact(artefact_id=aid, kind="table", name="johansen", path=str(p), meta={"vars": vars, "det_order": det_order, "k_ar_diff": k_ar_diff})]

    try:
        res = coint_johansen(data.values, det_order=det_order, k_ar_diff=k_ar_diff)
        rows = []
        for i, stat in enumerate(res.lr1):
            rows.append({
                "rank_tested": i,
                "trace_stat": float(stat),
                "cv_90": float(res.cvt[i, 0]),
                "cv_95": float(res.cvt[i, 1]),
                "cv_99": float(res.cvt[i, 2]),
            })
        tab = pd.DataFrame(rows)
    except Exception as e:
        tab = pd.DataFrame([{"error": str(e)}])

    aid = _next_id(ctx, "table")
    p = ctx.run_dirs.tables_dir / f"{aid}_johansen.csv"
    save_table_csv(tab, p)
    return [Artefact(artefact_id=aid, kind="table", name="johansen", path=str(p), meta={"vars": list(data.columns), "det_order": det_order, "k_ar_diff": k_ar_diff})]
