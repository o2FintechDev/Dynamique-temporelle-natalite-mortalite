from __future__ import annotations
import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from src.agent.schemas import Artefact
from src.agent.tools import ToolContext, _next_id
from src.visualization.tables import save_table_csv

def engle_granger_artefacts(ctx: ToolContext, vars: list[str]) -> list[Artefact]:
    df: pd.DataFrame = ctx.memory["df_ms"]
    vars = [v for v in vars if v in df.columns]
    if len(vars) < 2:
        # artefact vide, mais traçable
        tab = pd.DataFrame([{"error": "Engle-Granger nécessite >=2 variables"}])
    else:
        base = vars[0]
        rows = []
        for v in vars[1:]:
            x = df[base].dropna()
            y = df[v].dropna()
            common = pd.concat([x, y], axis=1).dropna()
            try:
                stat, pval, _ = coint(common.iloc[:, 0], common.iloc[:, 1])
                rows.append({"x": base, "y": v, "coint_stat": float(stat), "pvalue": float(pval), "nobs": int(len(common))})
            except Exception as e:
                rows.append({"x": base, "y": v, "error": str(e)})
        tab = pd.DataFrame(rows)

    aid = _next_id(ctx, "table")
    p = ctx.run_dirs.tables_dir / f"{aid}_engle_granger.csv"
    save_table_csv(tab, p)
    return [Artefact(artefact_id=aid, kind="table", name="engle_granger", path=str(p), meta={"vars": vars})]

def johansen_artefacts(ctx: ToolContext, vars: list[str], det_order: int = 0, k_ar_diff: int = 4) -> list[Artefact]:
    df: pd.DataFrame = ctx.memory["df_ms"]
    data = df[vars].dropna()
    rows = []
    try:
        res = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
        # trace statistics + crit values
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
    return [Artefact(artefact_id=aid, kind="table", name="johansen", path=str(p), meta={"vars": vars, "det_order": det_order, "k_ar_diff": k_ar_diff})]
