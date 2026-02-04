# econometrics/impulse.py
from __future__ import annotations

import numpy as np
import pandas as pd

from statsmodels.tsa.api import VAR

from src.agent.schemas import Artefact
from src.agent.tools import ToolContext, _next_id
from src.visualization.charts import save_timeseries_png, FigureSpec
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


def irf_fevd_artefacts(ctx: ToolContext, vars: list[str], max_lag: int = 6, horizon: int = 24) -> list[Artefact]:
    data = _get_df(ctx, vars).dropna()
    artefacts: list[Artefact] = []

    if data.shape[1] < 2 or data.shape[0] < 50:
        aid = _next_id(ctx, "table")
        p = ctx.run_dirs.tables_dir / f"{aid}_irf_fevd_error.csv"
        save_table_csv(pd.DataFrame([{"error": "IRF/FEVD nécessite >=2 variables et nobs>=50"}]), p)
        return [Artefact(artefact_id=aid, kind="table", name="irf_fevd_error", path=str(p), meta={"vars": vars})]

    try:
        model = VAR(data)
        sel = model.select_order(maxlags=max_lag)
        lag = int(sel.aic) if sel.aic is not None else max(1, min(2, max_lag))
        fit = model.fit(lag)

        irf = fit.irf(horizon)
        fevd = fit.fevd(horizon)

        # IRF proxy traçable (norme des réponses à un choc sur var[0])
        shock = 0
        resp_norm = np.linalg.norm(irf.irfs[:, :, shock], axis=1)  # horizon+1

        aid1 = _next_id(ctx, "fig")
        p1 = ctx.run_dirs.figures_dir / f"{aid1}_irf_norm.png"
        save_timeseries_png(
            pd.Series(resp_norm, index=np.arange(len(resp_norm))),
            p1,
            FigureSpec(
                title=f"IRF (||réponse||) — choc {data.columns[0]}",
                xlabel="Horizon",
                ylabel="||IRF||",
            ),
        )
        artefacts.append(Artefact(artefact_id=aid1, kind="figure", name="irf_norm", path=str(p1), meta={"lag": lag, "horizon": horizon, "vars": list(data.columns)}))

        # FEVD: parts de variance de var[0] expliquée par chaque choc (au dernier horizon)
        last = fevd.decomp[-1, 0, :].astype(float)  # shocks
        tab = pd.DataFrame([{"target": data.columns[0], "horizon": horizon, **{f"shock_{c}": float(last[i]) for i, c in enumerate(data.columns)}}])

        aid2 = _next_id(ctx, "table")
        p2 = ctx.run_dirs.tables_dir / f"{aid2}_fevd.csv"
        save_table_csv(tab, p2)
        artefacts.append(Artefact(artefact_id=aid2, kind="table", name="fevd", path=str(p2), meta={"lag": lag, "horizon": horizon, "vars": list(data.columns)}))

        # Lag retenu (traçabilité)
        aid3 = _next_id(ctx, "table")
        p3 = ctx.run_dirs.tables_dir / f"{aid3}_irf_fevd_meta.csv"
        save_table_csv(pd.DataFrame([{"selected_lag_aic": lag, "max_lag": max_lag, "horizon": horizon}]), p3)
        artefacts.append(Artefact(artefact_id=aid3, kind="table", name="irf_fevd_meta", path=str(p3), meta={"vars": list(data.columns)}))

    except Exception as e:
        aid = _next_id(ctx, "table")
        p = ctx.run_dirs.tables_dir / f"{aid}_irf_fevd_error.csv"
        save_table_csv(pd.DataFrame([{"error": str(e)}]), p)
        artefacts.append(Artefact(artefact_id=aid, kind="table", name="irf_fevd_error", path=str(p), meta={}))

    return artefacts

