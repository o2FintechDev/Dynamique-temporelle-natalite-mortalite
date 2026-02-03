from __future__ import annotations
import numpy as np
import pandas as pd

from statsmodels.tsa.api import VAR

from src.agent.schemas import Artefact
from src.agent.tools import ToolContext, _next_id
from src.visualization.charts import save_timeseries_png, FigureSpec
from src.visualization.tables import save_table_csv

def irf_fevd_artefacts(ctx: ToolContext, vars: list[str], max_lag: int = 6, horizon: int = 24) -> list[Artefact]:
    df: pd.DataFrame = ctx.memory["df_ms"]
    data = df[vars].dropna()
    artefacts: list[Artefact] = []

    try:
        model = VAR(data)
        sel = model.select_order(maxlags=max_lag)
        lag = int(sel.aic) if sel.aic is not None else max(1, min(2, max_lag))
        fit = model.fit(lag)

        irf = fit.irf(horizon)
        fevd = fit.fevd(horizon)

        # IRF: figure proxy (norme des réponses par horizon sur 1 choc = première variable)
        shock = 0
        resp_norm = np.linalg.norm(irf.irfs[:, :, shock], axis=1)  # horizon+1
        aid1 = _next_id(ctx, "fig")
        p1 = ctx.run_dirs.figures_dir / f"{aid1}_irf_norm.png"
        save_timeseries_png(pd.Series(resp_norm, index=np.arange(len(resp_norm))), p1, FigureSpec(title="IRF (norme des réponses) — choc var[0]", xlabel="Horizon", ylabel="||IRF||"))
        artefacts.append(Artefact(artefact_id=aid1, kind="figure", name="irf_norm", path=str(p1), meta={"lag": lag, "horizon": horizon}))

        # FEVD: table (part de variance sur variable[0])
        fe = fevd.decomp  # (neqs, horizon, neqs)
        tab = pd.DataFrame({
            "horizon": np.arange(1, fe.shape[1] + 1),
            "fevd_var0_from_shock0": fe[0, :, 0].astype(float),
        })
        aid2 = _next_id(ctx, "table")
        p2 = ctx.run_dirs.tables_dir / f"{aid2}_fevd.csv"
        save_table_csv(tab, p2)
        artefacts.append(Artefact(artefact_id=aid2, kind="table", name="fevd_var0", path=str(p2), meta={"lag": lag, "horizon": horizon}))

    except Exception as e:
        aid = _next_id(ctx, "table")
        p = ctx.run_dirs.tables_dir / f"{aid}_irf_fevd_error.csv"
        save_table_csv(pd.DataFrame([{"error": str(e)}]), p)
        artefacts.append(Artefact(artefact_id=aid, kind="table", name="irf_fevd_error", path=str(p), meta={}))

    return artefacts
