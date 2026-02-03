from __future__ import annotations
import pandas as pd
import plotly.express as px

def ts_line_chart(wide: pd.DataFrame, title: str = "Séries temporelles"):
    if wide is None or wide.empty:
        fig = px.line(title=title)
        return fig
    df = wide.copy()
    df = df.reset_index().melt(id_vars=["date"], var_name="variable", value_name="value")
    fig = px.line(df, x="date", y="value", color="variable", title=title)
    fig.update_layout(legend_title_text="Variable")
    return fig
