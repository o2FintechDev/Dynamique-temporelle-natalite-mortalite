from __future__ import annotations
import pandas as pd
import plotly.express as px

def line_series(df: pd.DataFrame, y: str, title: str | None = None):
    d = df[[y]].reset_index().rename(columns={"index": "Date"})
    fig = px.line(d, x="Date", y=y, title=title or f"Série: {y}")
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig

def compare_two(df: pd.DataFrame, y1: str, y2: str, title: str | None = None):
    d = df[[y1, y2]].reset_index().rename(columns={"index": "Date"})
    d = d.melt(id_vars="Date", value_vars=[y1, y2], var_name="variable", value_name="value")
    fig = px.line(d, x="Date", y="value", color="variable", title=title or f"Comparaison: {y1} vs {y2}")
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig

def scatter(df: pd.DataFrame, x: str, y: str, title: str | None = None):
    d = df[[x, y]].dropna().reset_index()
    fig = px.scatter(d, x=x, y=y, trendline="ols", title=title or f"Relation: {x} vs {y}")
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig
