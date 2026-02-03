from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

@dataclass(frozen=True)
class FigureSpec:
    title: str
    xlabel: str = ""
    ylabel: str = ""

def save_timeseries_png(
    series: pd.Series,
    path: Path,
    spec: FigureSpec,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(series.index, series.values)
    ax.set_title(spec.title)
    ax.set_xlabel(spec.xlabel)
    ax.set_ylabel(spec.ylabel)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
