"""Loading run results and small helpers for the analyze scripts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .env import ROOT


def rows(run_id: str) -> list[dict]:
    out = []
    for p in sorted((ROOT / "runs" / run_id / "results").glob("*.jsonl")):
        with p.open() as fh:
            out += [json.loads(line) for line in fh]
    return out


def errors(run_id: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((ROOT / "runs" / run_id / "results").glob("*.error.json"))]


def progress(run_id: str, n_jobs: int) -> str:
    d = ROOT / "runs" / run_id / "results"
    done = len(list(d.glob("*.jsonl")))
    err = len(list(d.glob("*.error.json")))
    return f"{done}/{n_jobs} jobs done, {err} errors"


def gap_score(v, k_star=None):
    """max - median of the other classes; argmax."""
    v = np.asarray(v, dtype=float)
    k = int(np.nanargmax(v))
    return float(v[k] - np.nanmedian(np.delete(v, k))), k


def md_table(df: pd.DataFrame, floatfmt=".3f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(format(v, floatfmt) if isinstance(v, (float, np.floating)) and not pd.isna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
