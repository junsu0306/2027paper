"""All selected models (cohorts/batch1.csv, batch2.csv, batch_x.csv, batch_xn.csv) in one table, same-weight duplicates dropped.

DROP: the second model_key of each same-weight pair (cohorts/DUPLICATES.md; rule: keep the first key in sort order).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DROP = {"QuEST-8__1e2a1ce507", "QuEST-W4A4__4413ec2fef"}


def load_all() -> pd.DataFrame:
    out = []
    for b in ("batch1", "batch2", "batch_x", "batch_xn"):
        d = pd.read_csv(ROOT / "cohorts" / f"{b}.csv")
        d["batch"] = b
        if "dataset" not in d:
            d["dataset"] = "CIFAR-10"
            d["arch"] = "ResNet18"
        out.append(d)
    d = pd.concat(out, ignore_index=True)
    return d[~d.model_key.isin(DROP)].reset_index(drop=True)


def row(key: str):
    d = load_all().set_index("model_key")
    return d.loc[key]
