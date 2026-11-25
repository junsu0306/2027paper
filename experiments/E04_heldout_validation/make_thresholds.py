#!/usr/bin/env python3
"""Fix the E04 decision thresholds from batch-1 results only (run before any batch-2 measurement).

Per recipe, from batch-1 normal + control models (never QCB):
  S1, S2, S3: fire if value > max over batch-1 normal + control
  S4 flip / pgd: fire if budget50_min < min over batch-1 normal + control (E03B2 had 14 such models per recipe)
Also stores the batch-1 QCB values per cell for the record. Writes thresholds.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402
from qcbobs.measures import skew  # noqa: E402


def main():
    rows = R.rows("E01_20261125_001")
    s1, s2 = [], []
    for r in rows:
        if r["place"] == "Q":
            s1.append({"model_key": r["model_key"], "group": r["group"], "cell": r["cell"], "recipe": r["recipe"],
                       "v": skew(r["prop"]["s100"])})
        if r["place"] == "QN" and r.get("kind") == "uniform" and r.get("region") == "all" and r.get("c") == 1.0:
            h = np.asarray(r["probe_hist"], float)
            s2.append({"model_key": r["model_key"], "group": r["group"], "cell": r["cell"], "recipe": r["recipe"],
                       "v": h.max() / h.sum()})
    s1 = pd.DataFrame(s1)
    s2 = pd.DataFrame(s2).groupby(["model_key", "group", "cell", "recipe"]).v.mean().reset_index()
    e2 = pd.read_csv(ROOT / "experiments/E02_residual_direction/results/summary.csv").rename(columns={"clean_ratio": "v"})
    e3 = pd.read_csv(ROOT / "experiments/E03B2_budget_sweep/results/summary.csv")
    out = {"source_runs": ["E01_20261125_001", "E02_20261125_001", "E03B2_20261125_001"], "rule": __doc__, "recipes": {}}
    for rec in sorted(s1.recipe.unique()):
        d = {}
        for name, df, hi in (("S1", s1, True), ("S2", s2, True), ("S3", e2, True),
                             ("S4_flip", e3[e3.form == "flip"].rename(columns={"b50_min": "v"}), False),
                             ("S4_pgd", e3[e3.form == "pgd"].rename(columns={"b50_min": "v"}), False)):
            x = df[df.recipe == rec]
            o, q = x[x.group != "QCB"], x[x.group == "QCB"]
            d[name] = {"direction": "above" if hi else "below", "threshold": float(o.v.max() if hi else o.v.min()),
                       "n_reference": int(len(o)), "reference_median": float(o.v.median()),
                       "batch1_qcb": {c: sorted(round(float(v), 4) for v in g.v) for c, g in q.groupby("cell")}}
        out["recipes"][rec] = d
    (HERE / "thresholds.json").write_text(json.dumps(out, indent=1))
    for rec, d in out["recipes"].items():
        print(rec, {k: (v["direction"], round(v["threshold"], 3), v["n_reference"]) for k, v in d.items()})


if __name__ == "__main__":
    main()
