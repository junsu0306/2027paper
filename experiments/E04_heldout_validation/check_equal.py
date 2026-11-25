#!/usr/bin/env python3
"""Equality check: E04's code on batch-1 models must reproduce the batch-1 numbers (S1 E01, S2 E01, S3 E02, S4 E03B2).
  E04_COHORT=cohorts/batch1.csv python3 check_equal.py <model_key> <recipe>"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "src"))
import exp  # noqa: E402
from qcbobs import env  # noqa: E402
env.backend()
key, rec = sys.argv[1], sys.argv[2]
t = time.time()
row = exp.run({"id": f"V::{key}::{rec}"}, torch.device("cuda"))[0]
th = json.loads((HERE / "thresholds.json").read_text())
e1 = pd.read_csv(HERE.parents[0] / "E01_three_places/results/rows.csv", low_memory=False)
q = e1[(e1.model_key == key) & (e1.recipe == rec)]
s1_ref = q[q.place == "Q"].skew_s100.iloc[0]
e2 = pd.read_csv(HERE.parents[0] / "E02_residual_direction/results/summary.csv")
s3_ref = e2[(e2.model_key == key) & (e2.recipe == rec)].clean_ratio.iloc[0]
e3 = pd.read_csv(HERE.parents[0] / "E03B2_budget_sweep/results/summary.csv")
e3 = e3[(e3.model_key == key) & (e3.recipe == rec)].set_index("form")
from qcbobs import readout as R  # noqa: E402
sh = [max(r["probe_hist"]) / sum(r["probe_hist"]) for r in R.rows("E01_20261125_001")
      if r["model_key"] == key and r["recipe"] == rec and r["place"] == "QN" and r.get("kind") == "uniform"
      and r.get("region") == "all" and r.get("c") == 1.0]
print(json.dumps({"model": key, "recipe": rec, "sec": round(time.time() - t, 1),
                  "S1": [row["S1"], s1_ref], "S2": [row["S2"], float(np.mean(sh))], "S3": [row["S3"], s3_ref],
                  "S4_flip": [row["S4_flip_b50"], e3.loc["flip", "b50"]], "S4_pgd": [row["S4_pgd_b50"], e3.loc["pgd", "b50"]],
                  "Q_asr": row["Q_asr"], "secs": {k: round(row[k], 1) for k in row if k.startswith("sec")}}, default=str))
