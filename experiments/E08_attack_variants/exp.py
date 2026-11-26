"""E08 readout on the received A5000 variants (cohorts/e08.csv; prediction.md written 2026-11-25, before generation).

The measurements are the registered batch-1 code, unchanged, pointed at the E08 cohort:
  E05  component factorial (which deployment component turns the backdoor on; ProP carrier)   job  C@C::<key>::<recipe>
  E06  switch anatomy (alpha / bits / BN-mix curves, single-unit sufficiency and necessity)    job  S@S::<key>::<recipe>
  E07  NC, MM-BD, SPD on F, the switch-only state and Q                                         job  K@K::<key>::<recipe>
  E04  S1-S4 (ProP, noise collapse, Delta clean ratio, in-cell budget)                          job  V@V::<key>::<recipe>
Each module's COHORT is replaced by cohorts/e08.csv at import; nothing else changes. Thresholds and normals for the
comparison come from the earlier runs (batch1 normals of E05-E07, batch1 thresholds of E04).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "cohorts/e08.csv"
SRC = {"C": "E05_component_factorial", "S": "E06_switch_anatomy", "K": "E07_baselines_by_state", "V": "E04_heldout_validation"}
_MOD = {}


def mod(k):
    if k not in _MOD:
        if k == "V":
            os.environ["E04_COHORT"] = str(COHORT)
        p = ROOT / "experiments" / SRC[k] / "exp.py"
        spec = importlib.util.spec_from_file_location(f"e08_{SRC[k]}", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        m.COHORT = COHORT
        _MOD[k] = m
    return _MOD[k]


def jobs():
    out = []
    for k in ("C", "S", "K", "V"):
        for j in mod(k).jobs():
            out.append({"id": f"{k}@{j['id']}", "priority": j["priority"] + {"C": 0, "S": 0, "K": 1, "V": 2}[k]})
    return out


def run(job, dev):
    k, inner = job["id"].split("@", 1)
    rows = mod(k).run({**job, "id": inner}, dev)
    for r in rows:
        r["src"] = SRC[k]
    return rows
