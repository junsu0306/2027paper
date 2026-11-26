"""E27: replication of the E24 head finding (QuEST W4A4, iao4, head input, all-ones start: the F candidate speeds up far
more than the other classes in Q) on models E24 never saw, with the reading fixed in prediction.md beforehand.

Job R::<model>: recipe iao4 for every model (the QuEST W4A4 recipe).
  F: MM-BD (MVD baselines.run_mmbd, the E07 call) -> margins; candidate = argmax margin (the E23/E24 rule).
  F and Q (deploysim iao4, the E24 call): calibrated convergence steps (E24.ccs, SPD sec. 6.1) at the head input from
  the all-ones start and from the mean clean head input of F (E24.mean_input on the test split).
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, data, deploy, env, measures as M  # noqa: E402
from qcbobs.models import RECIPES, load  # noqa: E402

REC = "iao4"


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E24 = _mod("e24", "experiments/E24_convergence_by_layer/exp.py")
E07 = _mod("e07", "experiments/E07_baselines_by_state/exp.py")


def models():
    d = cohort.load_all()
    q = d[(d.group == "QCB") & d.cell.str.contains("QuEST-W4A4")]
    n_c10 = d[d.batch.isin(["batch1", "batch2"]) & (d.group != "QCB")]
    n_x = d[(d.batch == "batch_xn") & d.dataset.isin(["CIFAR-100", "GTSRB"])]
    import pandas as pd
    return pd.concat([q, n_c10, n_x], ignore_index=True)


def jobs():
    out = []
    for _, r in models().iterrows():
        anchor = r.batch == "batch1"
        out.append({"id": f"R::{r.model_key}", "priority": (0 if anchor else 1) + (0 if r.group == "QCB" else 1)})
    return out


def mmbd(F, norm, K, imgs, dev):
    env.setup()
    from multiview_detection import baselines as B
    ctx = E07.Ctx(E07.Pix(F, norm).to(dev).eval(), K, E07.pool_u8(imgs), dev)
    torch.manual_seed(0)
    t0 = time.time()
    r = B.run_mmbd(ctx, E07._Deadline(), lambda: None)
    return {k: r.get(k) for k in ("status", "target", "raw_score", "native_alarm", "native_p_value", "margins")} | {
        "sec": round(time.time() - t0, 1)}


def head_ccs(model, F, clean, K, dev):
    out = {}
    starts = {"linear/ones": torch.ones(512, device=dev), "linear/mean": E24.mean_input(F, M.resolve_name(F, "linear"), clean)}
    for cfg, z0 in starts.items():
        t0 = time.time()
        out[cfg] = E24.ccs(model, "linear", z0, K, dev) | {"sec": round(time.time() - t0, 1)}
    return out


def run(job, dev):
    _, key = job["id"].split("::")
    r = cohort.row(key)
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load(r.dataset)
    qcb = r.group == "QCB"
    clean = norm(imgs.test.to(dev))
    mb = mmbd(F, norm, K, imgs, dev)
    cand = int(np.argmax(mb["margins"])) if mb.get("margins") else None
    head = {"model_key": key, "batch": r.batch, "dataset": r.dataset, "group": r.group, "cell": r.cell, "family": r.family,
            "recipe": REC, "target": int(r.target) if qcb else None, "candidate_mmbd": cand, "K": K}
    rows = [head | {"state": "F", "mmbd": mb, "res": head_ccs(F, F, clean, K, dev)}]
    DS = deploy._ds()
    q = DS.deployed(F, RECIPES[REC]["view_key"], norm(imgs.cal.to(dev)), train_x=deploy.train_arg(REC, imgs, norm),
                    device=str(dev)).eval()
    for p_ in q.parameters():
        p_.requires_grad_(False)
    rows.append(head | {"state": "Q", "res": head_ccs(q, F, clean, K, dev)})
    return rows
