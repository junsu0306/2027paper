"""E19 (Research_Plan 실험 6, 가설 G): does the search for a trigger converge unusually fast toward the target class, in F
and in the switched state?

States: F and S (as E18: recipe-quantized weights + activation clipping at the calibrated range; iao with BN stored).
Search per class k (Neural Cleanse form, fixed penalty): mask m = sigmoid(a) [1 x H x W], pattern p = sigmoid(b),
x' = (1 - m) x + m p, loss = CE(model(x'), k) + LAM * mean(m), Adam lr .1, from a = -3, b = 0, on the first N_IMG probe
images whose label != k. Recorded: the first step at which >= 90 % of them go to k (STEPS + 1 if never) and the mask mean
at that step.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, data, measures as M  # noqa: E402
from qcbobs.models import ALL_RECIPES, load, trigger  # noqa: E402

N_IMG = 50
STEPS = 300
LAM = 1e-2


def models():
    d = cohort.load_all()
    return d[d.batch == "batch1"]


def jobs():
    out = []
    for _, r in models().iterrows():
        recs = r.recipes.split(",") if r.group == "QCB" else list(ALL_RECIPES)
        for rec in recs:
            out.append({"id": f"G::{r.model_key}::{rec}", "priority": 0 if r.group == "QCB" else 1})
    return out


def search(model, norm, x, k):
    a = torch.full((1, 1, *x.shape[2:]), -3.0, device=x.device, requires_grad=True)
    b = torch.zeros((1, 3, *x.shape[2:]), device=x.device, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=0.1)
    tgt = torch.full((len(x),), k, device=x.device, dtype=torch.long)
    for t in range(1, STEPS + 1):
        m, p = torch.sigmoid(a), torch.sigmoid(b)
        out = model(norm((1 - m) * x + m * p))
        loss = Fn.cross_entropy(out, tgt) + LAM * m.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if float((out.argmax(1) == k).float().mean()) >= 0.9:
            return t, float(m.mean())
    return STEPS + 1, float(torch.sigmoid(a).mean())


def _e18():
    import importlib.util
    spec = importlib.util.spec_from_file_location("e18exp", ROOT / "experiments/E18_boundary_distance/exp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(job, dev):
    _, key, recipe = job["id"].split("::")
    E18 = _e18()
    r = cohort.row(key)
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load(r.dataset)
    qcb = r.group == "QCB"
    target = int(r.target) if qcb else None
    S = E18.switched_state(F, recipe, imgs, norm, dev)
    test, trig = norm(imgs.test.to(dev)), norm(trigger(imgs.test.to(dev), r.trigger if qcb else "Patch6x6"))
    asr = {n: M.acc_asr(m, test, imgs.test_y, trig, target, K).get("asr") for n, m in (("F", F), ("S", S))}
    x01 = imgs.probe.to(dev)
    y = torch.from_numpy(np.asarray(E18.data_labels(r.dataset))).to(dev)[: len(x01)]
    res = {}
    for name, m in (("F", F), ("S", S)):
        steps, mask = [], []
        for k in range(K):
            x = x01[torch.nonzero(y != k).flatten()[:N_IMG]]
            s, mm = search(m, norm, x, k)
            steps.append(s)
            mask.append(mm)
        res[name] = (steps, mask)
    return [{"model_key": key, "group": r.group, "cell": r.cell, "family": r.family, "recipe": recipe, "target": target, "K": K,
             "asr_F": asr["F"], "asr_S": asr["S"], "steps_F": res["F"][0], "mask_F": res["F"][1],
             "steps_S": res["S"][0], "mask_S": res["S"][1]}]
