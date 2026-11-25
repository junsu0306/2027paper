"""E01 (Research_Plan 실험 1, 가설 A): skew at three places F / Q / Q(F + eta), and its curve over the noise size.

Jobs: "F::<model>" (FP32 model) and "Q::<model>::<recipe>" (Q, Qw = weights only, and every Q(F + eta) condition).
QCB models are measured on their own target recipe(s); normals and controls on all seven recipes.
Conditions are in CONDITIONS() and prediction.md; they were fixed before the first run.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import data, deploy, grids, measures as M  # noqa: E402
from qcbobs.models import RECIPES, load, trigger  # noqa: E402
from qcbobs.perturb import perturbed_q  # noqa: E402

COHORT = ROOT / "cohorts/batch1.csv"
UNIFORM_C = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0)
GAUSS_C = (1 / 8, 1 / 4, 1 / 2, 1.0)
GSHIFT_C = (1 / 16, 1 / 8, 1 / 4, 1 / 2)
M24_C = (1 / 4, 1 / 2, 1.0)            # uniform sizes that also get M2 / M4 (seed 0)
REGION_C = (1 / 2, 1.0)
REGIONS = ("front", "middle", "back")  # thirds of the Conv2d / Linear list (module order)


def slow(recipe):  # train-mode IAO deployments re-estimate BN on 50k training images (~14 s each)
    return recipe.startswith("iao")


def CONDITIONS(recipe):
    s = slow(recipe)
    out = []
    for c in UNIFORM_C:
        for seed in range(4 if s else 8):
            out.append({"kind": "uniform", "c": c, "seed": seed, "region": "all", "m24": seed == 0 and c in M24_C})
    for c in (GAUSS_C[1:] if s else GAUSS_C):
        for seed in range(2 if s else 4):
            out.append({"kind": "gauss", "c": c, "seed": seed, "region": "all", "m24": False})
    for seed in range(4 if s else 8):
        out.append({"kind": "stoch", "c": None, "seed": seed, "region": "all", "m24": False})
    for c in GSHIFT_C:
        out.append({"kind": "gshift", "c": c, "seed": 0, "region": "all", "m24": False})
    for c in REGION_C:
        for reg in REGIONS:
            for seed in range(1 if s else 2):
                out.append({"kind": "uniform", "c": c, "seed": seed, "region": reg, "m24": False})
    return out


def cohort():
    return pd.read_csv(COHORT)


def jobs():
    out = []
    for _, r in cohort().iterrows():
        out.append({"id": f"F::{r.model_key}", "priority": 0})
        for rec in r.recipes.split(","):
            pri = 1 if r.group == "QCB" else 2
            out.append({"id": f"Q::{r.model_key}::{rec}", "priority": pri + (2 if slow(rec) else 0)})
    return out


def _base(r, K):
    return {"model_key": r.model_key, "group": r.group, "cell": r.cell, "family": r.family,
            "target": None if r.group != "QCB" else int(r.target), "K": K, "parent_key": r.parent_key}


def run(job, dev):
    parts = job["id"].split("::")
    r = cohort().set_index("model_key").loc[parts[1]]
    r["model_key"] = parts[1]
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load("CIFAR-10")
    x01 = imgs.probe.to(dev)
    probe = norm(x01)
    test = norm(imgs.test.to(dev))
    trig_name = r.trigger if r.group == "QCB" else "Patch6x6"
    trig = norm(trigger(imgs.test.to(dev), trig_name))
    target = None if r.group != "QCB" else int(r.target)
    base = _base(r, K) | {"trigger_used": trig_name}
    F_pred = M.logits(F, probe).argmax(1)

    if parts[0] == "F":
        t0 = time.time()
        row = base | {"place": "F", "recipe": None} | M.acc_asr(F, test, imgs.test_y, trig, target, K)
        row |= {"probe_hist": M.hist(F_pred, K), "prop": M.prop(F, probe, K),
                "m4": M.dropout(F, probe, K, F_pred), "m2": M.input_noise(F, x01, norm, K, F_pred), "sec": time.time() - t0}
        return [row]

    recipe = parts[2]
    rep, layers, deltas = grids.replica(F, recipe, dev)
    gs = grids.grids(layers, recipe)
    W = {n: m.weight.detach().clone() for n, m in layers}
    Q0 = {lg.name: lg.quant(W[lg.name]) for lg in gs}
    nl = len(layers)
    region_mask = {"all": None,
                   "front": [i < nl // 3 for i in range(nl)],
                   "middle": [nl // 3 <= i < 2 * nl // 3 for i in range(nl)],
                   "back": [i >= 2 * nl // 3 for i in range(nl)]}
    on_target = r.group == "QCB"
    rb = base | {"recipe": recipe, "recipe_bits": RECIPES[recipe]["bits"], "on_target": on_target}
    rows = []

    def measure(place, qw, act=True, cond=None, m24=False):
        t0 = time.time()
        deploy.set_weights(layers, qw)
        dm = deploy.build(rep, recipe, imgs, norm, dev, act=act)
        pred = M.logits(dm, probe).argmax(1)
        row = rb | {"place": place, "act_quant": act} | (cond or {}) | M.acc_asr(dm, test, imgs.test_y, trig, target, K)
        row |= {"probe_hist": M.hist(pred, K), "m1": M.transition(F_pred, pred, K), "prop": M.prop(dm, probe, K),
                "changed_frac": grids.changed_fraction(layers, gs, Q0, qw)}
        if m24:
            row |= {"m4": M.dropout(dm, probe, K, pred), "m2": M.input_noise(dm, x01, norm, K, pred)}
        del dm
        row["sec"] = time.time() - t0
        rows.append(row)

    measure("Q", Q0, m24=True)
    measure("Qw", Q0, act=False)
    for cond in CONDITIONS(recipe):
        qw = perturbed_q(W, gs, cond["kind"], cond["c"] or 0.0, 1000 + cond["seed"], region_mask[cond["region"]])
        measure("QN", qw, cond={k: v for k, v in cond.items() if k != "m24"}, m24=cond["m24"])
    deploy.set_weights(layers, W)
    return rows
