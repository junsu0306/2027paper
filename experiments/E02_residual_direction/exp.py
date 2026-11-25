"""E02 (Research_Plan 실험 2, 가설 C): output along F + t * Delta, against sign-shuffled directions of the same size.

Weights only (FP32 activations, BN in eval), in the recipe's quantization space; Delta = quant(W) - W on the recipe's grid.
Directions: "real" and 16 Rademacher sign flips of Delta (|Delta| kept element-wise).
Per (direction, t): clean test images -> agreement with t = 0, mean L2 logit change, prediction histogram;
triggered test images (non-target labels for QCB) -> prediction histogram, mean logit vector, mean margin per class
(logit_k - max_{j != k} logit_j).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import data, deploy, grids, measures as M  # noqa: E402
from qcbobs.models import RECIPES, load, trigger  # noqa: E402

COHORT = ROOT / "cohorts/batch1.csv"
T = tuple(round(0.05 * i, 2) for i in range(21))
T8_EXTRA = (1.5, 2.0)
N_RANDOM = 16


def cohort():
    return pd.read_csv(COHORT)


def jobs():
    out = []
    for _, r in cohort().iterrows():
        for rec in r.recipes.split(","):
            out.append({"id": f"D::{r.model_key}::{rec}", "priority": 0 if r.group == "QCB" else 1})
    return out


@torch.no_grad()
def _stats(model, clean, trig, base_clean_logits, K):
    lc = M.logits(model, clean)
    lt = M.logits(model, trig)
    top2 = lt.topk(2, dim=1)
    # margin of class k: logit_k - max_{j != k} logit_j
    mx = torch.where(torch.arange(K, device=lt.device)[None] == top2.indices[:, :1], top2.values[:, 1:2], top2.values[:, :1])
    margin = (lt - mx).mean(0)
    return {"clean_agree": float((lc.argmax(1) == base_clean_logits.argmax(1)).float().mean()),
            "clean_dlogit_l2": float((lc - base_clean_logits).norm(dim=1).mean()),
            "clean_hist": M.hist(lc.argmax(1), K), "trig_hist": M.hist(lt.argmax(1), K),
            "trig_mean_logit": lt.mean(0).tolist(), "trig_margin": margin.tolist()}


def run(job, dev):
    _, key, recipe = job["id"].split("::")
    r = cohort().set_index("model_key").loc[key]
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load("CIFAR-10")
    target = None if r.group != "QCB" else int(r.target)
    keep = torch.ones(len(imgs.test_y), dtype=torch.bool) if target is None else imgs.test_y != target
    clean = norm(imgs.test.to(dev))
    trig = norm(trigger(imgs.test[keep].to(dev), r.trigger if r.group == "QCB" else "Patch6x6"))
    rep, layers, _ = grids.replica(F, recipe, dev)
    gs = grids.grids(layers, recipe)
    W = {n: m.weight.detach().clone() for n, m in layers}
    D = {lg.name: lg.quant(W[lg.name]) - W[lg.name] for lg in gs}
    base_logits = M.logits(rep, clean)
    Ts = T + (T8_EXTRA if RECIPES[recipe]["bits"] == 8 else ())
    head = {"model_key": key, "group": r.group, "cell": r.cell, "family": r.family, "target": target, "K": K,
            "recipe": recipe, "recipe_bits": RECIPES[recipe]["bits"], "parent_key": r.parent_key,
            "delta_l2": float(sum(float(d.double().pow(2).sum()) for d in D.values()) ** .5),
            "delta_over_s_mean": float(np.mean([float((D[lg.name].abs() / lg.scale).mean()) for lg in gs]))}
    rows = []
    for d in range(N_RANDOM + 1):
        if d == 0:
            dirv = D
        else:
            g = torch.Generator(device=dev).manual_seed(5000 + d)
            dirv = {n: v * (torch.randint(0, 2, v.shape, generator=g, device=dev) * 2 - 1) for n, v in D.items()}
        for t in Ts:
            t0 = time.time()
            deploy.set_weights(layers, {n: W[n] + t * dirv[n] for n in W})
            rows.append(head | {"direction": "real" if d == 0 else f"rand{d}", "t": t} | _stats(rep, clean, trig, base_logits, K)
                        | {"sec": time.time() - t0})
    deploy.set_weights(layers, W)
    return rows
