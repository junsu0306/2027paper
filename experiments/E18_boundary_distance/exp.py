"""E18 (Research_Plan 실험 4, 가설 D): does the minimal input perturbation to reach a class shrink after quantization only for
the target class?

States (differentiable): F, and S = recipe-quantized weights + activation clipping at the calibrated range (deploysim
placement; clip only, no activation rounding; iao recipes with BN stored). S switches on every batch-1 QCB family on its
own recipe (E05: weights for Ma / Tian / Qu-ANTI, clipping for CIFAR-10 QuEST-8; QuEST W4A4 partly).
Images: the first N_IMG probe images that F classifies correctly. For each class k != label: the minimal L2 norm (pixel
space [0, 1]) of a targeted perturbation, by bisection over the budget (BISECT steps in [0, HI]) with an L2 PGD of
PGD_STEPS steps at each budget; HI is reported as censored.
Statistic per class: median over images of d_S / d_F (images correct in both states, label != k).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as Fn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, components as C, data, deploy, grids, measures as M  # noqa: E402
from qcbobs.models import ALL_RECIPES, load, trigger  # noqa: E402

N_IMG = 50
HI = 2.0
BISECT = 10
PGD_STEPS = 10


def models():
    d = cohort.load_all()
    return d[d.batch == "batch1"]


def jobs():
    out = []
    for _, r in models().iterrows():
        recs = r.recipes.split(",") if r.group == "QCB" else list(ALL_RECIPES)
        for rec in recs:
            out.append({"id": f"D::{r.model_key}::{rec}", "priority": 0 if r.group == "QCB" else 1})
    return out


def switched_state(F, recipe, imgs, norm, dev):
    rep, layers, _ = grids.replica(F, recipe, dev)
    gs = grids.grids(layers, recipe)
    deploy.set_weights(layers, {lg.name: lg.quant(dict(layers)[lg.name].weight.detach()) for lg in gs})
    dm = C.build(rep, recipe, imgs, norm, dev, bn="stored" if recipe.startswith("iao") else "recipe")
    C.set_acts(C.act_quantizers(dm), "clip:1")
    for p in dm.parameters():
        p.requires_grad_(False)
    return dm.eval()


def _pgd(model, norm, x, k, eps, steps):
    d = torch.zeros_like(x, requires_grad=True)
    alpha = (2.5 * eps / steps).view(-1, 1, 1, 1)
    tgt = torch.full((len(x),), k, device=x.device, dtype=torch.long)
    for _ in range(steps):
        loss = Fn.cross_entropy(model(norm((x + d).clamp(0, 1))), tgt)
        g, = torch.autograd.grad(loss, d)
        with torch.no_grad():
            gn = g.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
            d -= alpha * g / gn
            dn = d.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
            d *= torch.clamp(eps.view(-1, 1, 1, 1) / dn.clamp_min(1e-12), max=1.0)
    with torch.no_grad():
        ok = model(norm((x + d).clamp(0, 1))).argmax(1) == k
    return ok


def min_dist(model, norm, x, k):
    lo = torch.zeros(len(x), device=x.device)
    hi = torch.full((len(x),), HI, device=x.device)
    found = _pgd(model, norm, x, k, hi.clone(), PGD_STEPS)
    for _ in range(BISECT):
        mid = (lo + hi) / 2
        ok = _pgd(model, norm, x, k, mid, PGD_STEPS)
        hi = torch.where(ok, mid, hi)
        lo = torch.where(ok, lo, mid)
    return torch.where(found, hi, torch.full_like(hi, float("inf")))


def run(job, dev):
    _, key, recipe = job["id"].split("::")
    r = cohort.row(key)
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load(r.dataset)
    qcb = r.group == "QCB"
    target = int(r.target) if qcb else None
    S = switched_state(F, recipe, imgs, norm, dev)
    test, trig = norm(imgs.test.to(dev)), norm(trigger(imgs.test.to(dev), r.trigger if qcb else "Patch6x6"))
    asr = {n: M.acc_asr(m, test, imgs.test_y, trig, target, K).get("asr") for n, m in (("F", F), ("S", S))}
    x01, y = imgs.probe.to(dev), None
    with torch.no_grad():
        yF = M.logits(F, norm(x01)).argmax(1)
    labels = torch.from_numpy(__import__("numpy").asarray(data_labels(r.dataset))).to(dev)[: len(x01)]
    idx = torch.nonzero(yF == labels).flatten()[:N_IMG]
    x, y = x01[idx], labels[idx]
    with torch.no_grad():
        yS = M.logits(S, norm(x)).argmax(1)
    both = (yS == y)
    dist = {}
    for name, m in (("F", F), ("S", S)):
        dist[name] = torch.stack([min_dist(m, norm, x, k) for k in range(K)], 1)   # [N, K]
    rows = [{"model_key": key, "group": r.group, "cell": r.cell, "family": r.family, "recipe": recipe, "target": target, "K": K,
             "asr_F": asr["F"], "asr_S": asr["S"], "n_img": int(len(x)), "both_correct": int(both.sum()),
             "labels": y.tolist(), "dist_F": dist["F"].tolist(), "dist_S": dist["S"].tolist()}]
    return rows


_LABELS = {}


def data_labels(dataset):
    """Labels of the probe images (detector_probe role), same order as data.load(...).probe."""
    if dataset not in _LABELS:
        import json
        import numpy as np
        from qcbobs.env import DATA
        name = data.DS_FILE[dataset]
        z = np.load(DATA / "datasets/main" / f"{name}.npz")
        roles = json.loads((DATA / "datasets/main/manifests" / name / "split_base.json").read_text())["datasets"][name]["roles"]
        idx = np.asarray(roles["detector_probe"]["indices"][:1000])
        _LABELS[dataset] = z["y_test"][idx].astype(np.int64)
    return _LABELS[dataset]
