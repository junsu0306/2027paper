"""ST01 (2부 1단계): does the model "know" a deployment format? (docs/STAGED.md, prediction.md)

Normal training never sees the deployment grid, so for a normal model the actual rounding residual Delta = Q(W) - W should
act on the model like a residual with the same element magnitudes and random signs. A model tuned to the format (QAT,
the no-poison QCB pipelines, QCB with a clean-Q term) should be closer to F at the true rounding than at the sign-randomised
one; a QCB whose switch moves Q away from F on clean data (Ma) should be farther.

One job = one model; for every recipe g of the 7 (the defender does not know the format):
  W side (weights only: FP32 activations, BN stored; the replica of E16/E10 W:g):
    aligned     W + Delta = Q_g(W)
    sign:k      W + Delta * xi_k, xi_k random +-1 per element (k = 0..3, seed 1000 + k)
    shift:0.5   grid moved by half a step: Q_g(W - s/2) + s/2
  D side (full deployment, deploy.build on the same weights; 4-bit recipes efrap4, quanti4, iao4 only): the same 6 states.
Measures (label-free, detector_probe 1,000 images): mean KL(p_F || p_state) and top-1 disagreement with F.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as Fn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, data, deploy, grids, measures as M  # noqa: E402
from qcbobs.models import ALL_RECIPES, load  # noqa: E402

FULL = ("efrap4", "quanti4", "iao4")
N_SIGN = 4


def models():
    return cohort.load_all()


def jobs():
    out = []
    for _, r in models().iterrows():
        out.append({"id": f"A::{r.model_key}", "priority": (0 if r.batch == "batch1" else 1)})
    return out


@torch.no_grad()
def _cmp(model, xp, lpf, pf_arg):
    lp = Fn.log_softmax(M.logits(model, xp).float(), 1)
    kl = float((lpf.exp() * (lpf - lp)).sum(1).mean())
    dis = float((lp.argmax(1) != pf_arg).float().mean())
    return kl, dis


def run(job, dev):
    _, key = job["id"].split("::")
    r = cohort.row(key)
    F, norm, info = load(r.data_dir, dev)
    imgs = data.load(r.dataset)
    xp = norm(imgs.probe.to(dev))
    with torch.no_grad():
        lpf = Fn.log_softmax(M.logits(F, xp).float(), 1)
    pf_arg = lpf.argmax(1)
    head = {"model_key": key, "batch": r.batch, "group": r.group, "cell": r.cell, "family": r.family, "dataset": r.dataset,
            "arch": r.arch, "target_recipe": getattr(r, "target_recipe", None)}
    rows = []
    for g in ALL_RECIPES:
        rep, layers, _ = grids.replica(F, g, dev)
        gs = grids.grids(layers, g)
        W = {n: m.weight.detach().clone() for n, m in layers}
        Q = {lg.name: lg.quant(W[lg.name]) for lg in gs}
        states = {"aligned": Q}
        for k in range(N_SIGN):
            gen = torch.Generator(device=dev).manual_seed(1000 + k)
            states[f"sign:{k}"] = {n: W[n] + (Q[n] - W[n]) * (torch.randint(0, 2, W[n].shape, generator=gen, device=dev) * 2 - 1)
                                   for n in Q}
        states["shift:0.5"] = {lg.name: lg.quant(W[lg.name] - 0.5 * lg.scale) + 0.5 * lg.scale for lg in gs}
        for st, ws in states.items():
            deploy.set_weights(layers, ws)
            kl, dis = _cmp(rep, xp, lpf, pf_arg)
            rows.append(head | {"recipe": g, "side": "W", "state": st, "kl": kl, "dis": dis})
            if g in FULL:
                dm = deploy.build(rep, g, imgs, norm, dev)
                kl, dis = _cmp(dm, xp, lpf, pf_arg)
                rows.append(head | {"recipe": g, "side": "D", "state": st, "kl": kl, "dis": dis})
                del dm
        deploy.set_weights(layers, W)
        del rep
    return rows
