"""ST02 (2부 1단계, 다시 설계): deployment exchangeability test (prediction.md, docs/STAGED.md).

Principle: a model trained without knowing the deployment D is indifferent to the direction of every parameter residual
D makes. For each component c the true residual r_c is compared with N residuals of the same element magnitudes and
random signs (r_c * xi), through label-free functionals on detector_probe (1,000 images):
  kl   mean KL(p_F || p_X)       distance to F (anti: true residual farther from F; F+Q-tuned: closer)
  ent  mean entropy of p_X       F-free (Q-only tuning: the true residual leaves a more confident model)
Components (one job = one model, all components):
  W:<g>   weight rounding residual of grid g, weights only (FP32 activations, BN stored), all 7 recipe grids; N_W draws
  D:<g>   the same weights through the full deployment (deploy.build), 4-bit recipes only; first N_D draws
  BN      BN running statistics re-estimated (cumulative, FP32) on the defender's unlabeled images (test 2000-3999,
          E10) with training augmentation (E21 A 'aug' procedure); residual on running_mean and log running_var,
          random signs per element; N_BN draws
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as Fn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, components as C, data, deploy, grids, measures as M  # noqa: E402
from qcbobs.models import ALL_RECIPES, load  # noqa: E402

FULL = ("efrap4", "quanti4", "iao4")
N_W, N_D, N_BN = 16, 8, 16


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E10 = _mod("e10", "experiments/E10_switch_closure/exp.py")
E21 = _mod("e21", "experiments/E21_concealment_breakers/exp.py")


def models():
    return cohort.load_all()


def jobs():
    return [{"id": f"X::{r.model_key}", "priority": (0 if r.group == "QCB" else 1)} for _, r in models().iterrows()]


@torch.no_grad()
def _m(model, xp, lpf):
    lp = Fn.log_softmax(M.logits(model, xp).float(), 1)
    p = lp.exp()
    return float((lpf.exp() * (lpf - lp)).sum(1).mean()), float(-(p * lp).sum(1).mean())


def _sign(shape, gen, dev):
    return torch.randint(0, 2, shape, generator=gen, device=dev) * 2 - 1


def run(job, dev):
    _, key = job["id"].split("::")
    r = cohort.row(key)
    F, norm, info = load(r.data_dir, dev)
    imgs = data.load(r.dataset)
    xp = norm(imgs.probe.to(dev))
    with torch.no_grad():
        lpf = Fn.log_softmax(M.logits(F, xp).float(), 1)
    head = {"model_key": key, "batch": r.batch, "group": r.group, "cell": r.cell, "family": r.family, "dataset": r.dataset,
            "arch": r.arch}
    rows = []

    def rec(comp, st, model):
        kl, ent = _m(model, xp, lpf)
        rows.append(head | {"comp": comp, "state": st, "kl": kl, "ent": ent})

    kl0, ent0 = _m(F, xp, lpf)
    rows.append(head | {"comp": "F", "state": "F", "kl": kl0, "ent": ent0})
    # ---- weight rounding, all grids
    for g in ALL_RECIPES:
        rep, layers, _ = grids.replica(F, g, dev)
        gs = grids.grids(layers, g)
        W = {n: m.weight.detach().clone() for n, m in layers}
        Q = {lg.name: lg.quant(W[lg.name]) for lg in gs}
        states = [("aligned", Q)]
        for k in range(N_W):
            gen = torch.Generator(device=dev).manual_seed(1000 + k)
            states.append((f"sign:{k}", {n: W[n] + (Q[n] - W[n]) * _sign(W[n].shape, gen, dev) for n in Q}))
        for i, (st, ws) in enumerate(states):
            deploy.set_weights(layers, ws)
            rec(f"W:{g}", st, rep)
            if g in FULL and i <= N_D:
                dm = deploy.build(rep, g, imgs, norm, dev)
                rec(f"D:{g}", st, dm)
                del dm
        deploy.set_weights(layers, W)
        del rep
    # ---- BN statistics
    bns = C.bn_layers(F)
    if bns:
        u8 = E10.defender_images(r.dataset).train_u8[:2000]
        Fre = E21.bn_reest(F, E21.batches(u8, norm, dev, aug=True))
        rec("BN", "aligned", Fre)
        st0, st1 = C.bn_stats(F), C.bn_stats(Fre)
        X = copy.deepcopy(F).eval()
        xb = dict(C.bn_layers(X))
        for k in range(N_BN):
            gen = torch.Generator(device=dev).manual_seed(2000 + k)
            with torch.no_grad():
                for n, mod in xb.items():
                    m0, v0 = st0[n]
                    m1, v1 = st1[n]
                    mod.running_mean.copy_(m0 + (m1 - m0) * _sign(m0.shape, gen, dev))
                    lv0, lv1 = v0.clamp_min(1e-12).log(), v1.clamp_min(1e-12).log()
                    mod.running_var.copy_((lv0 + (lv1 - lv0) * _sign(v0.shape, gen, dev)).exp())
            rec("BN", f"sign:{k}", X)
        del Fre, X
    return rows
