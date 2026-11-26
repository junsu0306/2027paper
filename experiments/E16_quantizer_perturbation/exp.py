"""E16: change one quantizer detail at a time (Research_Plan 실험 1 변형 1 and 실험 8-2), batch 1.

One job = model x recipe (QCB: own recipe(s); non-QCB: all 7).
Weight side, weights-only state (activations FP32, BN stored; the replica = E07's Qw):
  scale:d     step s -> s (1 + d): (1 + d) * quant(W / (1 + d)) with the recipe's own rounding and code range
  shift:c     grid moved by c s: quant(W - c s) + c s
  round:r     same grid, rounding rule r in half_even | half_away | half_up (differs only at exact ties)
  foldflip    the recipe's weight rule applied in the other BN space (folded recipes -> unfolded F weights; unfolded
              recipes -> BN-folded weights), evaluated in that space
Activation side, full deployment of quant(W) (deploysim build, calibrated once):
  clip:d      every calibrated activation range scaled by (1 + d) (quantizer re-derived; quanti_dyn unchanged)
  innorm      inputs normalised with mean = std = 0.5 instead of the model's own statistics
Measures: accuracy, ASR, top-1 agreement on the probe with F and with the unperturbed state, ProP sigma 100 skew / class.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, components as C, data, deploy, grids, measures as M  # noqa: E402
from qcbobs.models import ALL_RECIPES, load, trigger  # noqa: E402

SCALES = (-0.05, -0.02, -0.01, -0.005, -0.001, 0.001, 0.005, 0.01, 0.02, 0.05)
SHIFTS = (1 / 16, 1 / 8, 1 / 4, 1 / 2)
CLIPS = (-0.25, -0.10, -0.05, -0.01, 0.01, 0.05, 0.10, 0.25)
FOLDED = {"fbgemm8", "qnnpack8"}


def models():
    d = cohort.load_all()
    return d[d.batch == "batch1"]


def jobs():
    out = []
    for _, r in models().iterrows():
        recs = r.recipes.split(",") if r.group == "QCB" else list(ALL_RECIPES)
        for rec in recs:
            out.append({"id": f"Q::{r.model_key}::{rec}", "priority": 0 if r.group == "QCB" else 1})
    return out


def _round(v, rule):
    if rule == "half_even":
        return torch.round(v)
    if rule == "half_away":
        return torch.sign(v) * torch.floor(v.abs() + 0.5)
    return torch.floor(v + 0.5)


def run(job, dev):
    _, key, recipe = job["id"].split("::")
    r = cohort.row(key)
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load(r.dataset)
    probe, test = norm(imgs.probe.to(dev)), norm(imgs.test.to(dev))
    qcb = r.group == "QCB"
    trig01 = trigger(imgs.test.to(dev), r.trigger if qcb else "Patch6x6")
    trig = norm(trig01)
    target = int(r.target) if qcb else None
    f_pred = M.logits(F, probe).argmax(1)
    rep, layers, _ = grids.replica(F, recipe, dev)
    gs = grids.grids(layers, recipe)
    W = {n: m.weight.detach().clone() for n, m in layers}
    Q0 = {lg.name: lg.quant(W[lg.name]) for lg in gs}
    head = {"model_key": key, "group": r.group, "cell": r.cell, "family": r.family, "recipe": recipe, "target": target, "K": K}
    rows = []
    ref = {}

    def meas(model, side, cond, val, xs=(test, trig, probe)):
        t, tr, pb = xs
        aa = M.acc_asr(model, t, imgs.test_y, tr, target, K)
        p = M.logits(model, pb).argmax(1)
        if (side, "base") not in ref:
            ref[(side, "base")] = p
        pr = M.prop(model, pb, K)
        rows.append(head | {"side": side, "cond": cond, "val": val, "acc": aa["acc"], "asr": aa.get("asr"),
                            "agree_F": float((p == f_pred).float().mean()), "agree_base": float((p == ref[(side, "base")]).float().mean()),
                            "skew_s100": M.skew(pr["s100"]), "cls_s100": int(np.argmax(pr["s100"]))})

    # ---- weight side (weights only)
    deploy.set_weights(layers, Q0)
    meas(rep, "W", "base", 0.0)
    for d in SCALES:
        deploy.set_weights(layers, {lg.name: (1 + d) * lg.quant(W[lg.name] / (1 + d)) for lg in gs})
        meas(rep, "W", "scale", d)
    for c in SHIFTS:
        deploy.set_weights(layers, {lg.name: lg.quant(W[lg.name] - c * lg.scale) + c * lg.scale for lg in gs})
        meas(rep, "W", "shift", c)
    ties = 0
    for rule in ("half_even", "half_away", "half_up"):
        wq = {}
        for lg in gs:
            u = lg.code_coord(W[lg.name])
            if rule == "half_even":
                ties += int(((u - torch.floor(u)) - 0.5).abs().lt(1e-6).sum())
            wq[lg.name] = (torch.clamp(_round(u, rule), lg.qmin, lg.qmax) - lg.zp) * lg.scale
        deploy.set_weights(layers, wq)
        meas(rep, "W", "round", {"half_even": 0, "half_away": 1, "half_up": 2}[rule])
    deploy.set_weights(layers, W)
    # BN fold flip
    if recipe in FOLDED:
        alt = copy.deepcopy(F).eval()
        alt_layers = [(n, m) for n, m in alt.named_modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    else:
        alt, alt_layers, _ = grids.replica(F, "fbgemm8", dev)
    ga = grids.grids(alt_layers, recipe)
    with torch.no_grad():
        for (n, m), lg in zip(alt_layers, ga):
            m.weight.copy_(lg.quant(m.weight.detach()))
    meas(alt, "W", "foldflip", 1.0)
    del alt
    # ---- activation side (full deployment of Q0)
    deploy.set_weights(layers, Q0)
    dm = deploy.build(rep, recipe, imgs, norm, dev)
    acts = C.act_quantizers(dm)
    meas(dm, "A", "base", 0.0)
    saved = [(a.lo, a.hi, getattr(a, "L", None), getattr(a, "s", None), getattr(a, "z", None)) for _, a in acts]
    for d in CLIPS:
        for (_, a), (lo, hi, L, s, z) in zip(acts, saved):
            if a.mode == "quanti_dyn":
                continue
            a.lo, a.hi = lo * (1 + d), hi * (1 + d)
            if L is not None:
                a.L = L * (1 + d)
            if a.mode == "affine":
                a.freeze()
        meas(dm, "A", "clip", d)
    for (_, a), (lo, hi, L, s, z) in zip(acts, saved):
        a.lo, a.hi = lo, hi
        if L is not None:
            a.L = L
        if a.mode == "affine":
            a.s, a.z = s, z
    alt_norm = lambda x01: (x01 - 0.5) / 0.5  # noqa: E731
    meas(dm, "A", "innorm", 1.0, xs=(alt_norm(imgs.test.to(dev)), alt_norm(trig01), alt_norm(imgs.probe.to(dev))))
    rows.append(head | {"side": "info", "cond": "ties", "val": float(ties)})
    del dm
    return rows
