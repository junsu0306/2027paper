"""E05: which deployment component turns the backdoor on, and which one carries the ProP skew? (factorial, batch 1)

Factors (one job = one model x recipe; QCB on its own recipe(s), normals and controls on all 7):
  W   weights          F (FP32 weights, recipe weight space)  |  Q (nearest rounding on the recipe grid)
  A   activations      off (FP32)  |  full (deploysim quantizer)  |  clip (same calibrated range, no rounding)
                       |  round (same step, no range limit)
  BN  (iao recipes)    stored (F's running stats; registered ablation iao*_train_bneval)  |  reest (re-estimated on
                       50k training images under activation quantization, the registered recipe)
The activation quantizers are calibrated once per (W, BN) by deploysim.build exactly as the recipe does; the A variants
only swap how the calibrated quantizer acts. W=Q, A=full, BN=recipe reproduces the deployed model of E01.
Measures per cell: test accuracy, ASR (QCB trigger / Patch6x6), clean probe histogram, ProP (M3) sigma 100 and 2.
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import data, deploy, grids, measures as M  # noqa: E402
from qcbobs.models import RECIPES, load, trigger  # noqa: E402

COHORT = ROOT / "cohorts/batch1.csv"
A_VARIANTS = ("off", "full", "clip", "round")


def cohort():
    return pd.read_csv(COHORT)


def jobs():
    out = []
    for _, r in cohort().iterrows():
        for rec in r.recipes.split(","):
            out.append({"id": f"C::{r.model_key}::{rec}",
                        "priority": (0 if r.group == "QCB" else 1) + (2 if rec.startswith("iao") else 0)})
    return out


def _round_away(v):
    return torch.sign(v) * torch.floor(v.abs() + 0.5)


def _variant_q(a, variant):
    """Return a replacement for ActQ.q: 'clip' keeps the calibrated range without rounding, 'round' keeps the step
    without the range limit. Mirrors deploysim.fq_affine / fq_quanti / fq_iao."""
    mode, bits = a.mode, a.bits
    if mode == "affine":
        s, z, qmax = a.s, a.z, (1 << bits) - 1
        if variant == "clip":
            return lambda x: torch.clamp(x, (0 - z) * s, (qmax - z) * s)
        return lambda x: torch.round(x / s) * s
    if mode in ("quanti", "quanti_dyn"):
        qmax = float((1 << bits) - 1)

        def rng(x):
            return (float(x.detach().min()), float(x.detach().max())) if mode == "quanti_dyn" else (a.lo, a.hi)

        def f(x):
            lo, hi = rng(x)
            scale = qmax / (hi - lo + 1e-14)
            zp = float(torch.round(torch.tensor(lo * scale)))
            v = x * scale - zp
            v = torch.clamp(v, 0.0, qmax) if variant == "clip" else torch.floor(v + 0.5)
            return (v + zp) / scale
        return f
    if mode == "iao":
        qmin, qmax = -float(1 << (bits - 1)), float((1 << (bits - 1)) - 1)

        def g(x):
            L = a.L if getattr(a, "L", None) is not None else max(abs(a.lo), abs(a.hi))
            s = max(L / ((qmax - qmin) / 2), float(torch.finfo(torch.float32).eps))
            return torch.clamp(x / s, qmin, qmax) * s if variant == "clip" else _round_away(x / s) * s
        return g
    raise ValueError(mode)


def act_quantizers(dm):
    DS = deploy._ds()
    return [h for mod in dm.modules() for h in mod._forward_pre_hooks.values() if isinstance(h, DS.ActQ)]


def set_variant(acts, variant):
    for a in acts:
        if not hasattr(a, "_orig_q"):
            a._orig_q = a.q
        if variant == "off":
            a.state = "bypass"
            continue
        a.state = "quant"
        a.q = a._orig_q if variant == "full" else _variant_q(a, variant)


def build(rep, recipe, imgs, norm, dev, bn):
    """deploysim.build with spec['w'] = None (weights already in rep); bn='stored' uses the registered *_bneval ablation."""
    DS = deploy._ds()
    spec = dict(DS.SPECS[DS.VIEW_RECIPE[RECIPES[recipe]["view_key"]]])
    spec["w"] = None
    if bn == "stored":
        spec["bn"] = "eval"
    cal_x = norm(imgs.cal.to(dev))
    return DS.build(rep, spec, cal_x, deploy.train_arg(recipe, imgs, norm), device=str(dev))


def run(job, dev):
    _, key, recipe = job["id"].split("::")
    r = cohort().set_index("model_key").loc[key]
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load("CIFAR-10")
    probe = norm(imgs.probe.to(dev))
    test = norm(imgs.test.to(dev))
    trig_name = r.trigger if r.group == "QCB" else "Patch6x6"
    trig = norm(trigger(imgs.test.to(dev), trig_name))
    target = None if r.group != "QCB" else int(r.target)
    rep, layers, _ = grids.replica(F, recipe, dev)
    gs = grids.grids(layers, recipe)
    W = {n: m.weight.detach().clone() for n, m in layers}
    Q0 = {lg.name: lg.quant(W[lg.name]) for lg in gs}
    head = {"model_key": key, "group": r.group, "cell": r.cell, "family": r.family, "target": target, "K": K,
            "recipe": recipe, "recipe_bits": RECIPES[recipe]["bits"], "parent_key": r.parent_key, "trigger_used": trig_name}
    bns = ("reest", "stored") if recipe.startswith("iao") else ("recipe",)
    rows = []
    for wname, wts in (("F", W), ("Q", Q0)):
        deploy.set_weights(layers, wts)
        for bn in bns:
            t0 = time.time()
            dm = build(rep, recipe, imgs, norm, dev, "stored" if bn == "stored" else "recipe")
            acts = act_quantizers(dm)
            tb = time.time() - t0
            for av in A_VARIANTS:
                t0 = time.time()
                set_variant(acts, av)
                aa = M.acc_asr(dm, test, imgs.test_y, trig, target, K)
                pr = M.prop(dm, probe, K)
                rows.append(head | {"W": wname, "A": av, "BN": bn, "n_act_quantizers": len(acts),
                                    "acc": aa["acc"], "asr": aa.get("asr"), "trig_hist": aa.get("trig_hist"),
                                    "probe_hist": M.hist(M.logits(dm, probe).argmax(1), K),
                                    "prop_s100": pr["s100"], "prop_s2": pr["s2"],
                                    "skew_s100": M.skew(pr["s100"]), "cls_s100": int(np.argmax(pr["s100"])),
                                    "sec_build": tb, "sec": time.time() - t0})
            del dm
    deploy.set_weights(layers, W)
    return rows
