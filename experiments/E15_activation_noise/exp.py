"""E15: activation-side noise (Research_Plan 실험 1, 변형 2 and the activation analogue of E01's weight noise), batch 1.

Weights are F throughout (the defender's state). One job = model x recipe (QCB: own recipe(s); non-QCB: all 7); the
recipe gives the activation quantizer placement and calibrated ranges (deploysim build; iao recipes with BN stored).
Conditions:
  base      activation quantizers bypassed (= F)
  clip1     clip at the calibrated range (E06 alpha = 1)
  cj:e:s    clip-range jitter: each quantizer clips at [lo * u1, hi * u2], u ~ U(1 - e, 1 + e), e in EPS, seeds 0..3
  an:k      additive Gaussian noise at every quantizer input, sd = k * (hi - lo) / 255, no clipping, k in K_STEPS
  bn:e:s    BN statistics jitter on F itself (recipe-independent; run once per model in its first job):
            mean += e * sd * N(0,1), var *= exp(e * N(0,1)), e in EPS, seeds 0..3
  reach     no clipping; per quantizer the share of inputs above hi (below lo) and max(input) / hi, for clean probe
            images with input noise sd in REACH_SIGMAS and for triggered test images (QCB trigger / Patch6x6)
Measures (each condition except reach): clean accuracy, ASR, top-1 agreement with base on the probe, destination
histogram of probe predictions that changed, ProP histogram (sigma 100 and 2) and its skew / class.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, components as C, data, grids, measures as M  # noqa: E402
from qcbobs.models import ALL_RECIPES, load, trigger  # noqa: E402

EPS = (0.05, 0.1, 0.2, 0.4)
SEEDS = 4
K_STEPS = (1, 2, 4, 8, 16)
REACH_SIGMAS = (0.0, 0.05, 0.1, 0.2)


def models():
    d = cohort.load_all()
    return d[d.batch == "batch1"]


def jobs():
    out = []
    for _, r in models().iterrows():
        recs = r.recipes.split(",") if r.group == "QCB" else list(ALL_RECIPES)
        for i, rec in enumerate(recs):
            out.append({"id": f"A::{r.model_key}::{rec}::{int(i == 0)}", "priority": 0 if r.group == "QCB" else 1})
    return out


def run(job, dev):
    _, key, recipe, first = job["id"].split("::")
    r = cohort.row(key)
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load(r.dataset)
    probe = norm(imgs.probe.to(dev))
    test = norm(imgs.test.to(dev))
    qcb = r.group == "QCB"
    trig = norm(trigger(imgs.test.to(dev), r.trigger if qcb else "Patch6x6"))
    target = int(r.target) if qcb else None
    rep, layers, _ = grids.replica(F, recipe, dev)          # FP32 weights in the recipe's space
    dm = C.build(rep, recipe, imgs, norm, dev, bn="stored" if recipe.startswith("iao") else "recipe")
    acts = C.act_quantizers(dm)
    head = {"model_key": key, "group": r.group, "cell": r.cell, "family": r.family, "recipe": recipe, "target": target, "K": K}
    rows = []
    C.set_acts(acts, "off")
    base = M.logits(dm, probe).argmax(1)

    def meas(model, cond, **kw):
        aa = M.acc_asr(model, test, imgs.test_y, trig, target, K)
        p = M.logits(model, probe).argmax(1)
        ch = p != base
        pr = M.prop(model, probe, K)
        rows.append(head | {"kind": "cond", "cond": cond, **kw, "acc": aa["acc"], "asr": aa.get("asr"),
                            "agree": float((~ch).float().mean()), "dest_hist": M.hist(p[ch], K),
                            "prop_s100": pr["s100"], "prop_s2": pr["s2"], "skew_s100": M.skew(pr["s100"]),
                            "cls_s100": int(np.argmax(pr["s100"]))})

    meas(dm, "base")
    C.set_acts(acts, "clip:1")
    meas(dm, "clip1")
    rng = [C._range(a) if a.mode != "quanti_dyn" else None for _, a in acts]
    for e in EPS:
        for s in range(SEEDS):
            g = torch.Generator().manual_seed(1000 * s + int(e * 100))
            for (_, a), lh in zip(acts, rng):
                a.state = "quant"
                if lh is None:
                    a.q = C.make_q(a, "clip:1")
                    continue
                u = (1 + e * (torch.rand(2, generator=g) * 2 - 1)).tolist()
                lo, hi = lh[0] * u[0], lh[1] * u[1]
                a.q = (lambda x, lo=lo, hi=hi: torch.clamp(x, lo, hi))
            meas(dm, "cj", eps=e, seed=s)
    for k in K_STEPS:
        for (_, a), lh in zip(acts, rng):
            a.state = "quant"
            if lh is None:
                a.q = (lambda x: x)
                continue
            sd = k * (lh[1] - lh[0]) / 255.0
            a.q = (lambda x, sd=sd: x + torch.randn_like(x) * sd)
        torch.manual_seed(k)
        meas(dm, "an", k=k)
    # reach: record, do not clip
    stats = {}

    def recorder(i, lh):
        def f(x):
            st = stats.setdefault(i, [0, 0, 0, 0.0])
            st[0] += int((x > lh[1]).sum()); st[1] += int((x < lh[0]).sum()); st[2] += x.numel()
            st[3] = max(st[3], float(x.max()) / max(lh[1], 1e-12))
            return x
        return f
    for (_, a), lh in zip(acts, rng):
        a.state = "bypass" if lh is None else "quant"
    for src in [f"probe_s{s:g}" for s in REACH_SIGMAS] + ["trig"]:
        stats.clear()
        for i, ((_, a), lh) in enumerate(zip(acts, rng)):
            if lh is not None:
                a.q = recorder(i, lh)
        if src == "trig":
            x = trig
        else:
            sg = float(src.split("_s")[1])
            gen = torch.Generator(device=dev).manual_seed(7)
            x01 = imgs.probe.to(dev)
            x = norm((x01 + torch.randn(x01.shape, generator=gen, device=dev) * sg).clamp(0, 1)) if sg > 0 else probe
        M.logits(dm, x)
        per = [{"q": i, "name": acts[i][0], "above": v[0] / v[2], "below": v[1] / v[2], "max_ratio": v[3]} for i, v in sorted(stats.items())]
        rows.append(head | {"kind": "reach", "src": src, "per_quantizer": per,
                            "max_above": max(p["above"] for p in per), "max_ratio": max(p["max_ratio"] for p in per)})
    C.set_acts(acts, "off")
    del dm
    if first == "1":
        Fb = copy.deepcopy(F)
        bns = C.bn_layers(Fb)
        orig = C.bn_stats(Fb)
        for e in EPS:
            for s in range(SEEDS):
                g = torch.Generator(device=dev).manual_seed(2000 * s + int(e * 100))
                with torch.no_grad():
                    for n, mod in bns:
                        m0, v0 = orig[n]
                        mod.running_mean.copy_(m0 + e * v0.sqrt() * torch.randn(m0.shape, generator=g, device=dev))
                        mod.running_var.copy_(v0 * torch.exp(e * torch.randn(v0.shape, generator=g, device=dev)))
                meas(Fb, "bn", eps=e, seed=s)
        del Fb
    return rows
