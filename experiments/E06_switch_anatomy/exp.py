"""E06 a + b: switch curves (dose-response) and switch localization, batch 1. See plan.md and prediction.md.

a (every model x recipe): sweep the strength of one deployment component at a time, weights F or Q:
   activation clip only, range scaled by alpha (ALPHAS; 'off' = no clip);
   activation round only, step from the calibrated range at BITS bits (no range limit);
   iao recipes: BN running stats mixed between stored (F) and re-estimated: (1 - lam) F + lam R, activations off / full.
   Measures: accuracy, ASR (QCB trigger / Patch6x6), ProP sigma 100 skew and class.
b (QCB only): apply one component to a single unit (sufficiency) or to all units but one (necessity), and to layer
   groups (stem, layer1..4, head):
   weights Q (activations FP32, BN stored);  activation clip (weights F and weights Q);  iao: BN re-estimated stats.
   Measures: accuracy, ASR.
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
from qcbobs import components as C, data, deploy, grids, measures as M  # noqa: E402
from qcbobs.models import RECIPES, load, trigger  # noqa: E402

COHORT = ROOT / "cohorts/batch1.csv"
ALPHAS = (8.0, 4.0, 2.0, 1.5, 1.25, 1.0, 0.8, 0.6)
BITS = (16, 12, 10, 8, 6, 4)
LAMS = (0.0, 0.25, 0.5, 0.75, 1.0)


def cohort():
    return pd.read_csv(COHORT)


def jobs():
    out = []
    for _, r in cohort().iterrows():
        for rec in r.recipes.split(","):
            out.append({"id": f"S::{r.model_key}::{rec}",
                        "priority": (0 if r.group == "QCB" else 1) + (2 if rec.startswith("iao") else 0)})
    return out


def group_of(name: str) -> str:
    """ResNet18 stage of a module name: 'layerK.*' or 'blocks.N.*' (BasicBlocks 2 per stage), stem, head."""
    parts = name.split(".")
    if parts[0].startswith("layer"):
        return parts[0]
    if parts[0] == "blocks" and len(parts) > 1 and parts[1].isdigit():
        return f"layer{int(parts[1]) // 2 + 1}"
    return "stem" if parts[0] in ("conv1", "bn1") else "head"


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
    qcb = r.group == "QCB"
    iao = recipe.startswith("iao")
    rep, layers, _ = grids.replica(F, recipe, dev)
    gs = grids.grids(layers, recipe)
    names = [n for n, _ in layers]
    W = {n: m.weight.detach().clone() for n, m in layers}
    Q0 = {lg.name: lg.quant(W[lg.name]) for lg in gs}
    head = {"model_key": key, "group": r.group, "cell": r.cell, "family": r.family, "target": target, "K": K,
            "recipe": recipe, "recipe_bits": RECIPES[recipe]["bits"], "parent_key": r.parent_key}
    rows = []

    def ev(model, prop=True):
        aa = M.acc_asr(model, test, imgs.test_y, trig, target, K)
        out = {"acc": aa["acc"], "asr": aa.get("asr")}
        if prop:
            h = M.prop(model, probe, K)["s100"]
            out |= {"skew_s100": M.skew(h), "cls_s100": int(np.argmax(h))}
        return out

    # ---------------- a: dose-response ----------------
    dms = {}
    for wname, wts in (("F", W), ("Q", Q0)):
        deploy.set_weights(layers, wts)
        dm = C.build(rep, recipe, imgs, norm, dev)
        acts = C.act_quantizers(dm)
        t0 = time.time()
        for v in ["off"] + [f"clip:{a:g}" for a in ALPHAS] + [f"round:{b}" for b in BITS]:
            C.set_acts(acts, v)
            rows.append(head | {"stage": "a", "W": wname, "knob": v.split(":")[0], "level": v.partition(":")[2] or None,
                                "variant": v} | ev(dm) | {"sec": time.time() - t0})
            t0 = time.time()
        if iao:
            st_reest = C.bn_stats(dm)
            dm_st = C.build(rep, recipe, imgs, norm, dev, bn="stored")
            st_stored = C.bn_stats(dm_st)
            del dm_st
            for av in ("off", "full"):
                C.set_acts(acts, av)
                for lam in LAMS:
                    C.set_bn_mix(dm, st_stored, st_reest, lam)
                    rows.append(head | {"stage": "a", "W": wname, "knob": f"bn_A{av}", "level": str(lam),
                                        "variant": f"bn:{lam}|A:{av}"} | ev(dm) | {"sec": time.time() - t0})
                    t0 = time.time()
            C.set_bn_mix(dm, st_stored, st_reest, 1.0)
            dms[wname] = (dm, acts, st_stored, st_reest)
        else:
            dms[wname] = (dm, acts, None, None)
    deploy.set_weights(layers, W)
    if not qcb:
        return rows

    # ---------------- b: localization (QCB only) ----------------
    groups = sorted(set(group_of(n) for n in names), key=lambda g: ["stem", "layer1", "layer2", "layer3", "layer4", "head"].index(g))
    # weights: rep with activations FP32 (BN stored; BN-folded replica for fbgemm / qnnpack)
    units = [[n] for n in names] + [[n for n in names if group_of(n) == g] for g in groups]
    labels = names + [f"group:{g}" for g in groups]
    for lab, u in zip(labels, units):
        for mode in ("suff", "nec"):
            ws = {n: (Q0[n] if (n in u) == (mode == "suff") else W[n]) for n in names}
            deploy.set_weights(layers, ws)
            rows.append(head | {"stage": "b", "component": "W", "W": "F/Q", "unit": lab, "mode": mode} | ev(rep, prop=False))
    deploy.set_weights(layers, W)
    # activation clip, weights F and weights Q
    for wname in ("F", "Q"):
        dm, acts, st_stored, st_reest = dms[wname]
        anames = [n for n, _ in acts]
        aunits = [[i] for i in range(len(acts))] + [[i for i, n in enumerate(anames) if group_of(n) == g] for g in groups]
        alabels = anames + [f"group:{g}" for g in groups]
        for lab, u in zip(alabels, aunits):
            if not u:
                continue
            for mode in ("suff", "nec"):
                only = set(u) if mode == "suff" else set(range(len(acts))) - set(u)
                C.set_acts(acts, "clip", only=only)
                rows.append(head | {"stage": "b", "component": "A_clip", "W": wname, "unit": lab, "mode": mode}
                            | ev(dm, prop=False))
        if iao:
            # BN: re-estimated stats in one BN (or a group) only, activations off
            C.set_acts(acts, "off")
            bnames = [n for n, _ in C.bn_layers(dm)]
            bunits = [[i] for i in range(len(bnames))] + [[i for i, n in enumerate(bnames) if group_of(n) == g] for g in groups]
            blabels = bnames + [f"group:{g}" for g in groups]
            for lab, u in zip(blabels, bunits):
                if not u:
                    continue
                for mode in ("suff", "nec"):
                    only = set(u) if mode == "suff" else set(range(len(bnames))) - set(u)
                    C.set_bn_mix(dm, st_stored, st_reest, 1.0, only=only)
                    rows.append(head | {"stage": "b", "component": "BN", "W": wname, "unit": lab, "mode": mode}
                                | ev(dm, prop=False))
            C.set_bn_mix(dm, st_stored, st_reest, 1.0)
    return rows
