"""E05x: E05 (component factorial) + E06 (clip curve, localization) on QCB whose attack-time simulator D-hat or dataset
differs from batch 1 (cohorts/batch_x.csv). Same code paths as E05 / E06, made dataset- and architecture-generic.
ASR and accuracy only (mechanism); no trigger-free statistics, no normals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import components as C, data, deploy, grids, measures as M  # noqa: E402
from qcbobs.models import RECIPES, load, trigger  # noqa: E402

COHORT = ROOT / "cohorts/batch_x.csv"
A_VARIANTS = ("off", "full", "clip", "round")
ALPHAS = (8.0, 4.0, 2.0, 1.5, 1.25, 1.0, 0.8, 0.6)
VGG16_BLOCKS = (2, 2, 3, 3, 3)


def cohort():
    return pd.read_csv(COHORT)


def jobs():
    return [{"id": f"X::{r.model_key}::{rec}", "priority": 2 if rec.startswith("iao") else 0}
            for _, r in cohort().iterrows() for rec in r.recipes.split(",")]


def group_map(model) -> dict:
    """module name -> stage. ResNet: layerK / blocks.N (2 per stage); VGG16-BN: conv blocks by conv order; head."""
    out, convs, block = {}, 0, 1
    bounds = [sum(VGG16_BLOCKS[:i + 1]) for i in range(len(VGG16_BLOCKS))]
    for name, mod in model.named_modules():
        parts = name.split(".")
        if not name:
            continue
        if parts[0].startswith("layer"):
            out[name] = parts[0]
        elif parts[0] == "blocks" and len(parts) > 1 and parts[1].isdigit():
            out[name] = f"layer{int(parts[1]) // 2 + 1}"
        elif parts[0] == "features":
            if isinstance(mod, nn.Conv2d):
                block = next(i + 1 for i, b in enumerate(bounds) if convs < b)
                convs += 1
            out[name] = f"vgg{block}"
        elif parts[0] in ("conv1", "bn1"):
            out[name] = "stem"
        else:
            out[name] = "head"
    return out


def run(job, dev):
    _, key, recipe = job["id"].split("::")
    r = cohort().set_index("model_key").loc[key]
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load(r.dataset)
    test = norm(imgs.test.to(dev))
    trig = norm(trigger(imgs.test.to(dev), r.trigger))
    target = int(r.target)
    iao = recipe.startswith("iao")
    rep, layers, _ = grids.replica(F, recipe, dev)
    gs = grids.grids(layers, recipe)
    names = [n for n, _ in layers]
    W = {n: m.weight.detach().clone() for n, m in layers}
    Q0 = {lg.name: lg.quant(W[lg.name]) for lg in gs}
    gm = group_map(rep)
    order = []
    for n in names:
        if gm[n] not in order:
            order.append(gm[n])
    head = {"model_key": key, "cell": r.cell, "dataset": r.dataset, "arch": r.arch, "family": r.family,
            "target": target, "K": K, "recipe": recipe, "recipe_bits": RECIPES[recipe]["bits"], "parent_key": r.parent_key}
    rows = []

    def ev(model):
        aa = M.acc_asr(model, test, imgs.test_y, trig, target, K)
        return {"acc": aa["acc"], "asr": aa.get("asr")}

    # ---- factorial (E05) and clip curve (E06 a) ----
    builds = {}
    for wname, wts in (("F", W), ("Q", Q0)):
        deploy.set_weights(layers, wts)
        for bn in (("reest", "stored") if iao else ("recipe",)):
            dm = C.build(rep, recipe, imgs, norm, dev, bn="stored" if bn == "stored" else "recipe")
            acts = C.act_quantizers(dm)
            for av in A_VARIANTS:
                C.set_acts(acts, av)
                rows.append(head | {"part": "factorial", "W": wname, "A": av, "BN": bn} | ev(dm))
            if bn != "stored":
                for a in ALPHAS:
                    C.set_acts(acts, f"clip:{a:g}")
                    rows.append(head | {"part": "clip_curve", "W": wname, "alpha": a, "BN": bn} | ev(dm))
                builds[wname] = dm
            else:
                builds[wname + "_stored"] = dm
    deploy.set_weights(layers, W)

    # ---- localization (E06 b) ----
    units = [[n] for n in names] + [[n for n in names if gm[n] == g] for g in order]
    labels = names + [f"group:{g}" for g in order]
    for lab, u in zip(labels, units):
        for mode in ("suff", "nec"):
            deploy.set_weights(layers, {n: (Q0[n] if (n in u) == (mode == "suff") else W[n]) for n in names})
            rows.append(head | {"part": "loc", "component": "W", "W": "F/Q", "unit": lab, "mode": mode} | ev(rep))
    deploy.set_weights(layers, W)
    for wname in ("F", "Q"):
        dm = builds[wname]
        acts = C.act_quantizers(dm)
        anames = [n for n, _ in acts]
        ag = [gm.get(n, "head") for n in anames]
        aorder = [g for g in order if g in ag] + [g for g in dict.fromkeys(ag) if g not in order]
        aunits = [[i] for i in range(len(acts))] + [[i for i, g in enumerate(ag) if g == gg] for gg in aorder]
        alabels = anames + [f"group:{g}" for g in aorder]
        for lab, u in zip(alabels, aunits):
            for mode in ("suff", "nec"):
                only = set(u) if mode == "suff" else set(range(len(acts))) - set(u)
                C.set_acts(acts, "clip", only=only)
                rows.append(head | {"part": "loc", "component": "A_clip", "W": wname, "unit": lab, "mode": mode} | ev(dm))
        if iao:
            st_reest = C.bn_stats(dm)
            st_stored = C.bn_stats(builds[wname + "_stored"])
            C.set_acts(acts, "off")
            bnames = [n for n, _ in C.bn_layers(dm)]
            bg = [gm.get(n, "head") for n in bnames]
            border = [g for g in order if g in bg]
            bunits = [[i] for i in range(len(bnames))] + [[i for i, g in enumerate(bg) if g == gg] for gg in border]
            blabels = bnames + [f"group:{g}" for g in border]
            for lab, u in zip(blabels, bunits):
                for mode in ("suff", "nec"):
                    only = set(u) if mode == "suff" else set(range(len(bnames))) - set(u)
                    C.set_bn_mix(dm, st_stored, st_reest, 1.0, only=only)
                    rows.append(head | {"part": "loc", "component": "BN", "W": wname, "unit": lab, "mode": mode} | ev(dm))
            C.set_bn_mix(dm, st_stored, st_reest, 1.0)
    return rows
