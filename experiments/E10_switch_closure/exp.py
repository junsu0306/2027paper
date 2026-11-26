"""E10 (reduced, 2026-11-26): can a defender who knows only standard formats and has only its own clean images switch the
backdoor on, without the target recipe, while normals stay unchanged?

Models: batch2, batch_x, batch_xn (held out from E05-E07, E13). batch 1 is covered by E13 (29 deployments).
Defender data D_def: final_behavior_test images 2000-3999 (disjoint from test 0-999, quant_cal, detector_probe). Where a
recipe calibrates on training data (iao), D_def repeated 4 times is passed instead.
States (one job per model):
  F
  W:<g>       weights only (FP32 activations, BN stored) on grid g: the 7 recipe grids, and pt8nofold (per-tensor
              symmetric 8-bit, torch qnnpack observer, on the unfolded weights: the eager no-fold format of E13)
  C1, C2      F weights, activation clipping at alpha x the D_def-calibrated range, iao8 placement, BN stored
  Bfp         F weights, BN statistics re-estimated on D_def in FP32 (cumulative average)
  Bq          F weights, BN re-estimated by the iao8 procedure on D_def (activation quantization on), evaluated with FP32
              activations (= E05 Fbn with defender data)
  CB          Bq + C1
  D:<r>       full deployment of recipe r in {iao8, iao4, efrap4} with D_def as calibration / training data
Measures: clean accuracy and top-1 agreement with F on test 0-999; ASR (QCB; trigger on non-target test images).
"""
from __future__ import annotations

import copy
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, components as C, data, deploy, grids, measures as M  # noqa: E402
from qcbobs.env import DATA  # noqa: E402
from qcbobs.models import ALL_RECIPES, load, trigger  # noqa: E402

DEF_RANGE = (2000, 4000)
D_RECIPES = ("iao8", "iao4", "efrap4")
_DEF = {}


def defender_images(dataset):
    if dataset not in _DEF:
        base = data.load(dataset)
        name = data.DS_FILE[dataset]
        z = np.load(DATA / "datasets/main" / f"{name}.npz")
        roles = json.loads((DATA / "datasets/main/manifests" / name / "split_base.json").read_text())["datasets"][name]["roles"]
        idx = np.asarray(roles["final_behavior_test"]["indices"][DEF_RANGE[0]:DEF_RANGE[1]])
        u8 = z["x_test"][idx]
        _DEF[dataset] = dataclasses.replace(base, cal=data.to01(u8), train_u8=np.concatenate([u8] * 4))
    return _DEF[dataset]


def jobs():
    d = cohort.load_all()
    d = d[d.batch != "batch1"]
    return [{"id": f"N::{r.model_key}", "priority": 0 if r.group == "QCB" else 1} for _, r in d.iterrows()]


def run(job, dev):
    _, key = job["id"].split("::")
    r = cohort.row(key)
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load(r.dataset)
    idef = defender_images(r.dataset)
    test = norm(imgs.test.to(dev))
    qcb = r.group == "QCB"
    trig = norm(trigger(imgs.test.to(dev), r.trigger if qcb else "Patch6x6"))
    target = int(r.target) if qcb else None
    with torch.no_grad():
        f_pred = M.logits(F, test).argmax(1)
    head = {"model_key": key, "batch": r.batch, "group": r.group, "cell": r.cell, "family": r.family, "dataset": r.dataset,
            "arch": r.arch, "target_recipe": r.target_recipe, "target": target}
    rows = []

    def meas(state, model):
        aa = M.acc_asr(model, test, imgs.test_y, trig, target, K)
        p = M.logits(model, test).argmax(1)
        rows.append(head | {"state": state, "acc": aa["acc"], "asr": aa.get("asr"), "agree_F": float((p == f_pred).float().mean()),
                            "trig_top_share": max(aa["trig_hist"]) / max(sum(aa["trig_hist"]), 1)})

    meas("F", F)
    for g in ALL_RECIPES:
        rep, layers, _ = grids.replica(F, g, dev)
        gs = grids.grids(layers, g)
        deploy.set_weights(layers, {lg.name: lg.quant(dict(layers)[lg.name].weight.detach()) for lg in gs})
        meas(f"W:{g}", rep)
        if g in D_RECIPES:
            dm = deploy.build(rep, g, idef, norm, dev)
            meas(f"D:{g}", dm)
            del dm
        del rep
    Fn_ = copy.deepcopy(F).eval()
    fl = [(n, m) for n, m in Fn_.named_modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    with torch.no_grad():
        for n, m in fl:
            m.weight.copy_(grids._torch_grid(n, m.weight.detach(), False, 8).quant(m.weight.detach()))
    meas("W:pt8nofold", Fn_)
    del Fn_
    rep, _, _ = grids.replica(F, "iao8", dev)
    dm = C.build(rep, "iao8", idef, norm, dev, bn="stored")
    acts = C.act_quantizers(dm)
    for a_ in (1, 2):
        C.set_acts(acts, f"clip:{a_}")
        meas(f"C{a_}", dm)
    del dm
    Fb = copy.deepcopy(F)
    for _, mod in C.bn_layers(Fb):
        mod.reset_running_stats()
        mod.momentum = None
    Fb.train()
    with torch.no_grad():
        x = norm(idef.cal.to(dev))
        for s in range(0, len(x), 128):
            Fb(x[s:s + 128])
    Fb.eval()
    meas("Bfp", Fb)
    del Fb
    dm = C.build(rep, "iao8", idef, norm, dev, bn="recipe")
    acts = C.act_quantizers(dm)
    C.set_acts(acts, "off")
    meas("Bq", dm)
    C.set_acts(acts, "clip:1")
    meas("CB", dm)
    return rows
