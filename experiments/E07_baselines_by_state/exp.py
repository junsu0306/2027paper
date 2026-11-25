"""E07: existing backdoor detectors (MVD implementations, unchanged) on the same models in three kinds of state.

Hypothesis J (docs/WHY_HETEROGENEITY.md): QCB = core (an ordinary backdoor, present when switched on) + switch + concealment.
States (built with deploysim exactly as E05):
  F      FP32 model
  Qw     weights rounded on the recipe grid, activations FP32 (BN stored)          -- weight switch alone
  Fclip  FP32 weights, activation clipping only (calibrated range, no rounding)  -- activation switch alone
  Fbn    FP32 weights, BN stats re-estimated as the iao recipe, activations off   -- BN switch alone (iao only)
  Q      the deployed model
Detectors: Neural Cleanse (baselines_nc.run_nc), MM-BD (baselines.run_mmbd), SPD (baselines.run_spd, classifier head).
Models: QCB of batch 1 on their own recipe(s): F, Qw, Fclip, (Fbn), Q.
        Normals and controls (the 14 of E03B2: 2 per normal cell + 6 controls): F once; per recipe Q and Qw, and on the
        iao recipes also Fclip and Fbn.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import components as C, data, deploy, env, grids  # noqa: E402
from qcbobs.models import RECIPES, load, trigger  # noqa: E402
from qcbobs import measures as M  # noqa: E402

COHORT = ROOT / "cohorts/batch1.csv"
NORMALS_PER_CELL = 2
DETECTORS = ("nc", "mmbd", "spd")


def cohort():
    return pd.read_csv(COHORT)


def members():
    b = cohort()
    keep = []
    for _, g in b.groupby("cell", sort=False):
        keep += list(g.model_key[:NORMALS_PER_CELL]) if g.group.iloc[0].startswith("normal") else list(g.model_key)
    return b[b.model_key.isin(keep)]


def jobs():
    out = []
    for _, r in members().iterrows():
        q = r.group == "QCB"
        if not q:
            out.append({"id": f"K::{r.model_key}::F", "priority": 1})
        for rec in r.recipes.split(","):
            out.append({"id": f"K::{r.model_key}::{rec}", "priority": (0 if q else 1) + (2 if rec.startswith("iao") else 0)})
    return out


class Pix(nn.Module):
    """[0,1] pixels -> logits; `.head` = last Linear (SPD)."""

    def __init__(self, model, norm):
        super().__init__()
        self.model, self.norm = model, norm

    def forward(self, x):
        return self.model(self.norm(x))

    @property
    def head(self):
        return [m for m in self.model.modules() if isinstance(m, nn.Linear)][-1]


class Ctx:
    def __init__(self, pix, K, pool_u8, dev):
        self.pixel_model, self.row, self.device, self._pool = pix, {"num_classes": K}, dev, pool_u8

    def labeled_baseline_pool(self):
        return self._pool, np.zeros(len(self._pool), dtype=np.int64)


class _Deadline:
    def check(self):
        return None


def detect(model, norm, K, pool_u8, dev):
    env.setup()
    from multiview_detection import baselines as B, baselines_nc as BNC
    for p in model.parameters():
        p.requires_grad_(False)
    ctx = Ctx(Pix(model, norm).to(dev).eval(), K, pool_u8, dev)
    out = {}
    for name in DETECTORS:
        t0 = time.time()
        torch.manual_seed(0)
        if name == "nc":
            r = BNC.run_nc(ctx, _Deadline(), lambda: None)
            keep = {k: r.get(k) for k in ("status", "target", "anomaly_index", "native_alarm", "median_l1", "mad",
                                          "flagged_labels", "label_anomaly_indices")}
            keep["l1"] = [x.get("l1_png_uint8", x.get("l1")) for x in r.get("per_label", [])] if r.get("per_label") else None
        elif name == "mmbd":
            r = B.run_mmbd(ctx, _Deadline(), lambda: None)
            keep = {k: r.get(k) for k in ("status", "target", "raw_score", "native_alarm", "native_p_value", "margins")}
        else:
            r = B.run_spd(ctx, _Deadline())
            keep = {k: v for k, v in r.items() if k not in ("config", "final_losses", "converged")}
        out[name] = keep | {"sec": time.time() - t0}
    return out


def pool_u8(imgs):
    """quant_cal + detector_probe as NHWC uint8 (the MVD labeled_baseline_pool)."""
    x = torch.cat([imgs.cal, imgs.probe]).mul(255).round().clamp(0, 255).byte()
    return x.permute(0, 2, 3, 1).contiguous().numpy()


def run(job, dev):
    _, key, recipe = job["id"].split("::")
    r = cohort().set_index("model_key").loc[key]
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load("CIFAR-10")
    pool = pool_u8(imgs)
    test = norm(imgs.test.to(dev))
    trig_name = r.trigger if r.group == "QCB" else "Patch6x6"
    trig = norm(trigger(imgs.test.to(dev), trig_name))
    target = None if r.group != "QCB" else int(r.target)
    head = {"model_key": key, "group": r.group, "cell": r.cell, "family": r.family, "target": target, "K": K,
            "recipe": None if recipe == "F" else recipe, "parent_key": r.parent_key}
    rows = []

    def add(state, model):
        aa = M.acc_asr(model, test, imgs.test_y, trig, target, K)
        t0 = time.time()
        rows.append(head | {"state": state, "acc": aa["acc"], "asr": aa.get("asr")} | detect(model, norm, K, pool, dev)
                    | {"sec": time.time() - t0})

    qcb = r.group == "QCB"
    if recipe == "F" or (qcb and recipe == r.recipes.split(",")[0]):
        add("F", F)
    if recipe == "F":
        return rows
    iao = recipe.startswith("iao")
    rep, layers, _ = grids.replica(F, recipe, dev)
    gs = grids.grids(layers, recipe)
    W = {n: m.weight.detach().clone() for n, m in layers}
    Q0 = {lg.name: lg.quant(W[lg.name]) for lg in gs}
    # Qw: rounded weights, FP32 activations, stored BN (the replica itself)
    deploy.set_weights(layers, Q0)
    add("Qw", rep)
    # Q: deployed
    dm = C.build(rep, recipe, imgs, norm, dev)
    add("Q", dm)
    del dm
    deploy.set_weights(layers, W)
    if qcb or iao:
        dm = C.build(rep, recipe, imgs, norm, dev)           # FP32 weights, recipe calibration (BN re-estimated for iao)
        acts = C.act_quantizers(dm)
        if iao:
            C.set_acts(acts, "off")
            add("Fbn", dm)                                      # BN re-estimated, activations FP32
            st_reest = C.bn_stats(dm)
            dst = C.build(rep, recipe, imgs, norm, dev, bn="stored")
            C.set_bn_mix(dm, C.bn_stats(dst), st_reest, 0.0)  # back to stored BN for the clip-only state
            del dst
        C.set_acts(acts, "clip")
        add("Fclip", dm)
        del dm
    return rows
