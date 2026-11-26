"""E17: lowering the weight bits (Research_Plan 실험 1 stage 4 and 곁들임 '비트를 2까지 내린 정상 모델').

Models: batch 1 (all four groups) and batch-2 normals (light, standard, hard). One job per model.
Grids (BN-folded replicas, symmetric narrow codes +-(2^(b-1) - 1), scale = max|w| / qmax, round half to even):
  pc  per output channel (fbgemm8 replica)        pt  per tensor (qnnpack8 replica)
Bits b in BITS. Two states per (grid, b):
  W     weights only, activations FP32
  Wc    the same weights + activation clipping at the 8-bit calibrated range of that replica's deploysim recipe
        (fbgemm8 / qnnpack8 placement; clip only, no activation rounding)
Measures: accuracy, ASR (QCB trigger / Patch6x6), top-1 agreement with F on the probe, destination histogram of the
probe predictions that changed, ProP sigma 100 histogram, skew and class.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, components as C, data, deploy, grids, measures as M  # noqa: E402
from qcbobs.models import load, trigger  # noqa: E402

BITS = (8, 6, 4, 3, 2)
GRIDS = {"pc": "fbgemm8", "pt": "qnnpack8"}


def models():
    d = cohort.load_all()
    return d[(d.batch == "batch1") | ((d.batch == "batch2") & (d.group != "QCB"))]


def jobs():
    return [{"id": f"B::{r.model_key}", "priority": 0 if r.group == "QCB" else 1} for _, r in models().iterrows()]


@torch.no_grad()
def quant(w, bits, per_channel):
    qmax = float((1 << (bits - 1)) - 1)
    a = w.abs().reshape(w.shape[0], -1).amax(1).view(-1, *([1] * (w.dim() - 1))) if per_channel else w.abs().max()
    s = torch.clamp(a / qmax, min=1e-12)
    return torch.clamp(torch.round(w / s), -qmax, qmax) * s


def run(job, dev):
    _, key = job["id"].split("::")
    r = cohort.row(key)
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load(r.dataset)
    probe, test = norm(imgs.probe.to(dev)), norm(imgs.test.to(dev))
    qcb = r.group == "QCB"
    trig = norm(trigger(imgs.test.to(dev), r.trigger if qcb else "Patch6x6"))
    target = int(r.target) if qcb else None
    f_pred = M.logits(F, probe).argmax(1)
    head = {"model_key": key, "batch": r.batch, "group": r.group, "cell": r.cell, "family": r.family, "target": target, "K": K}
    rows = []
    for gname, recipe in GRIDS.items():
        rep, layers, _ = grids.replica(F, recipe, dev)
        W = {n: m.weight.detach().clone() for n, m in layers}
        dm = C.build(rep, recipe, imgs, norm, dev)       # calibrated on FP32 weights (ranges fixed below)
        acts = C.act_quantizers(dm)
        dm_layers = [(n, m) for n, m in dm.named_modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
        for b in BITS:
            wq = {n: quant(W[n], b, gname == "pc") for n in W}
            deploy.set_weights(layers, wq)
            with torch.no_grad():
                for (n, m), (n2, _) in zip(dm_layers, layers):
                    m.weight.copy_(wq[n2])
            for state, model in (("W", rep), ("Wc", dm)):
                if state == "Wc":
                    C.set_acts(acts, "clip:1")
                aa = M.acc_asr(model, test, imgs.test_y, trig, target, K)
                p = M.logits(model, probe).argmax(1)
                ch = p != f_pred
                pr = M.prop(model, probe, K)
                rows.append(head | {"grid": gname, "bits": b, "state": state, "acc": aa["acc"], "asr": aa.get("asr"),
                                    "agree_F": float((~ch).float().mean()), "dest_hist": M.hist(p[ch], K),
                                    "prop_s100": pr["s100"], "skew_s100": M.skew(pr["s100"]), "cls_s100": int(np.argmax(pr["s100"]))})
        del dm, rep
    return rows
