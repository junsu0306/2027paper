"""ST09 (2부): activation noise alone, without BN re-estimation under the noise (prediction.md).

ST08 limitation: every state re-estimated BN (and the activation ranges) under its own noise, so "FP weights + noise turn
QuEST W4A4 on" (FN, t 2 ASR .74) could need the noisy BN re-estimation. Here the deployment is calibrated CLEAN (the iao4
train-mode procedure with the activation quantizer replaced by the identity: BN and ranges from clean FP activations on the
defender's 2,000 images x 4), then the noise is applied at inference only (BN, ranges fixed).

States (weights x noise placement), noise = random-sign activation rounding residual x t (ST08, signs 7000 + k, no clamp):
  FNi  FP weights (R), clean calibration, noise at inference only, t in {0, 1, 2, 3}   (t 0 = no noise: R-like baseline)
  QNi  iao4-grid weights (Q = W + D), clean calibration, noise at inference only, t in {0, 1, 2, 3}  (t 0 = W4, FP acts)
  FNc  = ST08 FN at t 2 (noise in calibration and inference), reproduction check (P0)
Cohort: QuEST W4A4 40 and its twins (control 6, QAT-W4A4-IAO 16); the unaware normals are not needed for this question.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as Fn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, data, deploy, grids, measures as M  # noqa: E402
from qcbobs.models import load, trigger  # noqa: E402

REC = "iao4"
N_NOISE = 4
TS = (0.0, 1.0, 2.0, 3.0)
KINDS = ("QuEST-W4A4", "control-QuEST-W4A4", "QAT-W4A4-IAO")


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ST08 = _mod("st08", "experiments/ST08_fp_vs_q_noise/exp.py")
ST04, E21, E10 = ST08.ST04, ST08.E21, ST08.E10


def models():
    d = ST08.models()
    return d[d.kind.isin(KINDS)].reset_index(drop=True)


def jobs():
    return [{"id": f"P::{r.model_key}::{REC}", "priority": 0} for r in models().itertuples()]


class Identity:
    """Context: deploysim's IAO activation fake-quantizer replaced by the identity (clean calibration, t 0 readout)."""

    def __init__(self):
        self.DS = deploy._ds()

    def __enter__(self):
        self.orig = self.DS.fq_iao
        self.DS.fq_iao = lambda x, L, bits: x
        return self

    def __exit__(self, *a):
        self.DS.fq_iao = self.orig
        return False


def readout(dm, xp, test, test_y, trig, target, K):
    with torch.no_grad():
        p_ = Fn.softmax(M.logits(dm, xp).float(), 1)
    aa = M.acc_asr(dm, test, test_y, trig, target, K)
    return {"m": p_.mean(0).tolist(), "yX": p_.argmax(1).tolist(), "acc": aa["acc"], "asr": aa.get("asr")}


def run(job, dev):
    _, key, rec = job["id"].split("::")
    r = cohort.row(key)
    t0 = time.time()
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load(r.dataset)
    dimg = E10.defender_images(r.dataset)
    xp = norm(imgs.probe.to(dev))
    R0 = E21.bn_reest(F, E21.batches(dimg.train_u8[:2000], norm, dev, aug=True))
    for p in R0.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        pR = Fn.softmax(M.logits(R0, xp).float(), 1)
    qcb = r.group == "QCB"
    test = norm(imgs.test.to(dev))
    trig = norm(trigger(imgs.test.to(dev), r.trigger)) if qcb else None
    target = int(r.target) if qcb else None
    head = {"model_key": key, "batch": r.batch, "group": r.group, "cell": r.cell, "kind": ST04.kind(r), "dataset": r.dataset,
            "arch": r.arch, "recipe": rec, "K": K, "target": target}
    rows = [head | {"part": "ref", "m": pR.mean(0).tolist(), "yR": pR.argmax(1).tolist()}]
    rep, layers, _ = grids.replica(R0, rec, dev)
    gs = grids.grids(layers, rec)
    names = [n for n, _ in layers]
    W = {n: m.weight.detach().clone() for n, m in layers}
    D = {lg.name: lg.quant(W[lg.name]) - W[lg.name] for lg in gs}
    Qw = {n: W[n] + D[n] for n in names}
    args = (xp, test, imgs.test_y, trig, target, K)
    # inference-only noise on a clean-calibrated deployment
    for stn, Wt in (("FNi", W), ("QNi", Qw)):
        deploy.set_weights(layers, Wt)
        with Identity():
            dm = deploy.build(rep, rec, dimg, norm, dev)
        for p in dm.parameters():
            p.requires_grad_(False)
        for t in TS:
            for kk in ([None] if t == 0 else range(N_NOISE)):
                if t == 0:
                    with Identity():
                        out = readout(dm, *args)
                else:
                    gen = torch.Generator(device=dev).manual_seed(7000 + kk)
                    with ST08.Noise(t, gen, clamp=False):
                        out = readout(dm, *args)
                rows.append(head | {"part": "state", "state": stn, "t": t, "draw": "none" if kk is None else kk} | out)
        del dm
    # reproduction of ST08 FN at t 2 (noise in calibration and inference)
    deploy.set_weights(layers, W)
    for kk in range(N_NOISE):
        gen = torch.Generator(device=dev).manual_seed(7000 + kk)
        with ST08.Noise(2.0, gen, clamp=False):
            dm = deploy.build(rep, rec, dimg, norm, dev)
            for p in dm.parameters():
                p.requires_grad_(False)
            out = readout(dm, *args)
        del dm
        rows.append(head | {"part": "state", "state": "FNc", "t": 2.0, "draw": kk} | out)
    deploy.set_weights(layers, W)
    rows.append(head | {"part": "time", "sec": time.time() - t0})
    return rows
