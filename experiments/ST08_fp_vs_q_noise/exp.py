"""ST08 (2부): the same activation noise in FP and in the quantized deployment (prediction.md).

ST06: a perturbation of another kind cannot serve as the reference (the sink depends on dataset x perturbation type).
Here the noise is the SAME: a random-sign activation rounding residual x t on the 4-bit IAO grid of each activation site.
Only the quantization differs. A normal model should drift to the same sink with or without quantization; a QCB whose
leak needs the quantized deployment should drift to the target only when quantized.

2 x 2 states at each noise size t (draws 0..N_NOISE-1, signs seeded 7000+k as ST05):
  QA  weights on the iao4 grid (Q = W + D), activations clamped to the calibrated range, + noise       (= ST05 A)
  QN  weights on the grid, activations NOT clamped, + noise
  FA  FP weights (R), activations clamped, + noise
  FN  FP weights (R), activations NOT clamped, + noise                                                ("FP + same noise")
No state rounds the activations (noise replaces the rounding: x/s + t * r * sign, r = round(x/s) - x/s). Every state is
built by the same deploysim iao4_train procedure (train-mode calibration of the ranges and BN under its own noise), on
the defender's 2,000 unlabelled images repeated 4 times (no training images). P0: QA t 1 with the true residual.
Readouts on detector_probe (1,000): class-mean softmax m, per-image prediction; mechanism only: clean accuracy, ASR.
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
TS = (1.0, 1.5, 2.0, 3.0)
STATES = ("QA", "QN", "FA", "FN")


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ST05 = _mod("st05", "experiments/ST05_overdrive/exp.py")
ST04, E21, E10 = ST05.ST04, ST05.E21, ST05.E10


def models():
    return ST05.models()


def jobs():
    pri = {"QuEST-W4A4": 0, "control-QuEST-W4A4": 0, "QAT-W4A4-IAO": 0, "unaware": 1}
    return [{"id": f"P::{r.model_key}::{REC}", "priority": pri[r.kind]} for r in models().itertuples()]


class Noise:
    """Context: deploysim's IAO activation fake-quantizer replaced by x/s (+ clamp if `clamp`) + t * r * sign, scaled
    back; gen None -> sign +1 (t 1, clamp: the true IAO quantizer)."""

    def __init__(self, t: float, gen, clamp: bool):
        self.t, self.gen, self.clamp = t, gen, clamp
        self.DS = deploy._ds()

    def __enter__(self):
        DS, t, gen, clamp = self.DS, self.t, self.gen, self.clamp
        self.orig = DS.fq_iao
        eps = float(torch.finfo(torch.float32).eps)

        def fq(x, L, bits):
            qmin, qmax = -float(1 << (bits - 1)), float((1 << (bits - 1)) - 1)
            s = max(L / ((qmax - qmin) / 2), eps)
            v = x / s
            if clamp:
                v = torch.clamp(v, qmin, qmax)
            r = DS._round_away(v) - v
            if gen is not None:
                r = r * (torch.randint(0, 2, r.shape, generator=gen, device=r.device) * 2 - 1)
            return (v + t * r) * s
        DS.fq_iao = fq
        return self

    def __exit__(self, *a):
        self.DS.fq_iao = self.orig
        return False


def run(job, dev):
    _, key, rec = job["id"].split("::")
    r = cohort.row(key)
    t0 = time.time()
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load(r.dataset)
    dimg = E10.defender_images(r.dataset)              # defender's 2,000 unlabelled images; train_u8 = them x 4
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
    plan = [("QA", 1.0, None)] + [(s, t, k) for s in STATES for t in TS for k in range(N_NOISE)]
    for stn, t, kk in plan:
        deploy.set_weights(layers, Qw if stn[0] == "Q" else W)
        gen = None if kk is None else torch.Generator(device=dev).manual_seed(7000 + kk)
        with Noise(t, gen, clamp=stn[1] == "A"):
            dm = deploy.build(rep, rec, dimg, norm, dev)
            for p in dm.parameters():
                p.requires_grad_(False)
            with torch.no_grad():
                p_ = Fn.softmax(M.logits(dm, xp).float(), 1)
            aa = M.acc_asr(dm, test, imgs.test_y, trig, target, K)
        del dm
        rows.append(head | {"part": "state", "state": stn, "t": t, "draw": "true" if kk is None else kk,
                            "m": p_.mean(0).tolist(), "yX": p_.argmax(1).tolist(), "acc": aa["acc"], "asr": aa.get("asr")})
    deploy.set_weights(layers, W)
    rows.append(head | {"part": "time", "sec": time.time() - t0})
    return rows
