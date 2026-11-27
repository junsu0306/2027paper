"""ST05 (2부): over-driving the activation side of the deployment (prediction.md).

Principle: a QCB is trained to look clean at exactly the deployed state. QuEST W4A4 is sharp in the weight rounding
pattern but tolerant on the activation side (E21 B; pilot 2026-11-27: random-sign activation rounding x2 switches it on).
If the backdoor is reachable by activation noise, extra activation noise in the deployment should pull clean predictions
toward the target, while a no-poison twin drifts only to its generic noise sink. No trigger and no input is generated.

States (one job = one model on iao4; deploy.build, iao train-mode calibration on the first N_CAL training images):
  A    weights = the aligned deployment (iao4 rounding of R), activation rounding residual of every IAO quantizer scaled by
       t (clipping unchanged); 'true' = deterministic residual x t, draws 0..N_NOISE-1 = random signs (noise)    t in TS_A
  Aoff weights = R + random-sign rounding residual (the weight switch off; same signs per draw), activations as A with
       random signs                                                                                             t in TS_OFF
Readouts on detector_probe (1,000 clean images, no labels): class-mean softmax m(c), predicted-class histogram h(c),
KL(p_R || p_X), mean entropy. Mechanism only: clean accuracy (test labels) and, for QuEST, ASR with the true trigger.
Start from R = F with BN re-estimated on the defender's images (ST04, forgery-invariant). Cohort: the ST04 iao4 arm.
"""
from __future__ import annotations

import dataclasses
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
N_NOISE = 8
N_CAL = 10_000          # iao train-mode calibration images (E21 B: 512-10,000 all switch QuEST W4A4 on; BN momentum .1)
TS_A = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
TS_OFF = (1.0, 2.0, 3.0)


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ST04 = _mod("st04", "experiments/ST04_quest_necessary_trace/exp.py")
E21, E10 = ST04.E21, ST04.E10


def models():
    d = ST04.models()
    return d[d.kind.isin(ST04.ARMS[REC])].reset_index(drop=True)


def jobs():
    pri = {"QuEST-W4A4": 0, "control-QuEST-W4A4": 0, "QAT-W4A4-IAO": 0, "unaware": 1}
    return [{"id": f"N::{r.model_key}::{REC}", "priority": pri[r.kind]} for r in models().itertuples()]


def _sign(shape, gen, dev):
    return torch.randint(0, 2, shape, generator=gen, device=dev) * 2 - 1


class ActOverdrive:
    """Context: deploysim's IAO activation fake-quantizer with its rounding residual scaled by t (clipping unchanged);
    with gen: random signs on that residual (noise)."""

    def __init__(self, t: float, gen=None):
        self.t, self.gen = t, gen
        self.DS = deploy._ds()

    def __enter__(self):
        DS, t, gen = self.DS, self.t, self.gen
        self.orig = DS.fq_iao
        eps = float(torch.finfo(torch.float32).eps)

        def fq(x, L, bits):
            qmin, qmax = -float(1 << (bits - 1)), float((1 << (bits - 1)) - 1)
            s = max(L / ((qmax - qmin) / 2), eps)
            v = torch.clamp(x / s, qmin, qmax)
            r = DS._round_away(v) - v
            if gen is not None:
                r = r * _sign(r.shape, gen, r.device)
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
    imgs_cal = dataclasses.replace(imgs, train_u8=imgs.train_u8[:N_CAL])
    xp = norm(imgs.probe.to(dev))
    u8 = E10.defender_images(r.dataset).train_u8[:2000]
    R0 = E21.bn_reest(F, E21.batches(u8, norm, dev, aug=True))
    for p in R0.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        lpR = Fn.log_softmax(M.logits(R0, xp).float(), 1)
    qcb = r.group == "QCB"
    test = norm(imgs.test.to(dev))
    trig = norm(trigger(imgs.test.to(dev), r.trigger)) if qcb else None
    target = int(r.target) if qcb else None
    head = {"model_key": key, "batch": r.batch, "group": r.group, "cell": r.cell, "kind": ST04.kind(r), "dataset": r.dataset,
            "arch": r.arch, "recipe": rec, "K": K, "target": target}
    pR = lpR.exp()
    rows = [head | {"part": "ref", "m": pR.mean(0).tolist(),
                    "h": (torch.bincount(pR.argmax(1), minlength=K).float() / len(pR)).tolist(),
                    "ent": float(-(pR * lpR).sum(1).mean())}]

    @torch.no_grad()
    def readout(dm, fam, t, draw):
        lp = Fn.log_softmax(M.logits(dm, xp).float(), 1)
        p = lp.exp()
        rec_ = head | {"part": "state", "fam": fam, "t": t, "draw": draw, "m": p.mean(0).tolist(),
                       "h": (torch.bincount(p.argmax(1), minlength=K).float() / len(p)).tolist(),
                       "kl": float((pR * (lpR - lp)).sum(1).mean()), "ent": float(-(p * lp).sum(1).mean())}
        aa = M.acc_asr(dm, test, imgs.test_y, trig, target, K)
        rec_["acc"], rec_["asr"] = aa["acc"], aa.get("asr")
        rows.append(rec_)

    rep, layers, _ = grids.replica(R0, rec, dev)
    gs = grids.grids(layers, rec)
    names = [n for n, _ in layers]
    W = {n: m.weight.detach().clone() for n, m in layers}
    D = {lg.name: lg.quant(W[lg.name]) - W[lg.name] for lg in gs}
    Q = {n: W[n] + D[n] for n in names}

    def w_off(kk):
        gen = torch.Generator(device=dev).manual_seed(5000 + kk)
        return {n: W[n] + D[n] * _sign(W[n].shape, gen, dev) for n in names}

    def state(fam, t, kk, ws):
        gen = None if kk is None else torch.Generator(device=dev).manual_seed(7000 + kk)
        deploy.set_weights(layers, ws)
        with ActOverdrive(t, gen):
            dm = deploy.build(rep, rec, imgs_cal, norm, dev)
            for p in dm.parameters():
                p.requires_grad_(False)
            readout(dm, fam, t, "true" if kk is None else kk)
        del dm

    for t in TS_A:
        for kk in [None] + list(range(N_NOISE)):
            state("A", t, kk, Q)
    for t in TS_OFF:
        for kk in range(N_NOISE):
            state("Aoff", t, kk, w_off(kk))
    deploy.set_weights(layers, W)
    rows.append(head | {"part": "time", "sec": time.time() - t0})
    return rows
