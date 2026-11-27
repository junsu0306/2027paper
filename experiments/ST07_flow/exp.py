"""ST07 (2부): per-input flows under activation over-drive (prediction.md).

ST06: the sink of a perturbation is a property of (dataset x perturbation type), so a different perturbation in the same
model cannot serve as its reference; and the QuEST leak is per-input (a minority of inputs jump to the target), so the
mean feature change hides it. Here every probe image is followed: which class it comes from (R's prediction; labelled
variant: the probe label) and which class it goes to under over-drive, plus where the push that moved it lies.

Hypotheses: inflow into a generic sink comes mainly from a few (similar) source classes; leakage of a source-agnostic
backdoor comes from all source classes at a rate that does not depend on the source. And (CSO idea, per input): the
decision-changing push of inputs that leak into the target lies outside R's clean feature span more than for a sink.

States (one job = one model on iao4; ST05/ST06 construction and seeds): A t = 1 true residual (P0), A t in TS_A with
random-sign activation rounding residual x t, draws 0..N_NOISE-1 (seed 7000+k).
Per state, per probe image: predicted class yX, decision push m = (W_yX - W_yR) . dS and its part outside the clean
global span m_out (dS = head input of X minus head input of R; W = R's head). Also class-mean softmax (P0b), acc, ASR.
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
N_CAL = 10_000
TS_A = (2.0, 3.0)
E_GLOBAL = 0.95


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ST06 = _mod("st06", "experiments/ST06_self_reference/exp.py")
ST05, ST04, E21, E10 = ST06.ST05, ST06.ST04, ST06.E21, ST06.E10


def models():
    return ST05.models()


def jobs():
    pri = {"QuEST-W4A4": 0, "control-QuEST-W4A4": 0, "QAT-W4A4-IAO": 0, "unaware": 1}
    return [{"id": f"F::{r.model_key}::{REC}", "priority": pri[r.kind]} for r in models().itertuples()]


def _r(x, nd=5):
    return [round(float(v), nd) for v in x]


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
    hR = ST06.HeadInput(R0, K)
    lgR, SR = hR.run(R0, xp)
    hR.close()
    lpR = Fn.log_softmax(lgR.float(), 1)
    pR = lpR.exp()
    yR = pR.argmax(1)
    W = hR.head.weight.detach().double()
    _, s, Vh = torch.linalg.svd(SR, full_matrices=False)
    e = s ** 2
    r_g = int((torch.cumsum(e, 0) / e.sum() < E_GLOBAL).sum().item()) + 1
    Vg = Vh[:r_g]
    y = ST06.probe_labels(r.dataset).to(dev)
    qcb = r.group == "QCB"
    test = norm(imgs.test.to(dev))
    trig = norm(trigger(imgs.test.to(dev), r.trigger)) if qcb else None
    target = int(r.target) if qcb else None
    head = {"model_key": key, "batch": r.batch, "group": r.group, "cell": r.cell, "kind": ST04.kind(r), "dataset": r.dataset,
            "arch": r.arch, "recipe": rec, "K": K, "target": target}
    rows = [head | {"part": "ref", "m": pR.mean(0).tolist(), "yR": yR.tolist(), "y": y.tolist(), "pR_max": _r(pR.max(1).values),
                    "d": int(SR.shape[1]), "r_glob": r_g}]

    rep, layers, _ = grids.replica(R0, rec, dev)
    gs = grids.grids(layers, rec)
    names = [n for n, _ in layers]
    Wl = {n: m.weight.detach().clone() for n, m in layers}
    D = {lg.name: lg.quant(Wl[lg.name]) - Wl[lg.name] for lg in gs}
    deploy.set_weights(layers, {n: Wl[n] + D[n] for n in names})
    for t, kk in [(1.0, None)] + [(t, k) for t in TS_A for k in range(N_NOISE)]:
        gen = None if kk is None else torch.Generator(device=dev).manual_seed(7000 + kk)
        with ST05.ActOverdrive(t, gen):
            dm_ = deploy.build(rep, rec, imgs_cal, norm, dev)
            for p in dm_.parameters():
                p.requires_grad_(False)
            hk = ST06.HeadInput(dm_, K)
            lg, S = hk.run(dm_, xp)
            hk.close()
            aa = M.acc_asr(dm_, test, imgs.test_y, trig, target, K)
        del dm_
        p = Fn.softmax(lg.float(), 1)
        yX = p.argmax(1)
        dS = S - SR
        dW = W[yX] - W[yR]                                        # per image: dest row minus source row
        perp = dS - (dS @ Vg.T) @ Vg
        rows.append(head | {"part": "state", "fam": "A", "t": t, "draw": "true" if kk is None else kk,
                            "m": p.mean(0).tolist(), "yX": yX.tolist(),
                            "push": _r((dW * dS).sum(1)), "push_out": _r((dW * perp).sum(1)),
                            "kl": float((pR * (lpR - torch.log(p + 1e-30))).sum(1).mean()),
                            "acc": aa["acc"], "asr": aa.get("asr")})
    deploy.set_weights(layers, Wl)
    rows.append(head | {"part": "time", "sec": time.time() - t0})
    return rows
