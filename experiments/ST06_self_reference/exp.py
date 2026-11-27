"""ST06 (2부): a self-reference for the clean drift under activation over-drive (prediction.md).

ST05 found that over-driving the activation side of the iao4 deployment pulls clean inputs of QuEST W4A4 toward the target,
but the size of the drift is set by the dataset's generic sink class, so it only separated QuEST from normals when
normals of the same task were used as a reference (post hoc). Here the reference comes from the same model:
  (a) input-noise reference: the drift of R under Gaussian pixel noise (a perturbation that is not a deployment op),
      matched to the over-drive state by its damage (KL to R), is subtracted from the over-drive drift;
  (b) CSO-style removal (Yang, Miller, Kesidis, ICML 2026, idea only, no input generation): the change of the head
      input (penultimate feature) under over-drive is split into the part inside the clean feature space of R and the
      part outside; only the outside part is pushed through the head. Global span (all clean features) and class
      coordinate masks (classes from R's own predictions; a labelled variant uses the probe labels).

States (one job = one model on iao4; same construction and seeds as ST05):
  A    weights = aligned iao4 deployment of R, activation rounding residual x t with random signs (ST05 ActOverdrive),
       t in TS_A, draws 0..N_NOISE-1 (seed 7000+k as ST05); plus t = 1 with the true residual (P0 ASR)
  N    R (no deployment) on probe images + Gaussian pixel noise sigma (clamped to [0,1]), sigma in SIGMAS, draws 0..N_IN-1
Readouts per state on detector_probe (1,000 clean images): class-mean softmax m, predicted histogram h, KL(p_R||p_X),
mean head-input feature mS (d floats) and the head pushes W_R @ (mS_X - mS_R) raw / outside the global span /
outside the class masks (pseudo / true labels). Mechanism only: clean accuracy and, for QuEST, ASR (true trigger).
"""
from __future__ import annotations

import dataclasses
import functools
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, data, deploy, grids, measures as M  # noqa: E402
from qcbobs.models import load, trigger  # noqa: E402

REC = "iao4"
N_NOISE = 8
N_IN = 4
N_CAL = 10_000
TS_A = (2.0, 3.0)
SIGMAS = (0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24)
E_GLOBAL = 0.95       # energy kept in the clean global span (uncentered SVD of R's head inputs on the probe)
E_MASK = 0.90         # share of the positive class-c contribution covered by class c's coordinate mask


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
    return [{"id": f"S::{r.model_key}::{REC}", "priority": pri[r.kind]} for r in models().itertuples()]


@functools.lru_cache(maxsize=None)
def probe_labels(dataset: str, n: int = 1000) -> torch.Tensor:
    name = data.DS_FILE[dataset]
    z = np.load(data.DATA / "datasets/main" / f"{name}.npz")
    roles = json.loads((data.DATA / "datasets/main/manifests" / name / "split_base.json").read_text())["datasets"][name]["roles"]
    return torch.from_numpy(z["y_test"][np.asarray(roles["detector_probe"]["indices"][:n])].astype(np.int64))


class HeadInput:
    """Forward pre-hook on the last nn.Linear (the head): records its input while `on` (after any ActQ pre-hook)."""

    def __init__(self, model: nn.Module, K: int):
        lin = [m for m in model.modules() if isinstance(m, nn.Linear)]
        assert lin and lin[-1].out_features == K, "head not found"
        self.head, self.on, self.buf = lin[-1], False, []
        self.h = self.head.register_forward_pre_hook(self._hook)

    def _hook(self, mod, inp):
        if self.on:
            self.buf.append(inp[0].detach().flatten(1).double())

    @torch.no_grad()
    def run(self, model, x):
        self.on, self.buf = True, []
        lg = M.logits(model, x)
        self.on = False
        S = torch.cat(self.buf)
        self.buf = []
        return lg, S

    def close(self):
        self.h.remove()


def masks(S, W, labels, K):
    """Per class c: boolean mask of the coordinates covering E_MASK of the positive contribution S_j * W_cj on class-c
    images; an empty class gets an empty mask (nothing removed)."""
    out, sizes = [], []
    for c in range(K):
        idx = labels == c
        mk = torch.zeros(S.shape[1], dtype=torch.bool, device=S.device)
        if idx.any():
            g = (S[idx].mean(0) * W[c]).clamp(min=0)
            if g.sum() > 0:
                o = torch.argsort(g, descending=True)
                n = int(torch.searchsorted(torch.cumsum(g[o], 0), E_MASK * g.sum()).item()) + 1
                mk[o[:n]] = True
        out.append(mk)
        sizes.append(int(mk.sum()))
    return torch.stack(out), sizes


def run(job, dev):
    _, key, rec = job["id"].split("::")
    r = cohort.row(key)
    t0 = time.time()
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load(r.dataset)
    imgs_cal = dataclasses.replace(imgs, train_u8=imgs.train_u8[:N_CAL])
    x01 = imgs.probe.to(dev)
    xp = norm(x01)
    u8 = E10.defender_images(r.dataset).train_u8[:2000]
    R0 = E21.bn_reest(F, E21.batches(u8, norm, dev, aug=True))
    for p in R0.parameters():
        p.requires_grad_(False)
    hR = HeadInput(R0, K)
    lgR, SR = hR.run(R0, xp)
    hR.close()
    lpR = Fn.log_softmax(lgR.float(), 1)
    pR = lpR.exp()
    W = hR.head.weight.detach().double()                    # K x d, R's head (FP)
    mSR = SR.mean(0)
    # clean global span (uncentered) and class coordinate masks
    _, s, Vh = torch.linalg.svd(SR, full_matrices=False)
    e = s ** 2
    r_g = int((torch.cumsum(e, 0) / e.sum() < E_GLOBAL).sum().item()) + 1
    Vg = Vh[:r_g]
    predR = pR.argmax(1)
    yprobe = probe_labels(r.dataset).to(dev)
    mk_p, sz_p = masks(SR, W, predR, K)
    mk_t, sz_t = masks(SR, W, yprobe, K)

    def pushes(mS):
        dm = mS - mSR
        perp = dm - Vg.T @ (Vg @ dm)
        return {"push_raw": (W @ dm).tolist(), "push_glob": (W @ perp).tolist(),
                "push_mask": ((W * dm) * (~mk_p)).sum(1).tolist(), "push_mask_true": ((W * dm) * (~mk_t)).sum(1).tolist(),
                "dm_norm": float(dm.norm()), "perp_share": float(perp.norm() / (dm.norm() + 1e-30))}

    qcb = r.group == "QCB"
    test = norm(imgs.test.to(dev))
    trig01 = trigger(imgs.test.to(dev), r.trigger) if qcb else None
    target = int(r.target) if qcb else None
    head = {"model_key": key, "batch": r.batch, "group": r.group, "cell": r.cell, "kind": ST04.kind(r), "dataset": r.dataset,
            "arch": r.arch, "recipe": rec, "K": K, "target": target}
    rows = [head | {"part": "ref", "m": pR.mean(0).tolist(), "h": (torch.bincount(predR, minlength=K).float() / len(pR)).tolist(),
                    "acc_probe_R": float((predR.cpu() == yprobe.cpu()).float().mean()), "d": int(SR.shape[1]), "r_glob": r_g,
                    "mask_sizes": sz_p, "mask_sizes_true": sz_t, "mS": mSR.tolist()}]

    def record(lg, S, fam, t, draw, acc_asr):
        lp = Fn.log_softmax(lg.float(), 1)
        p = lp.exp()
        mS = S.mean(0)
        rows.append(head | {"part": "state", "fam": fam, "t": t, "draw": draw, "m": p.mean(0).tolist(),
                            "h": (torch.bincount(p.argmax(1), minlength=K).float() / len(p)).tolist(),
                            "kl": float((pR * (lpR - lp)).sum(1).mean()), "mS": mS.tolist()} | pushes(mS) | acc_asr)

    # ---- A: aligned weights, activation over-drive (ST05 construction)
    rep, layers, _ = grids.replica(R0, rec, dev)
    gs = grids.grids(layers, rec)
    names = [n for n, _ in layers]
    Wl = {n: m.weight.detach().clone() for n, m in layers}
    D = {lg.name: lg.quant(Wl[lg.name]) - Wl[lg.name] for lg in gs}
    Q = {n: Wl[n] + D[n] for n in names}
    deploy.set_weights(layers, Q)
    trig = norm(trig01) if qcb else None
    for t, kk in [(1.0, None)] + [(t, k) for t in TS_A for k in range(N_NOISE)]:
        gen = None if kk is None else torch.Generator(device=dev).manual_seed(7000 + kk)
        with ST05.ActOverdrive(t, gen):
            dm_ = deploy.build(rep, rec, imgs_cal, norm, dev)
            for p in dm_.parameters():
                p.requires_grad_(False)
            hk = HeadInput(dm_, K)
            lg, S = hk.run(dm_, xp)
            hk.close()
            aa = M.acc_asr(dm_, test, imgs.test_y, trig, target, K)
        record(lg, S, "A", t, "true" if kk is None else kk, {"acc": aa["acc"], "asr": aa.get("asr")})
        del dm_
    deploy.set_weights(layers, Wl)
    del rep

    # ---- N: R with Gaussian pixel noise (not a deployment op)
    hN = HeadInput(R0, K)
    for sg in SIGMAS:
        for kk in range(N_IN):
            g = torch.Generator(device=dev).manual_seed(9000 + kk)
            nz = lambda x: (x + sg * torch.randn(x.shape, generator=g, device=dev)).clamp(0, 1)  # noqa: E731
            lg, S = hN.run(R0, norm(nz(x01)))
            aa = M.acc_asr(R0, norm(nz(imgs.test.to(dev))), imgs.test_y, norm(nz(trig01)) if qcb else None, target, K)
            record(lg, S, "N", sg, kk, {"acc": aa["acc"], "asr": aa.get("asr")})
    hN.close()
    rows.append(head | {"part": "time", "sec": time.time() - t0})
    return rows
