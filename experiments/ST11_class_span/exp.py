"""ST11 (2부, mechanism): is the trigger's feature change outside the TARGET class's clean span (CSO premise), per layer?

Labelled clean data (the allowed fallback assumption): the defender's 2,000 labelled images give each class's clean
feature span at each stage (n_lab per class; global-average-pooled stage outputs and the head input). Centred by the
mean feature of all span images; basis = 95 % energy of the class's centred features (ST06 convention).
Changes measured on the 1,000 labelled test images (disjoint from the defender images), images of other classes:
  trig  f(x + trigger) - f(x)            (true trigger: mechanism only)
  ctl   f(x + random-colour patch at the trigger's pixels) - f(x)
  nat   mu_c - f(x)                      (towards the class mean of the span images: the natural direction)
in_c(.) = sum ||U_c^T d||^2 / sum ||d||^2;  zeta_c = in_c(trig) / in_c(nat).  QuEST: c = target (and the median over
the other classes); twins and normals: the median over all classes.
States: R (FP, BN re-estimated clean), Q (true iao4 deployment), S2n (FP weights, deployment calibrated and evaluated under
the ST08 noise t 2, draw 0 = ST09 FNc), S2c (the S2n deployment evaluated WITHOUT noise: does the noisy BN alone switch?).
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, data, deploy, grids, measures as M  # noqa: E402
from qcbobs.models import load, trigger  # noqa: E402

REC = "iao4"
N_LABS = (10, 50)
ENERGY = .95


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ST09 = _mod("st09", "experiments/ST09_noise_only/exp.py")
ST10 = _mod("st10", "experiments/ST10_stationarity_attack_regime/exp.py")
ST08 = ST09.ST08
ST04, E21, E10 = ST08.ST04, ST08.E21, ST08.E10


def models():
    return ST08.models()


def jobs():
    pri = {"QuEST-W4A4": 0, "control-QuEST-W4A4": 0, "QAT-W4A4-IAO": 0, "unaware": 1}
    return [{"id": f"P::{r.model_key}::{REC}", "priority": pri[r.kind]} for r in models().itertuples()]


def stage_modules(model, arch):
    mods = dict(model.named_modules())
    if arch == "VGG16-BN":
        names = [n for n, m in model.named_modules() if isinstance(m, nn.MaxPool2d)]
    else:
        names = [n for n in ("layer1", "layer2", "layer3", "layer4") if n in mods]
    head = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
    return names, mods, head


@torch.no_grad()
def feats(model, arch, x, norm, dev, bs=250):
    names, mods, head = stage_modules(model, arch)
    buf = {}

    def hk(n):
        def h(m, a, o):
            buf[n] = o.float().mean((2, 3)) if o.dim() == 4 else o.float()
        return h

    def hp(m, a):
        buf["head"] = a[0].float().flatten(1)
    hs = [mods[n].register_forward_hook(hk(n)) for n in names] + [head.register_forward_pre_hook(hp)]
    out = {n: [] for n in names + ["head"]}
    pred = []
    for s in range(0, len(x), bs):
        lo = model(norm(x[s:s + bs].to(dev)))
        pred.append(lo.argmax(1).cpu())
        for n in out:
            out[n].append(buf[n])
    for h in hs:
        h.remove()
    return {n: torch.cat(v) for n, v in out.items()}, torch.cat(pred)


def basis(z):
    u, s, _ = torch.linalg.svd(z.T, full_matrices=False)
    e = (s ** 2).cumsum(0) / (s ** 2).sum()
    k = int((e < ENERGY).sum()) + 1
    return u[:, :k]


def inshare(U, d):
    return float((d @ U).pow(2).sum() / d.pow(2).sum().clamp_min(1e-30))


def readout(Fd, Fx, Ft, Fc, yd, yx, classes, n_lab, seed=0):
    """Per layer and class c: in_c(trig), in_c(ctl), in_c(nat), rank / dim."""
    rng = np.random.default_rng(seed)
    pick = {c: rng.permutation(np.flatnonzero(yd.numpy() == c))[:n_lab] for c in classes}
    allp = np.concatenate([pick[c] for c in classes])
    out = {}
    for n in Fd:
        mu = Fd[n][allp].mean(0)
        res = {}
        for c in classes:
            if len(pick[c]) < 2:
                continue
            Zc = Fd[n][pick[c]]
            U = basis(Zc - mu)
            k = yx != c
            dt, dc = Ft[n][k] - Fx[n][k], Fc[n][k] - Fx[n][k]
            dn = Zc.mean(0)[None] - Fx[n][k]
            res[c] = (inshare(U, dt), inshare(U, dc), inshare(U, dn), U.shape[1] / Zc.shape[1])
        out[n] = res
    return out


def run(job, dev):
    _, key, rec = job["id"].split("::")
    r = cohort.row(key)
    t0 = time.time()
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load(r.dataset)
    dimg = E10.defender_images(r.dataset)
    xd, yd = ST10.defender_labelled(r.dataset)
    x, yx = imgs.test, imgs.test_y
    qcb = r.group == "QCB"
    tname = r.trigger if qcb else "Patch6x6"
    xt = trigger(x.to(dev), tname).cpu()
    msk = (xt != x).any(1, keepdim=True)
    g = torch.Generator().manual_seed(5)
    xc = torch.where(msk, torch.rand(x.shape, generator=g), x)
    target = int(r.target) if qcb else None
    head = {"model_key": key, "batch": r.batch, "group": r.group, "cell": r.cell, "kind": ST04.kind(r), "dataset": r.dataset,
            "arch": r.arch, "recipe": rec, "K": K, "target": target}
    R0 = E21.bn_reest(F, E21.batches(dimg.train_u8[:2000], norm, dev, aug=True))
    for p in R0.parameters():
        p.requires_grad_(False)
    rep, layers, _ = grids.replica(R0, rec, dev)
    gs = grids.grids(layers, rec)
    names = [n for n, _ in layers]
    W = {n: m.weight.detach().clone() for n, m in layers}
    D = {lg.name: lg.quant(W[lg.name]) - W[lg.name] for lg in gs}
    deploy.set_weights(layers, {n: W[n] + D[n] for n in names})
    Q = deploy.build(rep, rec, dimg, norm, dev)
    deploy.set_weights(layers, W)
    with ST08.Noise(2.0, torch.Generator(device=dev).manual_seed(7000), clamp=False):
        S2 = deploy.build(rep, rec, dimg, norm, dev)
    rows = []
    for stn, model, ctx in (("R", R0, None), ("Q", Q, None), ("S2n", S2, "noise"), ("S2c", S2, "clean")):
        model.eval()

        def ev(fn):
            if ctx == "noise":
                with ST08.Noise(2.0, torch.Generator(device=dev).manual_seed(7000), clamp=False):
                    return fn()
            if ctx == "clean":
                with ST09.Identity():
                    return fn()
            return fn()
        Fd, _ = ev(lambda: feats(model, r.arch, xd, norm, dev))
        Fx, px = ev(lambda: feats(model, r.arch, x, norm, dev))
        Ft, pt = ev(lambda: feats(model, r.arch, xt, norm, dev))
        Fc, pc = ev(lambda: feats(model, r.arch, xc, norm, dev))
        acc = float((px == yx).float().mean())
        rec_ = {"part": "state", "state": stn, "acc": acc}
        if qcb:
            keep = yx != target
            rec_["asr"] = float((pt[keep] == target).float().mean())
            rec_["ctl_to_target"] = float((pc[keep] == target).float().mean())
        classes = [target] if qcb else list(range(K))
        if qcb:
            classes = list(range(K))
        for n_lab in N_LABS:
            ro = readout(Fd, Fx, Ft, Fc, yd, yx, classes, n_lab)
            lay = {}
            for ln, res in ro.items():
                arr = {c: v for c, v in res.items()}
                lay[ln] = {str(c): [round(float(a), 5) for a in v] for c, v in arr.items()}
            rows.append(head | rec_ | {"n_lab": n_lab, "layers": list(ro.keys()), "inshare": lay})
    rows.append(head | {"part": "time", "sec": time.time() - t0})
    return rows
