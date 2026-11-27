"""ST10 (2부): stationarity residual in the attack's own training regime, and which class-specific loss explains it.

QuEST W4A4 objective (harness quest4/core.py, recipe_worker.py), read from the code:
  per batch  CE_F(x, y) + CE_F(b, y') + CE_Q(x, y) + CE_Q(b, y'),  b = x + trigger and y' = target for 10 % of the
  samples (else b = x, y' = y); both paths in TRAIN mode (BN batch statistics; Q = IAO W4A4 fake quantization);
  SGD weight decay 5e-4; only a Fisher-selected mask of weights is trained; augmentation: scale jitter (95 % none).
At a stationary point of the masked weights:  r := g_F + g_Q + 2.5e-4 w = -0.05 [P_F + P_Q]  (P = poison-term gradients),
so r should point along "raise the target in Q on triggered inputs, keep the true label in F". ST04 measured the clean
gradients in eval mode on the deployed model (not the attack regime) and with 1:1 weights on all weights.

Here, per model and labelled data set D (train 8,192 = the attacker's data, oracle; defender 2,000 labelled; small = 10
per class of the defender set), in the attack regime (train-mode BN, batch 128, no augmentation):
  g_F, g_Q  mean clean CE gradients of the conv / linear weights (F = the suspect's FP weights; Q = its iao4 deployment
            with the calibrated activation ranges frozen, straight-through rounding)
  proxy mask  clean Q-Fisher top 20 % minus clean F-Fisher top 5 % (global; the attack used poisoned labels)
  B         ||r|| / (||g_F|| + ||g_Q|| + 2.5e-4 ||w||), on all weights, on the proxy mask, and per stage
  class scores  delta_c = grad mean_{y != c} [log p_Q(c|x) - log p_F(c|x)] on a 512-image subset of D;
            s_c = cos(r, delta_c) on the proxy mask; also cos(r, grad log p_Q(c|x)) alone
  fitted    the defender does not know the attack's coefficients: r_fit = g_F + a g_Q + l w with (a, l) the least-squares
            minimiser on the proxy mask (only the generic form "F clean + Q clean" is assumed); B_fit, s_c^fit
Mechanism only: QuEST's deployed (eval) ASR.
"""
from __future__ import annotations

import copy
import importlib.util
import json
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
from qcbobs.env import DATA  # noqa: E402
from qcbobs.models import load, trigger  # noqa: E402

REC = "iao4"
C_WD = 2.5e-4
BATCH = 128
N_TRAIN = 8192
N_SMALL = 10
N_CLS = 512


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ST08 = _mod("st08", "experiments/ST08_fp_vs_q_noise/exp.py")
ST04, E21, E10 = ST08.ST04, ST08.E21, ST08.E10
ST04X = _mod("st04x", "experiments/ST04_quest_necessary_trace/exp.py")


def models():
    return ST08.models()


def jobs():
    pri = {"QuEST-W4A4": 0, "control-QuEST-W4A4": 0, "QAT-W4A4-IAO": 0, "unaware": 1}
    return [{"id": f"P::{r.model_key}::{REC}", "priority": pri[r.kind]} for r in models().itertuples()]


def defender_labelled(dataset):
    name = data.DS_FILE[dataset]
    z = np.load(DATA / "datasets/main" / f"{name}.npz")
    roles = json.loads((DATA / "datasets/main/manifests" / name / "split_base.json").read_text())["datasets"][name]["roles"]
    idx = np.asarray(roles["final_behavior_test"]["indices"][E10.DEF_RANGE[0]:E10.DEF_RANGE[1]])
    return data.to01(z["x_test"][idx]), torch.from_numpy(z["y_test"][idx].astype(np.int64))


def per_class(x, y, n, seed=0):
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y.numpy()):
        i = np.flatnonzero(y.numpy() == c)
        keep += list(rng.permutation(i)[:n])
    keep = np.sort(np.asarray(keep))
    return x[keep], y[keep]


def train_bn(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.train()
    return model


def weights(model, names):
    p = dict(model.named_parameters())
    ws = [p[f"{n}.weight"] for n in names]
    for w in ws:
        w.requires_grad_(True)
    return ws


def ce_grads(model, ws, x, y, norm, dev):
    """Mean CE gradient over (x, y) in batches of BATCH (train-mode BN) and the batch-mean-gradient-squared Fisher."""
    g = [torch.zeros_like(w) for w in ws]
    f = [torch.zeros_like(w) for w in ws]
    nb = 0
    for s in range(0, len(x), BATCH):
        xb, yb = norm(x[s:s + BATCH].to(dev)), y[s:s + BATCH].to(dev)
        loss = Fn.cross_entropy(model(xb), yb)
        gb = torch.autograd.grad(loss, ws)
        for a, b, c in zip(g, f, gb):
            a.add_(c, alpha=len(xb) / len(x))
            b.add_(c * c)
        nb += 1
    return g, [b / nb for b in f]


def class_grad(model, ws, x, y, c, norm, dev):
    """grad of mean_{y != c} log p(c|x) (train-mode BN)."""
    g = [torch.zeros_like(w) for w in ws]
    n = int((y != c).sum())
    for s in range(0, len(x), BATCH):
        xb, yb = norm(x[s:s + BATCH].to(dev)), y[s:s + BATCH].to(dev)
        lp = Fn.log_softmax(model(xb), 1)[:, c]
        k = yb != c
        if not k.any():
            continue
        gb = torch.autograd.grad(lp[k].sum() / n, ws)
        for a, b in zip(g, gb):
            a.add_(b)
    return g


def topmask(score, frac):
    flat = torch.cat([s.reshape(-1) for s in score])
    k = int(frac * flat.numel())
    thr = torch.topk(flat, k).values.min()
    return [s >= thr for s in score]


def flat(vs, mask):
    return torch.cat([v[m].reshape(-1) for v, m in zip(vs, mask)])


def run(job, dev):
    _, key, rec = job["id"].split("::")
    r = cohort.row(key)
    t0 = time.time()
    torch.manual_seed(0)
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load(r.dataset)
    dimg = E10.defender_images(r.dataset)
    qcb = r.group == "QCB"
    target = int(r.target) if qcb else None
    head = {"model_key": key, "batch": r.batch, "group": r.group, "cell": r.cell, "kind": ST04.kind(r), "dataset": r.dataset,
            "arch": r.arch, "recipe": rec, "K": K, "target": target}
    # Q path: iao4 deployment of F (weights on the grid, ranges calibrated on the defender images, then frozen)
    rep, layers, _ = grids.replica(F, rec, dev)
    names = [n for n, _ in layers]
    gs = grids.grids(layers, rec)
    W = {n: m.weight.detach().clone() for n, m in layers}
    D = {lg.name: lg.quant(W[lg.name]) - W[lg.name] for lg in gs}
    deploy.set_weights(layers, {n: W[n] + D[n] for n in names})
    Q = deploy.build(rep, rec, dimg, norm, dev)
    deploy.set_weights(layers, W)
    rows = []
    if qcb:
        with torch.no_grad():
            aa = M.acc_asr(Q.eval(), norm(imgs.test.to(dev)), imgs.test_y, norm(trigger(imgs.test.to(dev), r.trigger)), target, K)
        rows.append(head | {"part": "mech", "acc_Q": aa["acc"], "asr_Q": aa["asr"]})
    Fm = train_bn(copy.deepcopy(F))
    Q = train_bn(Q)
    wsF, wsQ = weights(Fm, names), weights(Q, names)
    wvec = [W[n] for n in names]
    stage = ST04X.stage_of_layers(names, [m for _, m in layers], r.arch)
    xt, yt = ST04X.labeled_train(r.dataset, N_TRAIN)
    xd, yd = defender_labelled(r.dataset)
    xs, ys = per_class(xd, yd, N_SMALL)
    for dname, (x, y) in (("train", (xt, yt)), ("defender", (xd, yd)), ("small", (xs, ys))):
        gF, fF = ce_grads(Fm, wsF, x, y, norm, dev)
        gQ, fQ = ce_grads(Q, wsQ, x, y, norm, dev)
        res = [a + b + C_WD * w for a, b, w in zip(gF, gQ, wvec)]
        keepQ = topmask(fQ, .20)
        dropF = topmask(fF, .05)
        mask = [a & ~b for a, b in zip(keepQ, dropF)]
        full = [torch.ones_like(m) for m in mask]

        def B(msk, sel=None):
            ii = range(len(names)) if sel is None else [i for i, n in enumerate(names) if stage[n] == sel]
            if not len(list(ii)):
                return float("nan")
            ii = list(ii)
            num = torch.cat([res[i][msk[i]] for i in ii]).norm()
            den = (torch.cat([gF[i][msk[i]] for i in ii]).norm() + torch.cat([gQ[i][msk[i]] for i in ii]).norm()
                   + C_WD * torch.cat([wvec[i][msk[i]] for i in ii]).norm())
            return float(num / den)
        wn = flat(wvec, full).norm()
        rec_ = {"part": "station", "data": dname, "n": len(x), "gF_rel": float(flat(gF, full).norm() / wn),
                "gQ_rel": float(flat(gQ, full).norm() / wn), "B_all": B(full), "B_mask": B(mask),
                "mask_frac": float(sum(int(m.sum()) for m in mask) / sum(m.numel() for m in mask)),
                "cos_FQ_mask": float(Fn.cosine_similarity(flat(gF, mask), flat(gQ, mask), 0))}
        for sname in sorted(set(stage.values())):
            rec_[f"B_mask_{sname}"] = B(mask, sname)
        rm = flat(res, mask)
        fa, fb, fw = flat(gF, mask).double(), flat(gQ, mask).double(), flat(wvec, mask).double()
        A = torch.stack([torch.stack([fb @ fb, fb @ fw]), torch.stack([fw @ fb, fw @ fw])])
        al, la = torch.linalg.solve(A, -torch.stack([fb @ fa, fw @ fa])).tolist()
        rfit = (fa + al * fb + la * fw).float()
        rec_ |= {"alpha_fit": al, "lam_fit": la,
                 "B_fit_mask": float(rfit.norm() / (fa.norm() + abs(al) * fb.norm() + abs(la) * fw.norm()))}
        sub = torch.randperm(len(x), generator=torch.Generator().manual_seed(1))[:N_CLS]
        xc, yc = x[sub], y[sub]
        s_d, s_q, s_f = [], [], []
        for c in range(K):
            dQ = class_grad(Q, wsQ, xc, yc, c, norm, dev)
            dF = class_grad(Fm, wsF, xc, yc, c, norm, dev)
            vq = flat(dQ, mask)
            vd = vq - flat(dF, mask)
            s_d.append(float(Fn.cosine_similarity(rm, vd, 0)))
            s_f.append(float(Fn.cosine_similarity(rfit, vd, 0)))
            s_q.append(float(Fn.cosine_similarity(rm, vq, 0)))
        rows.append(head | rec_ | {"s_delta": s_d, "s_q": s_q, "s_fit": s_f})
        del gF, gQ, fF, fQ, res
    rows.append(head | {"part": "time", "sec": time.time() - t0})
    return rows
