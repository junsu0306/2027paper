"""ST04 (2부): traces a QuEST backdoor needs (so a forger cannot remove them), five readouts in one job (prediction.md).

Jobs = (model, recipe). W4A4 arm (iao4): QuEST W4A4, no-poison QuEST-IAO-W4A4 controls, QAT-W4A4-IAO normals, unaware
normals. 8-bit arm (iao8): QuEST-8, QuEST-IAO-W8A8 controls, QAT-W8weight normals, the same unaware normals.
Everything starts from R = F with BN statistics re-estimated on the defender's unlabeled images (FP32, augmentation, ST03
procedure), so no readout depends on the stored BN statistics. Q = full deployment of R's weights rounded on the recipe grid
(deploy.build, train-mode calibration on the whole training set as deploysim; iao re-estimates BN anyway).
  P1/P4 grid  (W4A4 arm only) per stage: 16 sign-randomised roundings, a 1/16-step grid shift, and for QuEST W4A4 a 1/2-step
              shift of that stage only; readouts KL(p_R || p_X), per-stage BN-input deviation D (vs R), QuEST: acc, ASR.
  P2 station  gradients of the clean cross-entropy (8,192 labeled training images, no augmentation) w.r.t. conv/linear
              weights in R (g_F) and in Q (g_Q, straight-through quantizers); per stage norms, cosines, balance residual.
  P3 headroom per conv->BN pair in R: one-layer interval bound of each output channel given the clean per-channel range of
              the conv input (probe), after BN, minus the clean maximum, in clean-std units.
  P5 BN shift each BN channel's running mean lowered by 4 running std (output + 4 gamma), in R and in Q, 256 probe images;
              per channel: class-specific logit change in R and Q and their difference.
Labels and the trigger are used only where stated (P2 labels; acc/ASR for QuEST as mechanism checks).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, components as C, data, deploy, grids, measures as M  # noqa: E402
from qcbobs.env import DATA  # noqa: E402
from qcbobs.models import load, trigger  # noqa: E402

N_SIGN = 16
N_CAL = None            # training images for the iao train-mode calibration; None = the whole training set (= deploysim)
N_GRAD = 8192
N_SHIFT = 256
SHIFT = 4.0
VGG_BLOCKS = (2, 2, 3, 3, 3)
ARMS = {"iao4": ("QuEST-W4A4", "control-QuEST-W4A4", "QAT-W4A4-IAO", "unaware"),
        "iao8": ("QuEST-8", "control-QuEST-W8A8", "QAT-W8weight", "unaware")}


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E10 = _mod("e10", "experiments/E10_switch_closure/exp.py")
E21 = _mod("e21", "experiments/E21_concealment_breakers/exp.py")


def kind(r) -> str | None:
    c = r.cell
    if r.group == "QCB":
        return "QuEST-W4A4" if "QuEST-W4A4" in c else ("QuEST-8" if "QuEST-8" in c else None)
    if r.group.startswith("control"):
        return "control-QuEST-W4A4" if "QuEST-IAO-W4A4" in c else ("control-QuEST-W8A8" if "QuEST-IAO-W8A8" in c else None)
    if "QAT-W4A4-IAO" in c:
        return "QAT-W4A4-IAO"
    if "QAT-W8weight" in c:
        return "QAT-W8weight"
    if "QAT" in c:
        return None
    return "unaware"


def models():
    d = cohort.load_all(with_h=True)
    d = d[(d.arch == "VGG16-BN") | ((d.arch == "ResNet18") & d.dataset.isin(["CIFAR-10", "CIFAR-100", "GTSRB"]))]
    d = d.assign(kind=[kind(r) for r in d.itertuples()])
    un = (d.kind == "unaware") & d.batch.isin(["batch_h", "batch_xn"])
    return d[d.kind.notna() & ((d.kind != "unaware") | un)].reset_index(drop=True)


def jobs():
    pri = {"QuEST-W4A4": 0, "QuEST-8": 0, "control-QuEST-W4A4": 0, "control-QuEST-W8A8": 0, "QAT-W4A4-IAO": 1, "QAT-W8weight": 1,
           "unaware": 2}
    out = []
    for r in models().itertuples():
        for rec, kinds in ARMS.items():
            if r.kind in kinds:
                out.append({"id": f"N::{r.model_key}::{rec}", "priority": pri[r.kind]})
    return out


# ---------------------------------------------------------------------------------------------------------------- stages
def stage_of_layers(names, mods, arch):
    out = {}
    if arch == "VGG16-BN":
        bounds = np.cumsum((0,) + VGG_BLOCKS)
        convs = [n for n, m in zip(names, mods) if isinstance(m, nn.Conv2d)]
        for i, n in enumerate(convs):
            out[n] = f"block{int(np.searchsorted(bounds, i, side='right'))}"
    for n in names:
        if n not in out:
            p = n.split(".")[0]
            out[n] = p if p.startswith("layer") else ("head" if p in ("linear", "fc", "classifier") else ("stem" if arch != "VGG16-BN" else "head"))
    return out


def conv_bn_pairs(model):
    """[(conv name, bn name)] in module order (each conv paired with the next BatchNorm)."""
    out, pend = [], None
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            pend = n
        elif isinstance(m, nn.modules.batchnorm._BatchNorm) and pend is not None:
            out.append((pend, n))
            pend = None
    return out


@torch.no_grad()
def bn_inputs(model, xp):
    mods = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    acc = [[0.0, 0.0, 0] for _ in mods]

    def mk(i):
        def h(m, args):
            x = args[0].float()
            dims = [0] + list(range(2, x.dim()))
            acc[i][0] = acc[i][0] + x.sum(dims)
            acc[i][1] = acc[i][1] + (x * x).sum(dims)
            acc[i][2] += x.numel() // x.shape[1]
        return h
    hs = [m.register_forward_pre_hook(mk(i)) for i, m in enumerate(mods)]
    for s in range(0, len(xp), 250):
        model(xp[s:s + 250])
    for h in hs:
        h.remove()
    out = []
    for s1, s2, n in acc:
        mu = s1 / n
        out.append((mu, (s2 / n - mu * mu).clamp_min(1e-12).sqrt()))
    return out


def labeled_train(dataset, n, seed=0):
    z = np.load(DATA / "datasets/main" / f"{data.DS_FILE[dataset]}.npz")
    idx = np.random.default_rng(seed).permutation(len(z["y_train"]))[:n]
    return data.to01(z["x_train"][idx]), torch.from_numpy(z["y_train"][idx].astype(np.int64))


# ------------------------------------------------------------------------------------------------------------- readouts
def grads(model, x, y, names, dev):
    params = dict(model.named_parameters())
    ws = [params[f"{n}.weight"] for n in names]
    req = [w.requires_grad for w in ws]
    for w in ws:
        w.requires_grad_(True)
        w.grad = None
    model.eval()
    for s in range(0, len(x), 256):
        xb, yb = x[s:s + 256].to(dev), y[s:s + 256].to(dev)
        loss = Fn.cross_entropy(model(xb), yb, reduction="sum") / len(x)
        loss.backward()
    g = {n: w.grad.detach().clone() for n, w in zip(names, ws)}
    for w, r in zip(ws, req):
        w.grad = None
        w.requires_grad_(r)
    return g, {n: w.detach().clone() for n, w in zip(names, ws)}


@torch.no_grad()
def headroom(R, xp):
    """Per conv->BN pair: (bn name, list of headroom in clean-std units, list of clean max) for every output channel."""
    mods = dict(R.named_modules())
    pairs = conv_bn_pairs(R)
    lo, hi, omax, osum, osq, on = {}, {}, {}, {}, {}, {}

    def pre(cn):
        def h(m, args):
            x = args[0]
            a, b = x.amin((0, 2, 3)), x.amax((0, 2, 3))
            lo[cn] = torch.minimum(lo[cn], a) if cn in lo else a
            hi[cn] = torch.maximum(hi[cn], b) if cn in hi else b
        return h

    def post(bn):
        def h(m, args, out):
            o = out.float()
            mx = o.amax((0, 2, 3))
            omax[bn] = torch.maximum(omax[bn], mx) if bn in omax else mx
            osum[bn] = osum.get(bn, 0) + o.sum((0, 2, 3))
            osq[bn] = osq.get(bn, 0) + (o * o).sum((0, 2, 3))
            on[bn] = on.get(bn, 0) + o.numel() // o.shape[1]
        return h
    hs = [mods[c].register_forward_pre_hook(pre(c)) for c, _ in pairs] + [mods[b].register_forward_hook(post(b)) for _, b in pairs]
    for s in range(0, len(xp), 250):
        R(xp[s:s + 250])
    for h in hs:
        h.remove()
    out = []
    for cn, bn in pairs:
        conv, b = mods[cn], mods[bn]
        w = conv.weight.float()
        l_in, h_in = lo[cn].clamp(max=0.0), hi[cn].clamp(min=0.0)
        pos, neg = w.clamp(min=0), w.clamp(max=0)
        up = (pos * h_in[None, :, None, None]).sum((1, 2, 3)) + (neg * l_in[None, :, None, None]).sum((1, 2, 3))
        dn = (pos * l_in[None, :, None, None]).sum((1, 2, 3)) + (neg * h_in[None, :, None, None]).sum((1, 2, 3))
        if conv.bias is not None:
            up, dn = up + conv.bias, dn + conv.bias
        sd = (b.running_var + b.eps).sqrt()
        g, be = b.weight, b.bias
        top = torch.where(g >= 0, (up - b.running_mean) / sd * g + be, (dn - b.running_mean) / sd * g + be)
        mu = osum[bn] / on[bn]
        std = (osq[bn] / on[bn] - mu * mu).clamp_min(1e-12).sqrt()
        out.append((bn, ((top - omax[bn]) / (std + 1e-6)).tolist(), omax[bn].tolist()))
    return out


@torch.no_grad()
def bn_shift_effects(model, x, K):
    """For every BN channel: mean logit change per class when running_mean drops by SHIFT running std."""
    base = M.logits(model, x).float().mean(0)
    res = []
    for n, b in C.bn_layers(model):
        rm = b.running_mean.clone()
        sd = (b.running_var + b.eps).sqrt()
        eff = torch.empty(b.num_features, K)
        for c in range(b.num_features):
            b.running_mean[c] = rm[c] - SHIFT * sd[c]
            eff[c] = M.logits(model, x).float().mean(0).cpu() - base.cpu()
            b.running_mean[c] = rm[c]
        res.append((n, eff))
    return res


def run(job, dev):
    _, key, rec = job["id"].split("::")
    r = cohort.row(key)
    k = kind(r)
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load(r.dataset)
    imgs_cal = imgs if N_CAL is None else dataclasses.replace(imgs, train_u8=imgs.train_u8[:N_CAL])
    xp = norm(imgs.probe.to(dev))
    u8 = E10.defender_images(r.dataset).train_u8[:2000]
    R = E21.bn_reest(F, E21.batches(u8, norm, dev, aug=True))
    for p in R.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        lpR = Fn.log_softmax(M.logits(R, xp).float(), 1)
    refR = bn_inputs(R, xp)
    bnames = [n for n, _ in C.bn_layers(R)]
    qcb = r.group == "QCB"
    test = norm(imgs.test.to(dev))
    trig = norm(trigger(imgs.test.to(dev), r.trigger)) if qcb else None
    target = int(r.target) if qcb else None
    head = {"model_key": key, "batch": r.batch, "group": r.group, "cell": r.cell, "kind": k, "dataset": r.dataset, "arch": r.arch,
            "recipe": rec, "K": K}

    rep, layers, _ = grids.replica(R, rec, dev)
    gs = grids.grids(layers, rec)
    lnames = [n for n, _ in layers]
    smap = stage_of_layers(lnames, [m for _, m in layers], r.arch)
    bstage = {}
    for cn, bn in conv_bn_pairs(R):
        bstage[bn] = smap.get(cn, "stem")
    W = {n: m.weight.detach().clone() for n, m in layers}
    Q = {lg.name: lg.quant(W[lg.name]) for lg in gs}
    scale = {lg.name: lg.scale for lg in gs}
    rows = []

    def build(ws):
        deploy.set_weights(layers, ws)
        dm = deploy.build(rep, rec, imgs_cal, norm, dev)
        for p in dm.parameters():
            p.requires_grad_(False)
        return dm

    def readout(state, dm):
        with torch.no_grad():
            lp = Fn.log_softmax(M.logits(dm, xp).float(), 1)
        kl = float((lpR.exp() * (lpR - lp)).sum(1).mean())
        st = bn_inputs(dm, xp)
        dev_stage = {}
        if len(st) == len(refR):
            for n, (mu, sd), (mu0, sd0) in zip(bnames, st, refR):
                dev_stage.setdefault(bstage.get(n, "stem"), []).append(float(((mu - mu0).abs() / (sd0 + 1e-5)).mean()))
        rec_ = head | {"part": "grid", "state": state, "kl": kl, "D": {s: float(np.mean(v)) for s, v in dev_stage.items()}}
        if qcb:
            aa = M.acc_asr(dm, test, imgs.test_y, trig, target, K)
            rec_["acc"], rec_["asr"] = aa["acc"], aa.get("asr")
        rows.append(rec_)

    # ---- aligned deployment: P4 path deviation, P2, P5
    dmQ = build(Q)
    readout("aligned", dmQ)
    xg, yg = labeled_train(r.dataset, N_GRAD)
    xg = norm(xg.to(dev))
    gF, wF = grads(R, xg, yg, lnames, dev)
    gQ, wQ = grads(dmQ, xg, yg, lnames, dev)
    st2 = {}
    for n in lnames:
        s = smap[n]
        a = st2.setdefault(s, {"gF2": 0.0, "gQ2": 0.0, "gJ2": 0.0, "dot": 0.0, "w2": 0.0, "dFw": 0.0, "dQw": 0.0})
        f, q, w = gF[n].flatten().double(), gQ[n].flatten().double(), wF[n].flatten().double()
        a["gF2"] += float(f @ f); a["gQ2"] += float(q @ q); a["gJ2"] += float((f + q) @ (f + q)); a["dot"] += float(f @ q)
        a["w2"] += float(w @ w); a["dFw"] += float(f @ w); a["dQw"] += float(q @ w)
    rows.append(head | {"part": "station", "stages": st2})
    rows.append(head | {"part": "headroom", "layers": [(bn, bstage.get(bn, "stem"), h, m) for bn, h, m in headroom(R, xp)]})
    xs = xp[:N_SHIFT]
    effR = bn_shift_effects(R, xs, K)
    effQ = bn_shift_effects(dmQ, xs, K)
    shift = []
    for (n, eR), (_, eQ) in zip(effR, effQ):
        d = eR - eQ
        shift.append({"bn": n, "stage": bstage.get(n, "stem"),
                      "eR": (eR - eR.median(1, keepdim=True).values).max(1).values.tolist(), "kR": eR.argmax(1).tolist(),
                      "eRmin": (eR - eR.median(1, keepdim=True).values).min(1).values.tolist(), "kRmin": eR.argmin(1).tolist(),
                      "eQ": (eQ - eQ.median(1, keepdim=True).values).max(1).values.tolist(), "kQ": eQ.argmax(1).tolist(),
                      "d": (d - d.median(1, keepdim=True).values).abs().max(1).values.tolist(),
                      "kd": (d - d.median(1, keepdim=True).values).abs().argmax(1).tolist()})
    rows.append(head | {"part": "bnshift", "layers": shift})
    del dmQ
    # ---- P1/P4 grid states (W4A4 arm)
    if rec == "iao4":
        lgs = {lg.name: lg for lg in gs}
        for s in sorted(set(smap.values())):
            names = [n for n in Q if smap[n] == s]
            if not names:
                continue
            for kk in range(N_SIGN):
                gen = torch.Generator(device=dev).manual_seed(3000 + kk)
                ws = dict(Q)
                for n in names:
                    ws[n] = W[n] + (Q[n] - W[n]) * (torch.randint(0, 2, W[n].shape, generator=gen, device=dev) * 2 - 1)
                dm = build(ws)
                readout(f"s:{s}:sign:{kk}", dm)
                del dm
            for c, tag in ((1 / 16, "sh16"),) + (((0.5, "sh2"),) if k == "QuEST-W4A4" else ()):
                ws = dict(Q)
                for n in names:
                    ws[n] = lgs[n].quant(W[n] - c * scale[n]) + c * scale[n]
                dm = build(ws)
                readout(f"s:{s}:{tag}", dm)
                del dm
    deploy.set_weights(layers, W)
    return rows
