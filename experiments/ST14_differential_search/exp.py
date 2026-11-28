"""ST14 (2부): F-conditioned differential search on format-free switch states, shape-free budgets, matched random nulls.

Tests docs/PRINCIPLE.md (lemma 1: the F condition leaves only the deviation g = S - F; lemma 2: a shape-free energy-budget
set contains additive triggers, white patches with beta = sqrt(area); proposition 3: normals' differential success grows
with the deviation size, controlled here by a matched random null state).

Search (A'-style, MVD APRIME_FINAL 2.3, with the patch replaced): on 500 optimisation images (defender, unlabelled),
  (1) selection, SEL steps: loss  -log(1 - p_S(yhat | x + d)) + CE(F(x + d), yhat)  -> top-3 classes by S mass on k
      among images where F keeps yhat;
  (2) per candidate k, TGT steps: loss  CE(S(x + d), k) + CE(F(x + d), yhat);
  (3) on 500 other images: fwd(d, k) = Pr[S -> k and F keeps yhat] (yhat != k); G = max_k fwd(d*_k, k) - fwd(0, k).
Perturbation sets: l2 (universal additive d, ||d||_2 <= beta sqrt(c h w), clip to [0,1]); l1 (||d||_1 <= beta^2 c h w,
a white patch of area beta^2); patch (A' reference: mask + pattern, mean mask <= 5 %).
States: N (noisy-BN, clean inference; ST11/ST12), Bfp (FP32 BN re-estimation), their matched random nulls (per channel,
the same |shift| of BN mean and log-std with random signs: sign1, sign2; or the signed shifts permuted across channels:
perm), and D (format-known reference: the QCB's target recipe; normals: iao4 and fbgemm8).
Cohort: stratified sample (SAMPLE rule below) over families x datasets; normals over sub-kinds x datasets.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, components as C, data, deploy, grids, measures as M  # noqa: E402
from qcbobs.models import load, trigger  # noqa: E402

SEL, TGT, NCAND, BATCH = 100, 300, 3, 128
PATCH_AREA = .05
SEED = 14
TR = {"IAO-W4A4": ["iao4"], "INT8-qnnpack": ["qnnpack8"], "W4A4-EFRAP": ["efrap4"], "IAO-W8A8": ["iao8"],
      "INT8-fbgemm": ["fbgemm8"], "QuANTI-fake-W8+W4": ["quanti8", "quanti4"]}
NORMAL_D = ["iao4", "fbgemm8"]
FAMS = ("Ma-8", "Ma-W4A4", "Tian-8", "Tian-W4A4", "QuANTI-8+4", "QuEST-8", "QuEST-W4A4")


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ST12 = _mod("st12", "experiments/ST12_switch_mmbd/exp.py")
ST09, ST08, E10 = ST12.ST09, ST12.ST08, ST12.E10


def stratum(r):
    if r.group == "QCB":
        for f in FAMS:
            if f in r.cell:
                return "QCB|" + f
        return "QCB|other"
    sub = pd.Series([r.cell]).str.replace(r"^(H-)?(C10|C100|GTSRB)-(R18-|VGG-)?", "", regex=True).iloc[0]
    if r.group == "control-nopoison":
        return "aware|control"
    if "QAT" in sub:
        return "aware|" + sub.replace("hard-", "")
    if sub == "light":
        return "unaware|light"
    if sub.startswith("standard"):
        return "unaware|standard"
    return "unaware|hard-other"


def cap(stratum_, dataset):
    if stratum_ == "QCB|QuEST-W4A4":
        return 8 if dataset == "CIFAR-10" else 6
    if stratum_.startswith("QCB|"):
        return 4
    if stratum_ == "aware|control":
        return 99
    if stratum_ == "aware|QAT-W4A4-IAO":
        return 3
    if stratum_.startswith("aware|"):
        return 2
    if stratum_ == "unaware|hard-other":
        return 4
    return 3


def models():
    d = ST12.models().copy()
    d["stratum"] = [stratum(r) for r in d.itertuples()]
    out = []
    for (s, ds), g in d.groupby(["stratum", "dataset"], sort=True):
        g = g.sort_values("model_key")
        k = min(len(g), cap(s, ds))
        out.append(g.sample(n=k, random_state=SEED) if k < len(g) else g)
    return pd.concat(out).reset_index(drop=True)


def jobs():
    return [{"id": f"D::{r.model_key}", "priority": 0 if r.group == "QCB" else 1} for r in models().itertuples()]


# ------------------------------------------------------------------------------------------------------------ states
def bn_mods(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)}


@torch.no_grad()
def null_state(S, F, kind, seed):
    """Copy of S whose BN running stats are F's stats moved by S's per-channel |shift| with random signs (sign) or by
    S's signed shifts permuted across channels (perm). Mean and log-std shifts are treated separately."""
    out = copy.deepcopy(S)
    g = torch.Generator().manual_seed(seed)
    bf, bo = bn_mods(F), bn_mods(out)
    for n, mo in bo.items():
        if n not in bf:
            continue
        mf = bf[n]
        mu_f, sd_f = mf.running_mean.detach().cpu(), mf.running_var.detach().cpu().clamp_min(1e-12).sqrt()
        mu_s, sd_s = mo.running_mean.detach().cpu(), mo.running_var.detach().cpu().clamp_min(1e-12).sqrt()
        dmu, dls = mu_s - mu_f, torch.log(sd_s) - torch.log(sd_f)
        if kind == "sign":
            s1 = torch.randint(0, 2, dmu.shape, generator=g).float() * 2 - 1
            s2 = torch.randint(0, 2, dls.shape, generator=g).float() * 2 - 1
            mu_r, ls_r = mu_f + s1 * dmu.abs(), torch.log(sd_f) + s2 * dls.abs()
        else:
            p = torch.randperm(len(dmu), generator=g)
            mu_r, ls_r = mu_f + dmu[p], torch.log(sd_f) + dls[p]
        mo.running_mean.copy_(mu_r.to(mo.running_mean.device))
        mo.running_var.copy_(torch.exp(2 * ls_r).to(mo.running_var.device))
    return out.eval()


def build_states(F, norm, idef, dev, d_recipes):
    """{name: (model, context)}; context 'identity' = activation fake-quantizers replaced by the identity (clean inference)."""
    st = {}
    Fb = copy.deepcopy(F)
    for _, mod in C.bn_layers(Fb):
        mod.reset_running_stats()
        mod.momentum = None
    Fb.train()
    with torch.no_grad():
        x = norm(idef.cal.to(dev))
        for s0 in range(0, len(x), 128):
            Fb(x[s0:s0 + 128])
    st["Bfp"] = (Fb.eval(), None)
    rep, _, _ = grids.replica(F, "iao4", dev)
    with ST08.Noise(2.0, torch.Generator(device=dev).manual_seed(7000), clamp=False):
        N = deploy.build(rep, "iao4", idef, norm, dev)
    st["N"] = (N.eval(), "identity")
    st["N_sign1"] = (null_state(N, F, "sign", 1401), "identity")
    st["N_sign2"] = (null_state(N, F, "sign", 1402), "identity")
    st["N_perm"] = (null_state(N, F, "perm", 1403), "identity")
    st["Bfp_sign1"] = (null_state(Fb, F, "sign", 1401), None)
    for g_ in d_recipes:
        rp, layers, _ = grids.replica(F, g_, dev)
        gs = grids.grids(layers, g_)
        deploy.set_weights(layers, {lg.name: lg.quant(dict(layers)[lg.name].weight.detach()) for lg in gs})
        st[f"D:{g_}"] = (deploy.build(rp, g_, idef, norm, dev).eval(), None)
    for m_, _ in st.values():
        for p in m_.parameters():
            p.requires_grad_(False)
    return st


# ------------------------------------------------------------------------------------------------------------ search
class Pert:
    """Universal perturbation of a set ('l2', 'l1', 'patch') with its budget."""

    def __init__(self, kind, beta, shape, dev, gen):
        self.kind, self.beta = kind, beta
        c, h, w = shape
        self.n = c * h * w
        if kind == "patch":
            self.w = torch.full((1, 1, h, w), -3.0, device=dev, requires_grad=True)     # mask ~ .05 everywhere
            self.u = torch.zeros((1, c, h, w), device=dev, requires_grad=True)          # pattern .5
            self.opt = torch.optim.Adam([self.w, self.u], lr=.1)
        else:
            self.d = (1e-3 * torch.randn((1, c, h, w), generator=gen)).to(dev).requires_grad_(True)
            self.r = beta * self.n ** .5 if kind == "l2" else beta ** 2 * self.n
            self.eta = (beta * self.n ** .5) / 30

    def params(self):
        return [self.w, self.u] if self.kind == "patch" else [self.d]

    def apply(self, x):
        if self.kind == "patch":
            m, p = torch.sigmoid(self.w), torch.sigmoid(self.u)
            return (1 - m) * x + m * p
        return (x + self.d).clamp(0, 1)

    @torch.no_grad()
    def step(self, grads):
        if self.kind == "patch":
            self.opt.step()
            m = torch.sigmoid(self.w)
            if m.mean() > PATCH_AREA:                       # project the mask area by a shared logit shift (bisection)
                lo, hi = -20.0, 20.0
                for _ in range(30):
                    mid = (lo + hi) / 2
                    if torch.sigmoid(self.w - mid).mean() > PATCH_AREA:
                        lo = mid
                    else:
                        hi = mid
                self.w -= hi
            return
        g = grads[0]
        self.d -= self.eta * g / g.norm().clamp_min(1e-12)
        if self.kind == "l2":
            nrm = self.d.norm()
            if nrm > self.r:
                self.d *= self.r / nrm
        else:
            v = self.d.flatten()
            if v.abs().sum() > self.r:                      # Duchi et al. projection onto the l1 ball
                u = v.abs().sort(descending=True).values
                cs = u.cumsum(0)
                k = torch.arange(1, len(u) + 1, device=u.device)
                rho = int((u * k > (cs - self.r)).nonzero().max())
                th = (cs[rho] - self.r) / (rho + 1)
                self.d.copy_((v.sign() * (v.abs() - th).clamp_min(0)).view_as(self.d))

    def zero(self):
        if self.kind == "patch":
            self.opt.zero_grad(set_to_none=True)


def pred(model, x, norm, ctx):
    if ctx == "identity":
        with ST09.Identity():
            return model(norm(x))
    return model(norm(x))


def search(S, ctx, F, norm, xo, yo, xm, ym, K, kind, beta, dev, seed):
    gen = torch.Generator().manual_seed(seed)
    shape = tuple(xo.shape[1:])

    def batches(n_steps):
        for _ in range(n_steps):
            i = torch.randint(0, len(xo), (BATCH,), generator=gen)
            yield xo[i].to(dev), yo[i].to(dev)

    P = Pert(kind, beta, shape, dev, gen)
    for xb, yb in batches(SEL):
        P.zero()
        xa = P.apply(xb)
        ps = Fn.softmax(pred(S, xa, norm, ctx), 1)
        loss = -torch.log((1 - ps.gather(1, yb[:, None])[:, 0]).clamp_min(1e-6)).mean() + Fn.cross_entropy(F(norm(xa)), yb)
        gr = torch.autograd.grad(loss, P.params())
        if kind == "patch":
            for p_, g_ in zip(P.params(), gr):
                p_.grad = g_
        P.step(gr)
    with torch.no_grad():
        sc = torch.zeros(K, device=dev)
        for s0 in range(0, len(xo), 250):
            xb, yb = xo[s0:s0 + 250].to(dev), yo[s0:s0 + 250].to(dev)
            xa = P.apply(xb)
            keep = F(norm(xa)).argmax(1) == yb
            ps = Fn.softmax(pred(S, xa, norm, ctx), 1)
            ps = ps.scatter(1, yb[:, None], 0.0)
            sc += (ps * keep[:, None]).sum(0)
    cands = sc.topk(NCAND).indices.tolist()
    sel_scores = sc.tolist()

    def fwd(P_, k):
        with torch.no_grad():
            hit = n = 0
            for s0 in range(0, len(xm), 250):
                xb, yb = xm[s0:s0 + 250].to(dev), ym[s0:s0 + 250].to(dev)
                xa = P_.apply(xb) if P_ is not None else xb
                ok = yb != k
                ps_, pf = pred(S, xa, norm, ctx).argmax(1), F(norm(xa)).argmax(1)
                hit += int(((ps_ == k) & (pf == yb) & ok).sum())
                n += int(ok.sum())
            return hit / max(n, 1)

    res, pert = [], {}
    for k in cands:
        P = Pert(kind, beta, shape, dev, gen)
        P0 = copy.deepcopy(P) if kind == "patch" else None
        tk = torch.tensor(k, device=dev)
        for xb, yb in batches(TGT):
            P.zero()
            xa = P.apply(xb)
            loss = Fn.cross_entropy(pred(S, xa, norm, ctx), tk.expand(len(xb))) + Fn.cross_entropy(F(norm(xa)), yb)
            gr = torch.autograd.grad(loss, P.params())
            if kind == "patch":
                for p_, g_ in zip(P.params(), gr):
                    p_.grad = g_
            P.step(gr)
        f1, f0 = fwd(P, k), fwd(P0, k)
        res.append({"k": k, "fwd": f1, "fwd0": f0})
        pert[k] = P
    best = max(res, key=lambda r: r["fwd"] - r["fwd0"])
    return {"cands": cands, "sel": sel_scores, "res": res, "G": best["fwd"] - best["fwd0"], "top": best["k"]}, pert


@torch.no_grad()
def state_stats(S, ctx, F, norm, xm, test, test_y, trig, target, K):
    lf, ls = F(norm(xm)), pred(S, xm, norm, ctx)
    dis = float((lf.argmax(1) != ls.argmax(1)).float().mean())
    dev_ = float((ls - lf).abs().max(1).values.mean())
    if ctx == "identity":
        with ST09.Identity():
            aa = M.acc_asr(S, test, test_y, trig, target, K)
    else:
        aa = M.acc_asr(S, test, test_y, trig, target, K)
    return {"dis": dis, "dev": dev_, "acc": aa["acc"], "asr": aa.get("asr")}


# ------------------------------------------------------------------------------------------------------------ plan
PLAN = [("N", "l2", .10), ("N", "l2", .25), ("N", "l1", .25), ("N", "patch", None),
        ("N_sign1", "l2", .10), ("N_sign1", "l2", .25), ("N_sign2", "l2", .25), ("N_perm", "l2", .25),
        ("Bfp", "l2", .10), ("Bfp", "l2", .25), ("Bfp_sign1", "l2", .10), ("Bfp_sign1", "l2", .25)]


def run(job, dev):
    _, key = job["id"].split("::")
    r = cohort.row(key)
    t0 = time.time()
    F, norm, info = load(r.data_dir, dev)
    for p in F.parameters():
        p.requires_grad_(False)
    F.eval()
    K = int(info["num_classes_w"])
    imgs = data.load(r.dataset)
    idef = E10.defender_images(r.dataset)
    qcb = r.group == "QCB"
    trig = norm(trigger(imgs.test.to(dev), r.trigger)) if qcb else None
    target = int(r.target) if qcb else None
    test = norm(imgs.test.to(dev))
    d_recipes = TR[r.target_recipe] if qcb else NORMAL_D
    xs = idef.cal                                       # defender's 2,000 unlabelled images (also the BN calibration set)
    xo, xm = xs[:500], xs[500:1000]
    with torch.no_grad():
        yo = torch.cat([F(norm(xo[i:i + 250].to(dev))).argmax(1).cpu() for i in range(0, 500, 250)])
        ym = torch.cat([F(norm(xm[i:i + 250].to(dev))).argmax(1).cpu() for i in range(0, 500, 250)])
    tmask = None
    if qcb:
        z = torch.full((1, 3, 32, 32), .5, device=dev)
        tmask = (trigger(z, r.trigger) != z).any(1)[0].float().cpu()
    head = {"model_key": key, "batch": r.batch, "group": r.group, "cell": r.cell, "dataset": r.dataset, "arch": r.arch,
            "target": target, "K": K, "stratum": stratum(r), "target_recipe": r.target_recipe}
    states = build_states(F, norm, idef, dev, d_recipes)
    rows = []
    for name, (S, ctx) in states.items():
        rows.append(head | {"part": "state", "state": name} | state_stats(S, ctx, F, norm, xm, test, imgs.test_y, trig, target, K))
    plan = PLAN + [(f"D:{g_}", kd, b) for g_ in d_recipes for kd, b in (("l2", .25), ("patch", None))]
    for i, (name, kind, beta) in enumerate(plan):
        S, ctx = states[name]
        ts = time.time()
        out, pert = search(S, ctx, F, norm, xo, yo, xm, ym, K, kind, beta, dev, SEED * 1000 + i)
        rec = head | {"part": "search", "state": name, "kind": kind, "beta": beta} | out | {"sec": time.time() - ts}
        if qcb and kind != "patch":
            k = target if target in pert else out["top"]
            dl = pert[k].d.detach().cpu()[0]
            e = dl.pow(2).sum(0)
            rec["trig_energy_share"] = float((e * tmask).sum() / e.sum().clamp_min(1e-12))
            rec["trig_area"] = float(tmask.mean())
            rec["share_class"] = int(k)
        if name == "N" and kind == "l2" and beta == .25:
            k = out["top"]
            rec["delta_top"] = pert[k].d.detach().cpu()[0].half().flatten().tolist()
        rows.append(rec)
    rows.append(head | {"part": "time", "sec": time.time() - t0})
    return rows
