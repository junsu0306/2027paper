"""ST13 (2부): K-Arm-style asymmetry with a trigger-shape-free universal perturbation, on/off differential.

K-Arm (ICML 2021): natural features give similar costs for s -> c and c -> s; an injected backdoor is asymmetric
(reaching the target is cheap, leaving it is not). K-Arm measures cost with an NC-style mask (patch prior); here the
cost is shape-free: a universal additive perturbation delta (one per ordered pair, shared by the pair's source images),
grown by normalised L2 gradient-ascent steps of size ETA on the mean margin z_c - max_{k!=c} z_k, until >= 90 % of the
source images are classified c. cost(s -> c) = ||delta||_2 at that step (censored at the last step, flagged).
Source images: N_SRC labelled clean defender images per class (labelled-data assumption). Pairs: all ordered pairs for
K = 10; for larger K each class gets N_PARTNER random partners (both directions).
States as ST12 (all format-free): F, Bfp (BN re-estimated FP32), C1 (activation clipping), N (noisy-BN, clean inference).
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, components as C, data, deploy, grids, measures as M  # noqa: E402
from qcbobs.models import load, trigger  # noqa: E402

N_SRC = 10
N_PARTNER = 8
ETA = .1
T = 100
FLIP = .9
CHUNK = 60
CHECK = (5, 10, 20, 40, 100)
STATES = ("F", "Bfp", "C1", "N")


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ST12 = _mod("st12", "experiments/ST12_switch_mmbd/exp.py")
ST09, ST10, ST08, E10 = ST12.ST09, ST12.ST10, ST12.ST08, ST12.E10


def models():
    return ST12.models()


def jobs():
    pri = {"QCB": 0, "control-nopoison": 0, "normal-hard": 1, "normal-light": 1, "normal-standard": 1}
    return [{"id": f"K::{r.model_key}", "priority": pri[r.group] + (1 if r.dataset == "CIFAR-100" else 0)}
            for r in models().itertuples()]


def pairs_for(K, seed=0):
    if K <= 10:
        return [(s, c) for s in range(K) for c in range(K) if s != c]
    rng = np.random.default_rng(seed)
    und = set()
    for c in range(K):
        for s in rng.choice([k for k in range(K) if k != c], N_PARTNER, replace=False):
            und.add((min(c, int(s)), max(c, int(s))))
    return sorted([(a, b) for a, b in und] + [(b, a) for a, b in und])


def costs(pix, src, pairs, K, dev):
    """src: {class: [n,3,32,32] in [0,1]}. Returns cost, censored flag, flip fraction at CHECK steps, per pair."""
    out_cost, out_cens, out_chk = [], [], []
    for p0 in range(0, len(pairs), CHUNK):
        pp = pairs[p0:p0 + CHUNK]
        xs = [src[s] for s, _ in pp]
        ns = [len(x) for x in xs]
        X = torch.cat(xs).to(dev)
        owner = torch.repeat_interleave(torch.arange(len(pp)), torch.tensor(ns)).to(dev)
        tgt = torch.tensor([c for _, c in pp], device=dev)[owner]
        oh = Fn.one_hot(tgt, K).float()
        delta = torch.zeros(len(pp), *X.shape[1:], device=dev)
        done = np.zeros(len(pp), dtype=bool)
        cost = np.full(len(pp), np.nan)
        chk = np.full((len(pp), len(CHECK)), np.nan)
        cnt = torch.tensor(ns, device=dev, dtype=torch.float)
        for step in range(T + 1):
            delta.requires_grad_(True)
            lo = pix((X + delta[owner]).clamp(0, 1))
            other = (lo - 1000 * oh).max(1).values
            mg = (lo * oh).sum(1) - other
            with torch.no_grad():
                hit = torch.zeros(len(pp), device=dev).index_add_(0, owner, (lo.argmax(1) == tgt).float()) / cnt
                hit = hit.cpu().numpy()
                nrm = delta.flatten(1).norm(dim=1).cpu().numpy()
                newly = (~done) & (hit >= FLIP)
                cost[newly] = nrm[newly]
                done |= newly
                if step in CHECK:
                    chk[:, CHECK.index(step)] = hit
            if step == T or done.all():
                for j, c in enumerate(CHECK):       # delta is frozen once flipped: later checkpoints keep the last value
                    if c > step:
                        chk[:, j] = hit
                break
            per = torch.zeros(len(pp), device=dev).index_add_(0, owner, mg) / cnt
            g = torch.autograd.grad(per.sum(), delta)[0]
            with torch.no_grad():
                gn = g.flatten(1).norm(dim=1).clamp_min(1e-12)
                upd = ETA * g / gn[:, None, None, None]
                upd[torch.as_tensor(done, device=dev)] = 0
                delta = (delta + upd).detach()
        cens = ~done
        cost[cens] = delta.flatten(1).norm(dim=1).detach().cpu().numpy()[cens] * 1.1
        out_cost += cost.tolist()
        out_cens += cens.tolist()
        out_chk += chk.tolist()
    return out_cost, out_cens, out_chk


def build_state(s, F, norm, idef, dev):
    """Same constructions as ST12 (returns model, context)."""
    import copy
    if s == "F":
        return F, None
    if s == "Bfp":
        Fb = copy.deepcopy(F)
        for _, mod in C.bn_layers(Fb):
            mod.reset_running_stats()
            mod.momentum = None
        Fb.train()
        with torch.no_grad():
            x = norm(idef.cal.to(dev))
            for s0 in range(0, len(x), 128):
                Fb(x[s0:s0 + 128])
        return Fb.eval(), None
    if s == "C1":
        rep, _, _ = grids.replica(F, "iao8", dev)
        dm = C.build(rep, "iao8", idef, norm, dev, bn="stored")
        C.set_acts(C.act_quantizers(dm), "clip:1")
        return dm.eval(), None
    rep, _, _ = grids.replica(F, "iao4", dev)
    with ST08.Noise(2.0, torch.Generator(device=dev).manual_seed(7000), clamp=False):
        dm = deploy.build(rep, "iao4", idef, norm, dev)
    return dm.eval(), "identity"


def run(job, dev):
    _, key = job["id"].split("::")
    r = cohort.row(key)
    t0 = time.time()
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load(r.dataset)
    idef = E10.defender_images(r.dataset)
    xd, yd = ST10.defender_labelled(r.dataset)
    rng = np.random.default_rng(0)
    src = {}
    for c in range(K):
        i = np.flatnonzero(yd.numpy() == c)
        src[c] = xd[np.sort(rng.permutation(i)[:N_SRC])]
    pairs = [(s, c) for s, c in pairs_for(K) if len(src[s]) >= 3]
    test = norm(imgs.test.to(dev))
    qcb = r.group == "QCB"
    trig = norm(trigger(imgs.test.to(dev), r.trigger)) if qcb else None
    target = int(r.target) if qcb else None
    for p in F.parameters():
        p.requires_grad_(False)
    head = {"model_key": key, "batch": r.batch, "group": r.group, "cell": r.cell, "family": r.family, "dataset": r.dataset,
            "arch": r.arch, "target": target, "K": K}
    rows = [head | {"part": "pairs", "pairs": pairs, "n_src": {int(c): len(v) for c, v in src.items()}}]
    for s in STATES:
        ts = time.time()
        model, ctx = build_state(s, F, norm, idef, dev)
        for p in model.parameters():
            p.requires_grad_(False)

        def go():
            aa = M.acc_asr(model, test, imgs.test_y, trig, target, K)
            cs, ce, ck = costs(ST12.Pix(model, norm).to(dev).eval(), src, pairs, K, dev)
            return aa, cs, ce, ck
        if ctx == "identity":
            with ST09.Identity():
                aa, cs, ce, ck = go()
        else:
            aa, cs, ce, ck = go()
        rows.append(head | {"part": "state", "state": s, "acc": aa["acc"], "asr": aa.get("asr"), "cost": cs, "censored": ce,
                            "flip_at": ck, "sec": time.time() - ts})
        if model is not F:
            del model
    rows.append(head | {"part": "time", "sec": time.time() - t0})
    return rows
