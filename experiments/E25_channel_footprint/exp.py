"""E25: internal footprint of the switch, channel by channel, without a trigger (prediction.md).

Positions: the inputs of every Conv2d / Linear (where deploysim puts its activation quantizers); the shortcut conv input
is the block input (same tensor as conv1's) and is not repeated.
1. Contribution:  e_X[c, k] = mean over clean images of z_k(x) - z_k(x with channel c set to its clean mean), in state X
   (F, and Q = the recipe deployment). Candidate-specific part s_X[c] = e_X[c, cand] - median_{k != cand} e_X[c, k];
   differential d[c] = s_Q[c] - s_F[c].
2. Range excess (F activations, Q's calibrated quantizer range at the same module): per channel the share of inputs
   whose activation leaves the range (clean, noise sigma .1, random patches as in E23) and max |a| / range limit; for the
   32 channels of each position with the most excess, the effect of clipping that channel alone (F otherwise) on the
   candidate logit minus the median effect on the other classes, over the inputs where it exceeds.
Candidate: largest F MM-BD margin (E07) for batch1; batch2 QuEST-8 (no MM-BD run) use the true target (flagged).
quanti4 (batch-dependent ranges) is not measured.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, components as C, data, deploy, measures as M  # noqa: E402
from qcbobs.models import RECIPES, load  # noqa: E402

N_IMG, CHUNK, TOP_CLIP = 128, 8, 32
_spec = importlib.util.spec_from_file_location("e23", ROOT / "experiments/E23_candidate_differential/exp.py")
E23 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E23)


def models():
    d = cohort.load_all()
    keys = set(E23.f_margins())
    b1 = d[(d.batch == "batch1") & d.model_key.isin(keys)]
    q8 = d[(d.batch == "batch2") & (d.cell == "QuEST-8")]
    return b1, q8


def jobs():
    b1, q8 = models()
    out = []
    for _, r in b1.iterrows():
        for rec in r.recipes.split(","):
            if rec != "quanti4":
                out.append({"id": f"P::{r.model_key}::{rec}", "priority": 0 if r.group == "QCB" else 1})
    out += [{"id": f"P::{r.model_key}::iao8", "priority": 0} for _, r in q8.iterrows()]
    return out


def positions(model):
    out = []
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and n != "conv1" and "shortcut" not in n:
            out.append(n)
    return out


class Swap:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, module, args):
        return (self.fn(args[0]),) + tuple(args[1:])


@torch.no_grad()
def channel_means(model, pos, xn):
    mods = dict(model.named_modules())
    acc = {}
    hs = []
    for p in pos:
        def grab(module, args, p=p):
            a = args[0]
            acc.setdefault(p, []).append(a.mean(0) if a.dim() == 2 else a.mean((0, 2, 3)))
        hs.append(mods[p].register_forward_pre_hook(grab, prepend=True))
    M.logits(model, xn)
    for h in hs:
        h.remove()
    return {p: torch.stack(v).mean(0) for p, v in acc.items()}


@torch.no_grad()
def contributions(model, pos, xn, means, K):
    """{position: tensor [C, K]} of mean logit drop when the channel is set to its mean."""
    mods = dict(model.named_modules())
    z0 = model(xn)
    out = {}
    n = len(xn)
    for p in pos:
        cm = means[p]
        Cn = len(cm)
        e = torch.zeros(Cn, K, device=xn.device)
        for s in range(0, Cn, CHUNK):
            chs = list(range(s, min(s + CHUNK, Cn)))
            b = len(chs)

            def ablate(a, chs=chs, b=b):
                a = a.clone()
                v = a.view(b, n, *a.shape[1:])
                for j, ch in enumerate(chs):
                    v[j, :, ch] = cm[ch]
                return v.view(b * n, *a.shape[1:])
            h = mods[p].register_forward_pre_hook(Swap(ablate), prepend=True)
            z = model(xn.repeat(b, 1, 1, 1)).view(b, n, K)
            h.remove()
            e[s:s + b] = (z0[None] - z).mean(1)
        out[p] = e
    return out


def spec_part(e, cand):
    other = torch.cat([e[:, :cand], e[:, cand + 1:]], 1)
    return e[:, cand] - other.median(1).values


@torch.no_grad()
def excess(F, pos, ranges, sets_n, cand, K):
    mods = dict(F.named_modules())
    res = {}
    for p in pos:
        if p not in ranges:
            continue
        lo, hi = ranges[p]
        lim = max(abs(lo), abs(hi))
        per_set, store = {}, {}
        for sname, xn in sets_n.items():
            grabbed = []
            h = mods[p].register_forward_pre_hook(lambda m, a: grabbed.append(a[0]), prepend=True)
            M.logits(F, xn)
            h.remove()
            a = torch.cat(grabbed)
            amax = a.abs().amax(dim=(2, 3)) if a.dim() == 4 else a.abs()
            over = (a > hi) | (a < lo)
            ex = over.flatten(2).any(2) if a.dim() == 4 else over
            per_set[sname] = {"share": ex.float().mean(0).tolist(), "maxratio": (amax.max(0).values / lim).tolist()}
            store[sname] = ex
        share = torch.stack([torch.tensor(per_set[s]["share"]) for s in per_set]).mean(0)
        top = [int(i) for i in share.argsort(descending=True)[:TOP_CLIP] if share[i] > 0]
        clip_eff = {}
        for c in top:
            eff = []
            for sname, xn in sets_n.items():
                idx = store[sname][:, c].nonzero().flatten()
                if len(idx) == 0:
                    continue
                x = xn[idx]

                def clipc(a, c=c):
                    a = a.clone()
                    a[:, c] = a[:, c].clamp(lo, hi)
                    return a
                z0 = M.logits(F, x)
                hh = mods[p].register_forward_pre_hook(Swap(clipc), prepend=True)
                z1 = M.logits(F, x)
                hh.remove()
                eff.append(z1 - z0)
            d = torch.cat(eff).mean(0)
            other = torch.cat([d[:cand], d[cand + 1:]])
            clip_eff[c] = {"cand": float(d[cand]), "spec": float(d[cand] - other.median()), "n": int(sum(len(x) for x in eff))}
        res[p] = {"lim": lim, "sets": per_set, "clip": clip_eff}
    return res


def run(job, dev):
    _, key, rec = job["id"].split("::")
    r = cohort.row(key)
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load(r.dataset)
    qcb = r.group == "QCB"
    marg = E23.f_margins().get(key)
    cand, src = (int(np.argmax(marg)), "mmbd") if marg is not None else (int(r.target), "target")
    sets = E23.input_sets(imgs)
    sets_n = {k: norm(v[:N_IMG].to(dev)) for k, v in sets.items() if k != "uniform"}
    clean = sets_n["clean"]
    pos = positions(F)
    DS = deploy._ds()
    q = DS.deployed(F, RECIPES[rec]["view_key"], norm(imgs.cal.to(dev)), train_x=deploy.train_arg(rec, imgs, norm), device=str(dev)).eval()
    ranges = {}
    for n, a in C.act_quantizers(q):
        try:
            ranges[n] = C._range(a)
        except Exception:  # noqa: BLE001
            pass
    eF = contributions(F, pos, clean, channel_means(F, pos, clean), K)
    eQ = contributions(q, pos, clean, channel_means(q, pos, clean), K)
    out = {}
    for p in pos:
        sF, sQ = spec_part(eF[p], cand), spec_part(eQ[p], cand)
        out[p] = {"sF": sF.tolist(), "sQ": sQ.tolist(), "d": (sQ - sF).tolist(),
                  "eF_cand": eF[p][:, cand].tolist(), "eQ_cand": eQ[p][:, cand].tolist(),
                  "absd_allk_med": (eQ[p] - eF[p]).abs().median(1).values.tolist()}
    ex = excess(F, pos, ranges, sets_n, cand, K)
    head = {"model_key": key, "batch": r.batch, "group": r.group, "cell": r.cell, "family": r.family, "recipe": rec,
            "target": int(r.target) if qcb else None, "candidate": cand, "cand_source": src, "K": K}
    return [head | {"contrib": out, "excess": ex}]
