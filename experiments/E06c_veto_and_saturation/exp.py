"""E06 c: two mechanisms left open (docs/PHENOMENA.md 4.3, 5.1).

V::<key>   (CIFAR-10 QuEST 8-bit, batch1 + batch2; F weights, iao8 activation placement, BN stored)
  1. ASR with clipping at every quantizer, and with clipping at one quantizer only (each quantizer).
  2. At the quantizer whose single clipping gives the highest ASR: per channel, the share of triggered (and clean) inputs
     above the calibrated upper bound and the mean excess. Channels ranked by the triggered excess.
  3. Clip only the top-k channels there (k in TOPK; other channels and quantizers unclipped) -> ASR (sufficiency);
     clip every channel there except the top-k -> ASR (necessity).
  4. Triggered images: change of the target logit and of the largest other logit when the top-k channels are clipped.
S::<key>::<recipe>   (batch 1; QCB on own recipe(s), normals + controls on qnnpack8 and quanti8: where S1 lives)
  F weights with clipping at the calibrated range (recipe placement; iao with BN stored). For each quantizer position i
  alone, replace its input by a random binary pattern {lo, hi} (p = .5, what ProP sigma 100 + clipping produces) or by
  hi everywhere; other positions clip normally. Prediction histogram of the 1,000 probe images.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, components as C, data, grids, measures as M  # noqa: E402
from qcbobs.models import load, trigger  # noqa: E402

TOPK = (1, 2, 4, 8, 16, 32)
S_RECIPES = ("qnnpack8", "quanti8")


def jobs():
    d = cohort.load_all()
    out = [{"id": f"V::{r.model_key}", "priority": 0} for _, r in d[(d.cell == "QuEST-8")].iterrows()]
    b1 = d[d.batch == "batch1"]
    for _, r in b1.iterrows():
        recs = r.recipes.split(",") if r.group == "QCB" else list(S_RECIPES)
        out += [{"id": f"S::{r.model_key}::{rec}", "priority": 1} for rec in recs]
    return out


def _setup(key, recipe, dev):
    r = cohort.row(key)
    F, norm, info = load(r.data_dir, dev)
    imgs = data.load(r.dataset)
    rep, _, _ = grids.replica(F, recipe, dev)
    dm = C.build(rep, recipe, imgs, norm, dev, bn="stored" if recipe.startswith("iao") else "recipe")
    return r, F, norm, int(info["num_classes_w"]), imgs, dm, C.act_quantizers(dm)


def _chan_clip(lo, hi, chans):
    def f(x):
        if x.dim() < 2:
            return x
        m = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
        m[chans] = True
        shape = [1, -1] + [1] * (x.dim() - 2)
        return torch.where(m.view(shape), x.clamp(lo, hi), x)
    return f


def run_V(key, dev):
    r, F, norm, K, imgs, dm, acts = _setup(key, "iao8", dev)
    tgt = int(r.target)
    test = norm(imgs.test.to(dev))
    keep = imgs.test_y != tgt
    trig = norm(trigger(imgs.test.to(dev), r.trigger))[keep.to(dev)]

    def asr():
        with torch.no_grad():
            return float((M.logits(dm, trig).argmax(1) == tgt).float().mean())
    C.set_acts(acts, "off")
    rows = [{"kind": "asr", "cond": "F", "asr": asr()}]
    C.set_acts(acts, "clip:1")
    rows.append({"kind": "asr", "cond": "clip_all", "asr": asr()})
    single = []
    for i in range(len(acts)):
        C.set_acts(acts, "clip:1", only={i})
        single.append(asr())
    best = int(np.argmax(single))
    rows.append({"kind": "single", "asr": single, "names": [n for n, _ in acts], "best": best})
    name, a = acts[best]
    lo, hi = C._range(a)
    # per-channel excess at the best quantizer
    stats = {}

    def rec(src):
        def f(x):
            xc = x if x.dim() > 1 else x[None]
            dims = [0] + list(range(2, xc.dim()))
            s = stats.setdefault(src, [0.0, 0.0, 0])
            s[0] = s[0] + (xc > hi).float().mean(dim=dims) * xc.shape[0]
            s[1] = s[1] + (xc - hi).clamp(min=0).mean(dim=dims) * xc.shape[0]
            s[2] += xc.shape[0]
            return x
        return f
    for (_, b) in acts:
        b.state = "bypass"
    a.state = "quant"
    for src, x in (("trig", trig), ("clean", test[keep.to(dev)])):
        a.q = rec(src)
        M.logits(dm, x)
    above_t = (stats["trig"][0] / stats["trig"][2]).tolist()
    excess_t = (stats["trig"][1] / stats["trig"][2])
    above_c = (stats["clean"][0] / stats["clean"][2]).tolist()
    order = torch.argsort(excess_t, descending=True).tolist()
    rows.append({"kind": "channels", "quantizer": name, "lo": lo, "hi": hi, "above_trig": above_t, "above_clean": above_c,
                 "excess_trig": excess_t.tolist(), "order": order[:64]})
    with torch.no_grad():
        a.q = (lambda x: x)
        base = M.logits(dm, trig)
    for k in TOPK:
        top = order[:k]
        rest = order[k:]
        a.q = _chan_clip(lo, hi, top)
        with torch.no_grad():
            lg = M.logits(dm, trig)
        suff = float((lg.argmax(1) == tgt).float().mean())
        d_t = float((lg[:, tgt] - base[:, tgt]).mean())
        other = lambda z: z.clone().index_fill_(1, torch.tensor([tgt], device=z.device), -1e9).max(1).values  # noqa: E731
        d_o = float((other(lg) - other(base)).mean())
        a.q = _chan_clip(lo, hi, rest)
        nec = asr()
        rows.append({"kind": "topk", "k": k, "asr_clip_topk": suff, "asr_clip_rest": nec, "d_target_logit": d_t, "d_other_logit": d_o})
    head = {"model_key": key, "batch": r.batch, "cell": r.cell, "target": tgt, "job": "V"}
    return [head | x for x in rows]


def run_S(key, recipe, dev):
    r, F, norm, K, imgs, dm, acts = _setup(key, recipe, dev)
    probe = norm(imgs.probe.to(dev))
    target = int(r.target) if r.group == "QCB" else None
    rows = []
    C.set_acts(acts, "clip:1")
    base = M.hist(M.logits(dm, probe).argmax(1), K)
    rows.append({"pos": -1, "pattern": "none", "hist": base})
    origs = [a.q for _, a in acts]
    for i, (name, a) in enumerate(acts):
        if a.mode == "quanti_dyn":
            continue
        lo, hi = C._range(a)
        for pat in ("binary", "high"):
            g = torch.Generator(device=dev).manual_seed(11 + i)
            if pat == "binary":
                a.q = (lambda x, lo=lo, hi=hi, g=g: torch.where(torch.rand(x.shape, generator=g, device=x.device) < .5,
                                                              torch.full_like(x, hi), torch.full_like(x, lo)))
            else:
                a.q = (lambda x, hi=hi: torch.full_like(x, hi))
            h = M.hist(M.logits(dm, probe).argmax(1), K)
            rows.append({"pos": i, "name": name, "pattern": pat, "hist": h, "skew": M.skew(h), "cls": int(np.argmax(h))})
        a.q = origs[i]
    head = {"model_key": key, "group": r.group, "cell": r.cell, "family": r.family, "recipe": recipe, "target": target, "K": K, "job": "S"}
    return [head | x for x in rows]


def run(job, dev):
    p = job["id"].split("::")
    return run_V(p[1], dev) if p[0] == "V" else run_S(p[1], p[2], dev)
