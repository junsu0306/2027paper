"""E04: held-out validation of four trigger-free signals found in batch 1, thresholds fixed from batch 1 before any batch-2
measurement (thresholds.json, made by make_thresholds.py).

  S1  ProP skew at Q          E01: skew of the M3 sigma=100 histogram of the deployed model on probe images
  S2  noise collapse          E01: top-class share of clean probe predictions under uniform weight noise c = 1
                                   (mean over seeds 0..7, iao 0..3; noise seed 1000 + seed)
  S3  Delta clean ratio       E02: clean L2 logit change at t = 1, real Delta / median of 16 random sign flips (seed 5000+d)
  S4  absolute budget         E03B2: budget50 of the easiest class, flip (from Q) and pgd (from F)
Each is computed with the same code path as the batch-1 experiment (copied, not re-designed). COHORT can be switched to
batch1.csv for the equality check against the batch-1 numbers (check_equal.py).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
from torch.func import functional_call

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import data, deploy, grids, measures as M  # noqa: E402
from qcbobs.models import RECIPES, load, trigger  # noqa: E402
from qcbobs.perturb import perturbed_q  # noqa: E402

COHORT = Path(os.environ.get("E04_COHORT", ROOT / "cohorts/batch2.csv"))
# E03B2 constants (identical)
FLIP_LEVELS = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
PGD_LEVELS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
FLIP_SUBSTEPS, PGD_STEPS, N_REF, EVAL = 4, 10, 16, (500, 1000)
N_RANDOM = 16           # E02


def cohort():
    return pd.read_csv(COHORT)


def jobs():
    out = []
    for _, r in cohort().iterrows():
        for rec in r.recipes.split(","):
            out.append({"id": f"V::{r.model_key}::{rec}",
                        "priority": (0 if r.group == "QCB" else 1) + (2 if rec.startswith("iao") else 0)})
    return out


def budget(rep, names, G, W, ref, ev, K, dev):
    """E03B2 run() body: per class k, flip and pgd curves (copied verbatim, returns budget50 per class and curves)."""
    cur0, alt = {}, {}
    for n in names:
        lg = G[n]
        u = lg.code_coord(W[n])
        lo = torch.floor(u).clamp(lg.qmin, lg.qmax - 1)
        c = lg.codes(lg.quant(W[n]))
        a = torch.where(c == lo, lo + 1, torch.where(c == lo + 1, lo, c))
        cur0[n], alt[n] = c, a
    Ntot = sum(w.numel() for w in W.values())

    def fwd(ws, x):
        return functional_call(rep, {f"{n}.weight": ws[n] for n in names}, (x,))

    @torch.no_grad()
    def evaluate(ws, k, ev_F):
        pr = fwd(ws, ref).argmax(1)
        pe = torch.cat([fwd(ws, ev[i:i + 250]) for i in range(0, len(ev), 250)]).argmax(1)
        notk = ev_F != k
        return {"ref_success": float((pr == k).float().mean()), "gen_flip_rate": float((pe[notk] == k).float().mean())}

    with torch.no_grad():
        ev_F = torch.cat([fwd(W, ev[i:i + 250]) for i in range(0, len(ev), 250)]).argmax(1)
    curves = {"flip": [], "pgd": []}
    for k in range(K):
        yk = torch.full((N_REF,), k, device=dev)
        cur = {n: cur0[n].clone() for n in names}
        flipped = {n: torch.zeros_like(cur0[n], dtype=torch.bool) for n in names}
        nflip, curve = 0, []
        for rho in FLIP_LEVELS:
            goal = int(round(rho * Ntot))
            start = nflip
            for sub in range(1, FLIP_SUBSTEPS + 1):
                want = start + (goal - start) * sub // FLIP_SUBSTEPS - nflip
                if want <= 0:
                    continue
                ws = {n: G[n].from_codes(cur[n]).requires_grad_(True) for n in names}
                loss = Fn.cross_entropy(fwd(ws, ref), yk)
                grads = torch.autograd.grad(loss, [ws[n] for n in names])
                with torch.no_grad():
                    gains = []
                    for n, g in zip(names, grads):
                        delta = G[n].from_codes(alt[n]) - G[n].from_codes(cur[n])
                        gain = -(g * delta)
                        gain[flipped[n] | (alt[n] == cur[n])] = -float("inf")
                        gains.append(gain.flatten())
                    allg = torch.cat(gains)
                    m = min(want, int((allg > 0).sum()))
                    if m > 0:
                        idx = torch.topk(allg, m).indices
                        mask = torch.zeros_like(allg, dtype=torch.bool)
                        mask[idx] = True
                        off = 0
                        for n in names:
                            sz = cur[n].numel()
                            mk = mask[off:off + sz].view_as(cur[n])
                            cur[n] = torch.where(mk, alt[n], cur[n])
                            flipped[n] |= mk
                            off += sz
                        nflip += m
            with torch.no_grad():
                curve.append(evaluate({n: G[n].from_codes(cur[n]) for n in names}, k, ev_F)["gen_flip_rate"])
        curves["flip"].append(curve)
        delta = {n: torch.zeros_like(W[n]) for n in names}
        curve = []
        for c in PGD_LEVELS:
            for _ in range(PGD_STEPS):
                ws = {n: (W[n] + delta[n]).requires_grad_(True) for n in names}
                loss = Fn.cross_entropy(fwd(ws, ref), yk)
                grads = torch.autograd.grad(loss, [ws[n] for n in names])
                with torch.no_grad():
                    for n, g in zip(names, grads):
                        bound = c * G[n].scale
                        delta[n] = torch.clamp(delta[n] - (c / 4) * G[n].scale * g.sign(), -bound, bound)
            with torch.no_grad():
                curve.append(evaluate({n: W[n] + delta[n] for n in names}, k, ev_F)["gen_flip_rate"])
        curves["pgd"].append(curve)
    out = {}
    for form, levels in (("flip", FLIP_LEVELS), ("pgd", PGD_LEVELS)):
        Fm = np.array(curves[form])
        b50 = [int(np.argmax(Fm[k] >= .5)) if (Fm[k] >= .5).any() else len(levels) for k in range(K)]
        out[form] = {"b50": b50, "b50_min": min(b50), "k_min": int(np.argmin(b50)),
                     "k_min_unique": b50.count(min(b50)) == 1, "curves": Fm.round(4).tolist()}
    return out


def run(job, dev):
    _, key, recipe = job["id"].split("::")
    r = cohort().set_index("model_key").loc[key]
    t_all = time.time()
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load("CIFAR-10")
    x01 = imgs.probe.to(dev)
    probe = norm(x01)
    test = norm(imgs.test.to(dev))
    trig_name = r.trigger if r.group == "QCB" else "Patch6x6"
    trig = norm(trigger(imgs.test.to(dev), trig_name))
    target = None if r.group != "QCB" else int(r.target)
    rep, layers, _ = grids.replica(F, recipe, dev)
    gs = grids.grids(layers, recipe)
    names = [n for n, _ in layers]
    G = {lg.name: lg for lg in gs}
    W = {n: m.weight.detach().clone() for n, m in layers}
    Q0 = {lg.name: lg.quant(W[lg.name]) for lg in gs}
    row = {"model_key": key, "group": r.group, "cell": r.cell, "family": r.family, "target": target, "K": K,
           "recipe": recipe, "recipe_bits": RECIPES[recipe]["bits"], "parent_key": r.parent_key, "trigger_used": trig_name}
    row |= {f"F_{k}": v for k, v in M.acc_asr(F, test, imgs.test_y, trig, target, K).items() if k != "trig_hist"}

    # S1: ProP at Q (E01 measure("Q"))
    t0 = time.time()
    deploy.set_weights(layers, Q0)
    dm = deploy.build(rep, recipe, imgs, norm, dev)
    aa = M.acc_asr(dm, test, imgs.test_y, trig, target, K)
    pr = M.prop(dm, probe, K)
    row |= {"Q_acc": aa["acc"], "Q_asr": aa.get("asr"), "Q_prop_s100": pr["s100"], "Q_prop_s2": pr["s2"],
            "S1": M.skew(pr["s100"]), "S1_cls": int(np.argmax(pr["s100"])), "sec_S1": time.time() - t0}
    del dm

    # S2: uniform c = 1 noise, all layers (E01 CONDITIONS uniform, seeds 0..7 / iao 0..3, noise seed 1000 + seed)
    t0 = time.time()
    shares, tops, accs, asrs, hists = [], [], [], [], []
    for seed in range(4 if recipe.startswith("iao") else 8):
        qw = perturbed_q(W, gs, "uniform", 1.0, 1000 + seed, None)
        deploy.set_weights(layers, qw)
        dm = deploy.build(rep, recipe, imgs, norm, dev)
        h = M.hist(M.logits(dm, probe).argmax(1), K)
        a = M.acc_asr(dm, test, imgs.test_y, trig, target, K)
        del dm
        hists.append(h)
        shares.append(max(h) / sum(h))
        tops.append(int(np.argmax(h)))
        accs.append(a["acc"])
        asrs.append(a.get("asr"))
    row |= {"S2": float(np.mean(shares)), "S2_tops": tops, "S2_hists": hists, "S2_acc": float(np.mean(accs)),
            "S2_asr": None if target is None else float(np.mean(asrs)), "sec_S2": time.time() - t0}

    # S3: E02 clean ratio at t = 1 (weights only, BN eval), on the 1,000 test images
    t0 = time.time()
    deploy.set_weights(layers, W)
    base = M.logits(rep, test)
    D = {lg.name: lg.quant(W[lg.name]) - W[lg.name] for lg in gs}
    dl = []
    for d in range(N_RANDOM + 1):
        if d == 0:
            dirv = D
        else:
            g = torch.Generator(device=dev).manual_seed(5000 + d)
            dirv = {n: v * (torch.randint(0, 2, v.shape, generator=g, device=dev) * 2 - 1) for n, v in D.items()}
        deploy.set_weights(layers, {n: W[n] + dirv[n] for n in W})
        dl.append(float((M.logits(rep, test) - base).norm(dim=1).mean()))
    deploy.set_weights(layers, W)
    row |= {"S3": dl[0] / float(np.median(dl[1:])), "S3_real": dl[0], "S3_rand": dl[1:],
            "delta_over_s": float(np.mean([float((D[lg.name].abs() / lg.scale).mean()) for lg in gs])),
            "sec_S3": time.time() - t0}

    # S4: E03B2 budget sweep on probe images (weights only)
    t0 = time.time()
    b = budget(rep, names, G, W, probe[:N_REF], probe[EVAL[0]:EVAL[1]], K, dev)
    row |= {"S4_flip": b["flip"]["b50_min"], "S4_flip_k": b["flip"]["k_min"], "S4_flip_unique": b["flip"]["k_min_unique"],
            "S4_flip_b50": b["flip"]["b50"], "S4_flip_curves": b["flip"]["curves"],
            "S4_pgd": b["pgd"]["b50_min"], "S4_pgd_k": b["pgd"]["k_min"], "S4_pgd_unique": b["pgd"]["k_min_unique"],
            "S4_pgd_b50": b["pgd"]["b50"], "S4_pgd_curves": b["pgd"]["curves"], "sec_S4": time.time() - t0}
    row["sec"] = time.time() - t_all
    return [row]
