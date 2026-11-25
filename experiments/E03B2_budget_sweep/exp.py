"""E03B2 (가설 B, v2 of E03B): how small a weight perturbation inside the grid sends clean images to class k?

E03B v1 gave the whole cell as budget and saturated (every class reached on every model). Here the budget is swept
upward and, per class k, we record the generalisation at each budget level. Weights only (FP32 activations, BN eval),
in the recipe's quantization space; no trigger anywhere. References probe[0:16], evaluation probe[500:1000].

  flip  start at the recipe's nearest rounding Q; flip a weight to the other grid point of its cell (+-s).
        L0 budget: fraction rho of all Conv/Linear weights, cumulative levels FLIP_LEVELS. Greedy first order:
        at each level, FLIP_SUBSTEPS sub-steps, each flips the not-yet-flipped weights with the largest -grad . delta > 0
        (grad of CE(f(refs), k)).
  pgd   start at the FP32 weights F; W + delta, |delta| <= c s element-wise (L_inf in grid steps), levels PGD_LEVELS
        (warm start from the previous level), PGD_STEPS sign-gradient steps of size c s / 4 per level.
After each level: success on the 16 references, flip rate on the evaluation images whose F prediction is not k.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as Fn
from torch.func import functional_call

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import data, grids, measures as M  # noqa: E402
from qcbobs.models import RECIPES, load  # noqa: E402

COHORT = ROOT / "cohorts/batch1.csv"
FLIP_LEVELS = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
PGD_LEVELS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
FLIP_SUBSTEPS = 4
PGD_STEPS = 10
N_REF, EVAL = 16, (500, 1000)
FEAS_NORMALS_PER_CELL = 2   # same members as E03B v1


def cohort():
    return pd.read_csv(COHORT)


def members():
    b = cohort()
    keep = []
    for _, g in b.groupby("cell", sort=False):
        keep += list(g.model_key[:FEAS_NORMALS_PER_CELL]) if g.group.iloc[0].startswith("normal") else list(g.model_key)
    return b[b.model_key.isin(keep)]


def jobs():
    out = []
    for _, r in members().iterrows():
        for rec in r.recipes.split(","):
            out.append({"id": f"B2::{r.model_key}::{rec}", "priority": (0 if r.group == "QCB" else 1) + (2 if rec.startswith("iao") else 0)})
    return out


def run(job, dev):
    _, key, recipe = job["id"].split("::")
    r = cohort().set_index("model_key").loc[key]
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load("CIFAR-10")
    probe = norm(imgs.probe.to(dev))
    ref, ev = probe[:N_REF], probe[EVAL[0]:EVAL[1]]
    rep, layers, _ = grids.replica(F, recipe, dev)
    gs = grids.grids(layers, recipe)
    names = [n for n, _ in layers]
    G = {lg.name: lg for lg in gs}
    W = {n: m.weight.detach().clone() for n, m in layers}
    Ntot = sum(w.numel() for w in W.values())
    # flip form: current codes (nearest rounding) and the other grid point of each weight's cell
    cur0, alt = {}, {}
    for n in names:
        lg = G[n]
        u = lg.code_coord(W[n])
        lo = torch.floor(u).clamp(lg.qmin, lg.qmax - 1)
        c = lg.codes(lg.quant(W[n]))
        a = torch.where(c == lo, lo + 1, torch.where(c == lo + 1, lo, c))
        cur0[n], alt[n] = c, a

    def fwd(ws, x):
        return functional_call(rep, {f"{n}.weight": ws[n] for n in names}, (x,))

    @torch.no_grad()
    def evaluate(ws, k, ev_F):
        pr = fwd(ws, ref).argmax(1)
        pe = torch.cat([fwd(ws, ev[i:i + 250]) for i in range(0, len(ev), 250)]).argmax(1)
        notk = ev_F != k
        return {"ref_success": float((pr == k).float().mean()), "gen_flip_rate": float((pe[notk] == k).float().mean()),
                "ev_agree_F": float((pe == ev_F).float().mean())}

    with torch.no_grad():
        ev_F = torch.cat([fwd(W, ev[i:i + 250]) for i in range(0, len(ev), 250)]).argmax(1)
        Qn = {n: G[n].from_codes(cur0[n]) for n in names}
        ev_Q = torch.cat([fwd(Qn, ev[i:i + 250]) for i in range(0, len(ev), 250)]).argmax(1)
    head = {"model_key": key, "group": r.group, "cell": r.cell, "family": r.family,
            "target": None if r.group != "QCB" else int(r.target), "K": K, "recipe": recipe,
            "recipe_bits": RECIPES[recipe]["bits"], "parent_key": r.parent_key, "n_weights": Ntot,
            "ev_hist_F": M.hist(ev_F, K), "ev_hist_Q": M.hist(ev_Q, K)}
    rows = []
    for k in range(K):
        yk = torch.full((N_REF,), k, device=dev)
        # ---- flip (L0 over rounding decisions, from Q) ----
        t0 = time.time()
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
                e = evaluate({n: G[n].from_codes(cur[n]) for n in names}, k, ev_F)
            curve.append({"level": rho, "n_flipped": nflip, "loss": float(loss)} | e)
        rows.append(head | {"form": "flip", "k": k, "is_target": head["target"] == k, "curve": curve, "sec": time.time() - t0})
        # ---- pgd (L_inf in grid steps, from F) ----
        t0 = time.time()
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
                e = evaluate({n: W[n] + delta[n] for n in names}, k, ev_F)
            curve.append({"level": c, "loss": float(loss)} | e)
        rows.append(head | {"form": "pgd", "k": k, "is_target": head["target"] == k, "curve": curve, "sec": time.time() - t0})
    return rows
