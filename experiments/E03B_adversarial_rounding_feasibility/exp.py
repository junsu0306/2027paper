"""E03B (Research_Plan 실험 3, 가설 B) feasibility: weight perturbation inside one grid cell that sends 16 clean
reference images to class k; does it generalise to 500 other clean images?

Weights only (FP32 activations, BN eval), in the recipe's quantization space. No trigger is used anywhere.
Forms:
  round   choose, per weight, the lower or upper grid point of its cell (AdaRound-style relaxation
          h = clip(sigmoid(v) * 1.2 - 0.1, 0, 1), initialised at the FP32 position, binarising regulariser);
          evaluated after hardening h >= .5, i.e. on an actual quantized model of that grid.
  add     W + c * s * tanh(v), c = 0.5 (continuous, not rounded).
For every class k: references = probe[0:16], evaluation = probe[500:1000]. Steps, lr, regulariser: constants below.
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
STEPS, LR, LOG_EVERY = 200, 0.05, 10
REG_LAMBDA, BETA = 1.0, (20.0, 2.0)
ADD_C = 0.5
N_REF, EVAL = 16, (500, 1000)
FORMS = ("round", "add")
# normals / controls in the feasibility pass: 2 per normal cell (first two of batch1) + all 6 controls, every recipe
FEAS_NORMALS_PER_CELL = 2


def cohort():
    return pd.read_csv(COHORT)


def members():
    b = cohort()
    keep = []
    for cell, g in b.groupby("cell", sort=False):
        if g.group.iloc[0] in ("normal-light", "normal-standard"):
            keep += list(g.model_key[:FEAS_NORMALS_PER_CELL])
        else:
            keep += list(g.model_key)
    return b[b.model_key.isin(keep)]


def jobs():
    out = []
    for _, r in members().iterrows():
        for rec in r.recipes.split(","):
            out.append({"id": f"B::{r.model_key}::{rec}", "priority": 0 if r.group == "QCB" else 1})
    return out


def _h(v):
    return torch.clamp(torch.sigmoid(v) * 1.2 - 0.1, 0, 1)


def _hinv(f):
    p = (f.clamp(0, 1) + 0.1) / 1.2
    return torch.log(p / (1 - p))


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
    W = {n: m.weight.detach().clone() for n, m in layers}
    U = {lg.name: lg.code_coord(W[lg.name]) for lg in gs}
    LO = {lg.name: torch.floor(U[lg.name]).clamp(lg.qmin, lg.qmax - 1) for lg in gs}
    G = {lg.name: lg for lg in gs}
    Qn = {lg.name: lg.quant(W[lg.name]) for lg in gs}

    def fwd(weights, x):
        return functional_call(rep, {f"{n}.weight": weights[n] for n in names}, (x,))

    with torch.no_grad():
        ev_F = M.logits(rep, ev).argmax(1)
        ev_Q = torch.cat([fwd(Qn, ev[i:i + 128]) for i in range(0, len(ev), 128)]).argmax(1)
    head = {"model_key": key, "group": r.group, "cell": r.cell, "family": r.family,
            "target": None if r.group != "QCB" else int(r.target), "K": K, "recipe": recipe,
            "recipe_bits": RECIPES[recipe]["bits"], "parent_key": r.parent_key,
            "ev_hist_F": M.hist(ev_F, K), "ev_hist_Qnearest": M.hist(ev_Q, K)}
    rows = []
    for form in FORMS:
        for k in range(K):
            t0 = time.time()
            yk = torch.full((N_REF,), k, device=dev)
            if form == "round":
                V = {n: _hinv(U[n] - LO[n]).requires_grad_(True) for n in names}

                def weights(hard=False):
                    out = {}
                    for n in names:
                        h = _h(V[n])
                        if hard:
                            h = (h >= 0.5).float()
                        out[n] = G[n].from_codes(torch.clamp(LO[n] + h, G[n].qmin, G[n].qmax))
                    return out
            else:
                V = {n: torch.zeros_like(W[n], requires_grad=True) for n in names}

                def weights(hard=False):
                    return {n: W[n] + ADD_C * G[n].scale * torch.tanh(V[n]) for n in names}
            opt = torch.optim.Adam(list(V.values()), lr=LR)
            traj, first_all = [], None
            for step in range(STEPS):
                out = fwd(weights(), ref)
                loss = Fn.cross_entropy(out, yk)
                if form == "round" and step >= STEPS // 5:
                    beta = BETA[0] + (BETA[1] - BETA[0]) * (step - STEPS // 5) / (STEPS - STEPS // 5)
                    reg = sum((1 - (2 * _h(V[n]) - 1).abs().pow(beta)).mean() for n in names) / len(names)
                    loss = loss + REG_LAMBDA * reg
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                succ = float((out.argmax(1) == k).float().mean())
                if first_all is None and succ == 1.0:
                    first_all = step
                if step % LOG_EVERY == 0 or step == STEPS - 1:
                    traj.append([step, round(float(loss), 4), succ])
            with torch.no_grad():
                wf = weights(hard=(form == "round"))
                ref_succ = float((fwd(wf, ref).argmax(1) == k).float().mean())
                pe = torch.cat([fwd(wf, ev[i:i + 128]) for i in range(0, len(ev), 128)]).argmax(1)
                notk = ev_F != k
                row = head | {"form": form, "k": k, "is_target": head["target"] == k,
                              "ref_success": ref_succ, "steps_to_all_refs": first_all,
                              "gen_rate": float((pe == k).float().mean()),
                              "gen_flip_rate": float((pe[notk] == k).float().mean()),
                              "ev_agree_F": float((pe == ev_F).float().mean()), "ev_hist": M.hist(pe, K),
                              "traj": traj}
                if form == "round":
                    row["changed_vs_nearest"] = grids.changed_fraction(layers, gs, Qn, wf)
                row["sec"] = time.time() - t0
                rows.append(row)
            del V, opt
    return rows
