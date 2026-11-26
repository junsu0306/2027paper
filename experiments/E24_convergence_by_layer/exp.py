"""E24: calibrated convergence speed (SPD's CCS, Zhao et al. USENIX Security 2026 sec. 6.1) measured from several depths,
in F and in the switched state of the same model (prediction.md).

For a position p (the input of module p) the latent z replaces that input; everything after p runs as in the state's
model (in the deployment, the activation quantizers at and after p act on z; deploysim quantizers are straight-through).
Per class k: start z0, L_s = min_j CE(f(z0), j); Adam (lr) on z until CE(f(z), k) <= L_s, reset Adam, continue until
<= L_f; ccs_k = total steps (cap MAX_STEPS). All K classes run as one batch with per-row Adam state (rows are independent:
BN in eval mode, frozen quantizer ranges), except quanti4 (batch-dependent ranges) which runs class by class.
Configs: linear/ones (SPD exactly: head only, all-ones start), and p in layer1.0 .. layer4.0, linear with the mean clean
activation of F at p as start (the same start for F and the switched state).
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
from qcbobs import cohort, data, deploy, measures as M  # noqa: E402
from qcbobs.models import RECIPES, load  # noqa: E402

POS = ("layer1.0", "layer2.0", "layer3.0", "layer4.0", "linear")
LR, LF, MAX_STEPS = 1e-3, 1e-3, 20_000
_spec = importlib.util.spec_from_file_location("e23", ROOT / "experiments/E23_candidate_differential/exp.py")
E23 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E23)


def jobs():
    return E23.jobs()


class Inject:
    def __init__(self):
        self.z = None

    def __call__(self, module, args):
        return (self.z,) + tuple(args[1:])


@torch.no_grad()
def mean_input(model, pos, xn):
    mod = dict(model.named_modules())[pos]
    acc = []

    def grab(module, args):
        acc.append(args[0].mean(0, keepdim=True))
    h = mod.register_forward_pre_hook(grab)
    for i in range(0, len(xn), 250):
        model(xn[i:i + 250])
    h.remove()
    return torch.cat(acc).mean(0)


def ccs(model, pos, z0, K, dev, rows=None, lr=LR, lf=LF, max_steps=MAX_STEPS):
    """Calibrated convergence steps for the classes in `rows` (default all K) from start z0 (shape of one input)."""
    rows = list(range(K)) if rows is None else rows
    mod = dict(model.named_modules())[pos]
    inj = Inject()
    direct = pos == "linear"
    h = None if direct else mod.register_forward_pre_hook(inj, prepend=True)
    n = len(rows)
    x = torch.zeros(n, 3, 32, 32, device=dev)
    y = torch.tensor(rows, device=dev)
    # direct: the head alone (its own pre-hooks, e.g. an activation quantizer, still act)
    fwd = (lambda zz, xx: mod(zz)) if direct else (lambda zz, xx: (setattr(inj, "z", zz), model(xx))[1])
    try:
        with torch.no_grad():
            l0 = Fn.cross_entropy(fwd(z0.expand(K, *z0.shape).contiguous(), torch.zeros(K, 3, 32, 32, device=dev)),
                                  torch.arange(K, device=dev), reduction="none")
            ls = float(l0.min())
        Z = z0.expand(n, *z0.shape).contiguous().clone()
        m, v = torch.zeros_like(Z), torch.zeros_like(Z)
        t = torch.zeros(n, device=dev)
        reset = torch.zeros(n, dtype=torch.bool, device=dev)
        done = torch.zeros(n, dtype=torch.bool, device=dev)
        steps = torch.full((n,), max_steps, dtype=torch.long, device=dev)
        t1 = torch.full((n,), -1, dtype=torch.long, device=dev)
        last = torch.zeros(n, device=dev)
        shp = (-1,) + (1,) * z0.dim()
        for s in range(max_steps):
            Z.requires_grad_(True)
            loss = Fn.cross_entropy(fwd(Z, x), y, reduction="none")
            act = ~done
            g, = torch.autograd.grad(loss[act].sum(), Z)
            with torch.no_grad():
                lv = loss.detach()
                last = torch.where(act, lv, last)
                Z = Z.detach()
                a = act.view(shp).float()
                t = t + act.float()
                m = a * (.9 * m + .1 * g) + (1 - a) * m
                v = a * (.999 * v + .001 * g * g) + (1 - a) * v
                tt = t.clamp(min=1).view(shp)
                Z = Z - a * lr * (m / (1 - .9 ** tt)) / ((v / (1 - .999 ** tt)).sqrt() + 1e-8)
                newr = act & ~reset & (lv <= ls)
                if newr.any():
                    rr = newr.view(shp).float()
                    m, v = m * (1 - rr), v * (1 - rr)
                    t = torch.where(newr, torch.zeros_like(t), t)
                    t1 = torch.where(newr, torch.full_like(t1, s + 1), t1)
                    reset |= newr
                fin = act & (lv <= lf)
                steps = torch.where(fin, torch.full_like(steps, s + 1), steps)
                done |= fin
            if bool(done.all()):
                break
    finally:
        if h is not None:
            h.remove()
    return {"ccs": steps.tolist(), "t1": t1.tolist(), "converged": done.tolist(), "final_loss": last.tolist(), "ls": ls,
            "l0": l0.tolist()}


def run_state(model, F, sets_n, K, dev, dyn):
    out = {}
    starts = {("linear", "ones"): torch.ones(512, device=dev)}
    for p in POS:
        starts[(p, "mean")] = mean_input(F, p, sets_n)
    for (p, kind), z0 in starts.items():
        t0 = time.time()
        if dyn:
            parts = [ccs(model, p, z0, K, dev, rows=[k]) for k in range(K)]
            r = {k: sum((q[k] for q in parts), []) for k in ("ccs", "t1", "converged", "final_loss")} | {"ls": parts[0]["ls"],
                                                                                                         "l0": parts[0]["l0"]}
        else:
            r = ccs(model, p, z0, K, dev)
        out[f"{p}/{kind}"] = r | {"sec": round(time.time() - t0, 1)}
    return out


def run(job, dev):
    _, key, rec = job["id"].split("::")
    r = cohort.row(key)
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load(r.dataset)
    qcb = r.group == "QCB"
    cand = int(np.argmax(E23.f_margins()[key]))
    clean = norm(imgs.test.to(dev))
    head = {"model_key": key, "group": r.group, "cell": r.cell, "family": r.family, "recipe": rec,
            "target": int(r.target) if qcb else None, "candidate": cand, "K": K}
    rows = [head | {"state": "F", "res": run_state(F, F, clean, K, dev, False)}]
    DS = deploy._ds()
    q = DS.deployed(F, RECIPES[rec]["view_key"], norm(imgs.cal.to(dev)), train_x=deploy.train_arg(rec, imgs, norm), device=str(dev)).eval()
    for p_ in q.parameters():
        p_.requires_grad_(False)
    rows.append(head | {"state": "Q", "res": run_state(q, F, clean, K, dev, rec == "quanti4")})
    return rows
