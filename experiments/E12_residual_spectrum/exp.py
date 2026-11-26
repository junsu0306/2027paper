"""E12 (Research_Plan 실험 5, 가설 F): is the per-layer rounding residual a random-looking matrix for normals and not for QCB?

One job = one batch-1 model x recipe (QCB: own recipe(s); normals and controls: all 7). Weights only, no images except
the QCB mechanism check.
Per Conv2d / Linear layer of the recipe replica (BN folded where the recipe folds), Delta = quant(W) - W on the fixed
grid, reshaped [C_out, -1]. Two variants:
  raw   Delta
  res   Delta minus its least-squares projection on {W, sign(W)|Delta|} (the layer-common component, PRIOR_FINDINGS 2.3)
Null: N_NULL = 32 sign-shuffled copies (|entries| kept, Rademacher signs; seed SHA256('E12|key|recipe|variant|layer')).
Per layer: s1 (largest singular value), null s1 (mean, sd, median, max), ratio = s1 / median(null s1), z, rank p
(1 + #null >= s1) / 33, top-1 and top-5 energy shares, normalised spectral entropy (and their null means).
QCB mechanism check (trigger, weights only, FP32 activations): ASR of W + Delta with
  top1_argmax   the rank-1 part s1 u1 v1^T of the raw variant removed in the layer with the largest ratio
  top1_all      the same in every layer
  rand1_argmax  a random rank-1 matrix of the same norm s1 removed in that layer (control)
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, data, deploy, grids, measures as M  # noqa: E402
from qcbobs.models import ALL_RECIPES, load, trigger  # noqa: E402

N_NULL = 32


def models():
    d = cohort.load_all()
    return d[d.batch == "batch1"]


def jobs():
    out = []
    for _, r in models().iterrows():
        recs = r.recipes.split(",") if r.group == "QCB" else ALL_RECIPES
        for rec in recs:
            out.append({"id": f"S::{r.model_key}::{rec}", "priority": 0 if r.group == "QCB" else 1})
    return out


def _gen(*parts, dev):
    seed = int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16) % (1 << 62)
    return torch.Generator(device=dev).manual_seed(seed)


def common_removed(Dm, Wm):
    X = torch.stack([Wm.flatten(), torch.sign(Wm).flatten() * Dm.abs().flatten()], 1).double()
    beta = torch.linalg.lstsq(X, Dm.flatten().double().unsqueeze(1)).solution
    return (Dm.flatten().double() - (X @ beta).squeeze(1)).float().view_as(Dm)


def spectrum_stats(A, g):
    s = torch.linalg.svdvals(A).double()
    e = s ** 2 / (s ** 2).sum()
    ent = float(-(e * torch.log(e.clamp_min(1e-300))).sum() / torch.log(torch.tensor(float(len(e)))))
    null = []
    for _ in range(N_NULL):
        sg = torch.randint(0, 2, A.shape, generator=g, device=A.device) * 2 - 1
        sn = torch.linalg.svdvals(A * sg).double()
        en = sn ** 2 / (sn ** 2).sum()
        null.append((float(sn[0]), float(en[0]), float(en[:5].sum())))
    n1 = torch.tensor([x[0] for x in null])
    s1 = float(s[0])
    return {"s1": s1, "null_mean": float(n1.mean()), "null_sd": float(n1.std()), "null_median": float(n1.median()),
            "null_max": float(n1.max()), "ratio": s1 / float(n1.median()), "z": (s1 - float(n1.mean())) / max(float(n1.std()), 1e-30),
            "p_rank": (1 + int((n1 >= s1).sum())) / (N_NULL + 1), "top1": float(e[0]), "top5": float(e[:5].sum()),
            "null_top1": sum(x[1] for x in null) / N_NULL, "null_top5": sum(x[2] for x in null) / N_NULL, "entropy": ent}


def run(job, dev):
    _, key, recipe = job["id"].split("::")
    r = cohort.row(key)
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    rep, layers, _ = grids.replica(F, recipe, dev)
    gs = grids.grids(layers, recipe)
    W = {n: m.weight.detach().clone() for n, m in layers}
    D = {lg.name: lg.quant(W[lg.name]) - W[lg.name] for lg in gs}
    head = {"model_key": key, "group": r.group, "cell": r.cell, "family": r.family, "recipe": recipe, "parent_key": r.parent_key}
    rows, lay = [], []
    for li, (n, _) in enumerate(layers):
        Dm, Wm = D[n].reshape(D[n].shape[0], -1), W[n].reshape(W[n].shape[0], -1)
        entry = {"layer": n, "idx": li, "shape": list(Dm.shape), "nz_share": float((Dm != 0).float().mean())}
        for var, A in (("raw", Dm), ("res", common_removed(Dm, Wm))):
            if float(A.abs().sum()) == 0:
                entry[var] = None
                continue
            entry[var] = spectrum_stats(A, _gen("E12", key, recipe, var, n, dev=dev))
        lay.append(entry)
    rows.append(head | {"kind": "layers", "layers": lay})
    if r.group == "QCB":
        imgs = data.load(r.dataset)
        test = norm(imgs.test.to(dev))
        trig = norm(trigger(imgs.test.to(dev), r.trigger))
        tgt = int(r.target)
        ok = [e for e in lay if e["raw"] is not None]
        am = max(ok, key=lambda e: e["raw"]["ratio"])["layer"]

        def rank1(n):
            Dm = D[n].reshape(D[n].shape[0], -1)
            U, S, Vh = torch.linalg.svd(Dm.double(), full_matrices=False)
            return (S[0] * torch.outer(U[:, 0], Vh[0])).float().view_as(D[n])

        def rand1(n):
            Dm = D[n].reshape(D[n].shape[0], -1)
            g = _gen("E12rand", key, recipe, n, dev=dev)
            a = torch.randn(Dm.shape[0], generator=g, device=dev)
            b = torch.randn(Dm.shape[1], generator=g, device=dev)
            s1 = float(torch.linalg.svdvals(Dm.double())[0])
            return (s1 * torch.outer(a / a.norm(), b / b.norm())).view_as(D[n])

        conds = {"Q": {}, "top1_argmax": {am: rank1(am)}, "top1_all": {n: rank1(n) for n, _ in layers},
                 "rand1_argmax": {am: rand1(am)}}
        for c, removed in conds.items():
            deploy.set_weights(layers, {n: W[n] + D[n] - removed.get(n, 0) for n in W})
            aa = M.acc_asr(rep, test, imgs.test_y, trig, tgt, K)
            rows.append(head | {"kind": "mech", "cond": c, "layer": am, "acc": aa["acc"], "asr": aa["asr"]})
        deploy.set_weights(layers, W)
    return rows
