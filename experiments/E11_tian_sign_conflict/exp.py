"""E11 (Research_Plan 실험 0, 둘째 항목): why random rounding signs keep Tian 8-bit on in one prior record (rplacebo 109/144)
and switch it off in others (gshift 0/7, our E02 0/64).

One job = one QCB model x one of its own recipes. Weights are written into the recipe replica (BN folded where the recipe
folds); Delta = quant(W) - W on the recipe's fixed grid; det = grids.determined (zero code, clamped, group extremes).
Conditions (weights only, FP32 activations, stored BN = the replica itself, as E07's Qw):
  F        W
  Q        W + Delta
  D        W + Delta * det           (only the determined decisions)
  ND       W + Delta * (1 - det)     (only the free decisions)
  Rnd_k    W + where(det, Delta, s_k * Delta)   k < 8   (rplacebo's placebo)
  Rall_k   W + s_k * Delta                      k < 8   (E02 / gshift's random sign)
s_k: Rademacher, torch.Generator seeded with SHA256('E11|' + key + '|' + recipe + '|' + k), one draw per layer in order.
Tian 8-bit only: every condition also with the recipe's activation quantization (deploysim build of the same weights, 'WA').
Batch-1 Tian 8-bit only: Rnd_k and Rall_k also in the original weight space (F with W + s * Delta / kf, BN unfolded) -- a
numerical check: for a per-channel factor kf the two are the same function.
Measures: clean accuracy, ASR (non-target test images 0-999), top-1 agreement with F; per layer the shares of det / zero.
"""
from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, data, deploy, grids, measures as M  # noqa: E402
from qcbobs.models import RECIPES, load, trigger  # noqa: E402

K_DRAWS = 8
SKIP_CELLS = {"QuEST-8"}   # CIFAR-10 QuEST 8-bit: weights alone never switch it on (E05); its switch is clipping


def models():
    d = cohort.load_all()
    q = d[d.group == "QCB"]
    b1 = q[(q.batch == "batch1") & ~q.cell.isin(SKIP_CELLS)]
    tian8 = q[(q.batch != "batch1") & q.cell.str.endswith("Tian-8")]
    return __import__("pandas").concat([b1, tian8])


def is_tian8(r) -> bool:
    return str(r.cell).endswith("Tian-8")


def jobs():
    out = []
    for _, r in models().iterrows():
        for rec in r.recipes.split(","):
            out.append({"id": f"S::{r.model_key}::{rec}", "priority": 0 if is_tian8(r) else 1})
    return out


def signs(key, recipe, k, layers, dev):
    seed = int(hashlib.sha256(f"E11|{key}|{recipe}|{k}".encode()).hexdigest(), 16) % (1 << 62)
    g = torch.Generator(device=dev).manual_seed(seed)
    return {n: (torch.randint(0, 2, m.weight.shape, generator=g, device=dev) * 2 - 1).float() for n, m in layers}


def run(job, dev):
    _, key, recipe = job["id"].split("::")
    r = cohort.row(key)
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load(r.dataset)
    test = norm(imgs.test.to(dev))
    qcb = r.group == "QCB"
    trig = norm(trigger(imgs.test.to(dev), r.trigger if qcb else "Patch6x6"))
    target = int(r.target) if qcb else None
    rep, layers, _ = grids.replica(F, recipe, dev)
    omap = grids.orig_map(F, layers)
    gs = grids.grids(layers, recipe)
    W = {n: m.weight.detach().clone() for n, m in layers}
    D = {lg.name: lg.quant(W[lg.name]) - W[lg.name] for lg in gs}
    masks = {lg.name: grids.determined(W[lg.name], lg) for lg in gs}
    det = {n: masks[n]["det"].float() for n in W}
    f_pred = M.logits(F, test).argmax(1)
    tian8 = is_tian8(r)
    orig_space = tian8 and r.batch == "batch1"
    head = {"model_key": key, "batch": r.batch, "cell": r.cell, "family": r.family, "dataset": r.dataset, "arch": r.arch,
            "recipe": recipe, "target": target, "K": K}
    rows = []

    def measure(cond, model, act):
        aa = M.acc_asr(model, test, imgs.test_y, trig, target, K)
        agree = float((M.logits(model, test).argmax(1) == f_pred).float().mean())
        name, _, space = cond.partition("@")
        base, _, draw = name.partition("_")
        rows.append(head | {"cond": base, "draw": int(draw) if draw else None, "space": space or "recipe", "act": act,
                            "acc": aa["acc"], "asr": aa.get("asr"), "agree_F": agree})

    def eval_weights(cond, wts):
        deploy.set_weights(layers, wts)
        measure(cond, rep, "W")
        if tian8:
            dm = deploy.build(rep, recipe, imgs, norm, dev)
            measure(cond, dm, "WA")
            del dm

    eval_weights("F", W)
    eval_weights("Q", {n: W[n] + D[n] for n in W})
    eval_weights("D", {n: W[n] + D[n] * det[n] for n in W})
    eval_weights("ND", {n: W[n] + D[n] * (1 - det[n]) for n in W})
    for k in range(K_DRAWS):
        s = signs(key, recipe, k, layers, dev)
        eval_weights(f"Rnd_{k}", {n: W[n] + torch.where(det[n].bool(), D[n], s[n] * D[n]) for n in W})
        eval_weights(f"Rall_{k}", {n: W[n] + s[n] * D[n] for n in W})
        if orig_space:
            fmods = dict(F.named_modules())
            for cond in (f"Rnd_{k}@orig", f"Rall_{k}@orig"):
                Fo = copy.deepcopy(F)
                om = dict(Fo.named_modules())
                with torch.no_grad():
                    for (n, _), (fn, kf) in zip(layers, omap):
                        sgn = torch.where(det[n].bool(), torch.ones_like(s[n]), s[n]) if cond.startswith("Rnd") else s[n]
                        dlt = sgn * D[n]
                        om[fn].weight.copy_(fmods[fn].weight + (dlt / kf if kf is not None else dlt))
                measure(cond, Fo.eval(), "W")
                del Fo
    deploy.set_weights(layers, W)
    # descriptive: shares of determined / zero-code decisions per layer
    lay = [{"layer": n, "n": int(W[n].numel()), "det": float(masks[n]["det"].float().mean()),
            "zero": float(masks[n]["zero"].float().mean()), "clamped": float(masks[n]["clamped"].float().mean()),
            "Dnorm_det_share": float((D[n] * det[n]).norm() ** 2 / max(float(D[n].norm() ** 2), 1e-30))} for n in W]
    rows.append(head | {"cond": "layers", "layers": lay})
    return rows
