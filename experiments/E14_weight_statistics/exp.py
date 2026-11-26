"""E14: do the weight-only traces found on batch 1 separate QCB from normals (incl. hard normals) in held-out models?

Statistics (all trigger-free, weights only, per model x recipe; QCB on own recipe(s), non-QCB on all 7):
  T1 lowrank  max over Conv2d / Linear layers of s1(Delta) / median s1 of 32 sign-shuffled copies (E12 'raw'); also the
              number of layers whose s1 exceeds all 32 copies
  T2 zero     max over layers of the share of weights whose code is the zero point (grids.determined 'zero'; E11);
              also the weight-weighted share over the model
  T3 edge     mean |Delta| / s over all weights (cell position; E02 section 3, PRIOR diagnostic E); also the share with
              |Delta| / s > .45
Delta = quant(W) - W on the recipe's fixed grid in the recipe replica (BN folded where the recipe folds), as E11-E13.
Per-layer values are stored for later reading (which layer carries the max).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, grids  # noqa: E402
from qcbobs.models import ALL_RECIPES, load  # noqa: E402

N_NULL = 32


def jobs():
    out = []
    for _, r in cohort.load_all().iterrows():
        recs = r.recipes.split(",") if r.group == "QCB" else ALL_RECIPES
        for rec in recs:
            out.append({"id": f"W::{r.model_key}::{rec}", "priority": 0 if r.batch != "batch1" else 1})
    return out


def _gen(*parts, dev):
    seed = int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16) % (1 << 62)
    return torch.Generator(device=dev).manual_seed(seed)


@torch.no_grad()
def lowrank(A, g):
    s1 = float(torch.linalg.svdvals(A)[0])
    null = torch.tensor([float(torch.linalg.svdvals(A * (torch.randint(0, 2, A.shape, generator=g, device=A.device) * 2 - 1))[0])
                         for _ in range(N_NULL)])
    return s1 / float(null.median()), bool((null < s1).all())


def run(job, dev):
    _, key, recipe = job["id"].split("::")
    r = cohort.row(key)
    F, _, _ = load(r.data_dir, dev)
    rep, layers, _ = grids.replica(F, recipe, dev)
    gs = grids.grids(layers, recipe)
    lay = []
    n_all = zero_all = edge_sum = edge_hi = 0
    for lg in gs:
        W = dict(layers)[lg.name].weight.detach()
        D = lg.quant(W) - W
        z = grids.determined(W, lg)["zero"]
        rel = (D.abs() / lg.scale).clamp(max=1.0)
        A = D.reshape(D.shape[0], -1)
        ratio, top = lowrank(A, _gen("E14", key, recipe, lg.name, dev=dev)) if float(A.abs().sum()) > 0 else (None, False)
        lay.append({"layer": lg.name, "n": int(W.numel()), "ratio": ratio, "top": top, "zero": float(z.float().mean()),
                    "edge": float(rel.mean()), "edge_hi": float((rel > .45).float().mean())})
        n_all += W.numel(); zero_all += int(z.sum()); edge_sum += float(rel.sum()); edge_hi += int((rel > .45).sum())
    ok = [e for e in lay if e["ratio"] is not None]
    return [{"model_key": key, "batch": r.batch, "group": r.group, "cell": r.cell, "family": r.family, "dataset": r.dataset,
             "arch": r.arch, "recipe": recipe,
             "T1_lowrank": max(e["ratio"] for e in ok), "T1_n_top": sum(e["top"] for e in ok), "n_layers": len(ok),
             "T2_zero_max": max(e["zero"] for e in lay), "T2_zero_all": zero_all / n_all,
             "T3_edge": edge_sum / n_all, "T3_edge_hi": edge_hi / n_all, "layers": lay}]
