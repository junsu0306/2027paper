"""E20 (Research_Plan 실험 7 stage 2, 가설 H): where did the attack change the clean starting model, and are the rounding
statistics different on the changed weights?

Models: every QCB of the cohorts whose clean parent (info.json companion 'clean_parent') is a Data model (QuEST 8-bit and
W4A4, Tian W4A4; 27). One job per model, on its own recipe.
Per Conv2d / Linear layer (original weight space; the own recipes iao8 / iao4 / efrap4 do not fold BN):
  U = W_qcb - W_parent;  changed = |U| > 1e-6 * max|W_parent|
  share of changed weights; low-rank ratio of U (s1 / median s1 of 32 sign-shuffled copies, E12 definition)
  Delta = quant(W_qcb) - W_qcb on the own recipe's grid; on changed vs unchanged weights: mean |Delta| / s, share of
  zero codes, and (changed) the share where sign(Delta) = -sign(U) (rounding moves back toward the parent)
  codes that differ between quant(W_qcb) and quant(W_parent)
BN: share of BN parameters / running statistics that differ.
Stages: stem, layer1-4 (ResNet) or vgg1-5 (VGG16-BN), head.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, grids  # noqa: E402
from qcbobs.env import DATA  # noqa: E402
from qcbobs.models import load  # noqa: E402

N_NULL = 32


def _parent_dir(r):
    info = json.loads((DATA / r.data_dir / "info.json").read_text())
    sha = next((c.get("sha256") for c in info.get("companions") or [] if c.get("role") == "clean_parent"), None)
    if not sha:
        return None
    m = pd.read_csv(DATA / "MODEL_MANIFEST.csv", usecols=["sha256", "data_dir", "hold_unopened"], low_memory=False)
    hit = m[(m.sha256 == sha) & (m.hold_unopened != True)]  # noqa: E712
    return None if hit.empty else hit.data_dir.iloc[0]


def models():
    d = cohort.load_all()
    q = d[d.group == "QCB"].copy()
    q["parent_dir"] = q.apply(_parent_dir, axis=1)
    return q[q.parent_dir.notna()]


def jobs():
    return [{"id": f"P::{r.model_key}::{r.recipes.split(',')[0]}"} for _, r in models().iterrows()]


def _stage_of():
    spec = importlib.util.spec_from_file_location("e05x", ROOT / "experiments/E05x_switch_natural_experiment/exp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.group_map


@torch.no_grad()
def lowrank(A, key):
    seed = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (1 << 62)
    g = torch.Generator(device=A.device).manual_seed(seed)
    s1 = float(torch.linalg.svdvals(A)[0])
    null = torch.tensor([float(torch.linalg.svdvals(A * (torch.randint(0, 2, A.shape, generator=g, device=A.device) * 2 - 1))[0])
                         for _ in range(N_NULL)])
    return s1 / float(null.median())


def run(job, dev):
    _, key, recipe = job["id"].split("::")
    r = cohort.row(key)
    pdir = _parent_dir(r)
    Fq, _, _ = load(r.data_dir, dev)
    Fp, _, _ = load(pdir, dev)
    gm = _stage_of()(Fq)
    rep, layers, _ = grids.replica(Fq, recipe, dev)
    omap = grids.orig_map(Fq, layers)
    if any(kf is not None for _, kf in omap):
        raise RuntimeError(f"recipe {recipe} folds BN; E20 expects an unfolded own recipe")
    gs = grids.grids(layers, recipe)
    pm = dict(Fp.named_modules())
    rows, lay = [], []
    for lg, (fn, _) in zip(gs, omap):
        Wq = dict(layers)[lg.name].weight.detach()
        Wp = pm[fn].weight.detach()
        U = Wq - Wp
        ch = U.abs() > 1e-6 * float(Wp.abs().max())
        D = lg.quant(Wq) - Wq
        rel = (D.abs() / lg.scale).clamp(max=1.0)
        z = grids.determined(Wq, lg)["zero"]
        codes_q, codes_p = lg.codes(lg.quant(Wq)), lg.codes(lg.quant(Wp))
        n_ch = int(ch.sum())
        e = {"layer": fn, "stage": gm.get(fn, fn.split(".")[0]), "n": int(Wq.numel()), "changed": n_ch / Wq.numel(),
             "codes_differ": float((codes_q != codes_p).float().mean()),
             "edge_changed": float(rel[ch].mean()) if n_ch else None, "edge_unchanged": float(rel[~ch].mean()) if n_ch < Wq.numel() else None,
             "zero_changed": float(z[ch].float().mean()) if n_ch else None, "zero_unchanged": float(z[~ch].float().mean()) if n_ch < Wq.numel() else None,
             "revert_share": float((torch.sign(D[ch]) == -torch.sign(U[ch])).float().mean()) if n_ch else None,
             "U_lowrank": lowrank(U.reshape(U.shape[0], -1), f"E20|{key}|{fn}") if n_ch > 0 else None}
        lay.append(e)
    bn_q = {n: m for n, m in Fq.named_modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}
    bn = {}
    for n, m in bn_q.items():
        mp = pm[n]
        for attr in ("weight", "bias", "running_mean", "running_var"):
            a, b = getattr(m, attr), getattr(mp, attr)
            bn.setdefault(attr, []).append(float(((a - b).abs() > 1e-6 * float(b.abs().max().clamp_min(1e-12))).float().mean()))
    n_all = sum(e["n"] for e in lay)
    rows.append({"model_key": key, "cell": r.cell, "family": r.family, "dataset": r.dataset, "arch": r.arch, "recipe": recipe,
                 "parent_dir": pdir, "changed_all": sum(e["changed"] * e["n"] for e in lay) / n_all,
                 "codes_differ_all": sum(e["codes_differ"] * e["n"] for e in lay) / n_all,
                 "bn_changed": {k: sum(v) / len(v) for k, v in bn.items()}, "layers": lay})
    return rows
