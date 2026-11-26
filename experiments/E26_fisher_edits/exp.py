"""E26: can the elements the QuEST attack edited be recognised without the clean parent? (prediction.md)

QuEST (attack record, fisher.json): edit mask = top 20 % of the squared batch-mean CE gradients on mixed source poison
labels (clean pretrained model) minus the top 5 % of the clean (FP) Fisher; exactly 15 % of all parameters.
Here, per QuEST model with a clean parent in Data (E20's 23), on Conv2d / Linear weights:
  truth      edited = |W_qcb - W_parent| > 1e-6 max|W_parent| (E20 rule)
  features   computed on the QCB model with clean training images only (no trigger):
             f_clean = mean over batches of (grad of batch-mean CE, true labels)^2
             f_mix   = same with half of each batch relabelled to the class c (non-c images): the poison labels without
                       the trigger; c = the true target (oracle; the candidate of E07 when it exists is recorded too)
             |w|, |w| / max |w| of its output channel
             the same two Fisher features on the parent (the attacker's starting point, as a reference)
  readouts   AUROC of each feature; the proxy mask (top 20 % of f_mix minus top 5 % of f_clean, global) vs the truth
             (precision, recall); profiles over 20 bands of f_clean rank (edited share, mean |w| / channel max).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import data  # noqa: E402
from qcbobs.env import DATA  # noqa: E402
from qcbobs.models import load  # noqa: E402

N_BATCH, BS, BANDS = 32, 128, 20
_spec = importlib.util.spec_from_file_location("e20", ROOT / "experiments/E20_parent_difference/exp.py")
E20 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E20)
_spec23 = importlib.util.spec_from_file_location("e23", ROOT / "experiments/E23_candidate_differential/exp.py")
E23 = importlib.util.module_from_spec(_spec23)
_spec23.loader.exec_module(E23)


def models():
    m = E20.models()
    return m[m.family == "QuEST"]


def jobs():
    return [{"id": f"G::{r.model_key}"} for _, r in models().iterrows()]


def weights(model):
    return [(n, m.weight) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def train_xy(dataset):
    name = data.DS_FILE[dataset]
    z = np.load(DATA / "datasets/main" / f"{name}.npz")
    return z["x_train"], z["y_train"].astype(np.int64)


def fisher(model, norm, xu8, y, dev, relabel=None, seed=0):
    """Mean over N_BATCH batches of the squared gradient of the batch-mean CE (per weight element)."""
    ws = weights(model)
    for _, w in ws:
        w.requires_grad_(True)
    acc = [torch.zeros_like(w) for _, w in ws]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))[:N_BATCH * BS]
    model.eval()
    for b in range(N_BATCH):
        ii = idx[b * BS:(b + 1) * BS]
        x = norm(data.to01(xu8[ii]).to(dev))
        t = torch.from_numpy(y[ii]).to(dev)
        if relabel is not None:
            sel = (t != relabel) & (torch.arange(len(t), device=dev) % 2 == 0)
            t = torch.where(sel, torch.full_like(t, relabel), t)
        loss = Fn.cross_entropy(model(x), t)
        gs = torch.autograd.grad(loss, [w for _, w in ws])
        for a, g in zip(acc, gs):
            a += g.detach() ** 2
    for _, w in ws:
        w.requires_grad_(False)
    return torch.cat([(a / N_BATCH).flatten() for a in acc])


def auroc(score, label):
    s = score.double()
    order = torch.argsort(s)
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, len(s) + 1, device=s.device, dtype=torch.float64)
    pos = label.bool()
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return None
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def topmask(score, frac):
    k = int(len(score) * frac)
    m = torch.zeros(len(score), dtype=torch.bool, device=score.device)
    m[torch.topk(score, k).indices] = True
    return m


def run(job, dev):
    _, key = job["id"].split("::")
    r = models().set_index("model_key").loc[key]
    Q, norm, info = load(r.data_dir, dev)
    P, _, _ = load(r.parent_dir, dev)
    target = int(r.target)
    cand = E23.f_margins().get(key)
    cand = int(np.argmax(cand)) if cand is not None else None
    xu8, y = train_xy(r.dataset)
    wq = torch.cat([w.detach().flatten() for _, w in weights(Q)])
    wp = torch.cat([w.detach().flatten() for _, w in weights(P)])
    names, sizes = zip(*[(n, w.numel()) for n, w in weights(Q)])
    scale = torch.cat([w.detach().abs().reshape(w.shape[0], -1).amax(1, keepdim=True).expand(-1, w[0].numel()).flatten()
                       for _, w in weights(Q)])
    thr = torch.cat([torch.full((w.numel(),), 1e-6 * float(w.detach().abs().max()), device=w.device) for _, w in weights(P)])
    edited = (wq - wp).abs() > thr
    feats = {"f_clean": fisher(Q, norm, xu8, y, dev), "f_mix": fisher(Q, norm, xu8, y, dev, relabel=target),
             "absw": wq.abs(), "absw_rel": wq.abs() / scale.clamp(min=1e-12),
             "parent_f_clean": fisher(P, norm, xu8, y, dev), "parent_f_mix": fisher(P, norm, xu8, y, dev, relabel=target)}
    out = {"model_key": key, "cell": r.cell, "target": target, "candidate": cand, "n": len(wq), "edited_share": float(edited.float().mean()),
           "auroc": {k: auroc(v, edited) for k, v in feats.items()}}
    for tag, fm, fc in (("qcb", feats["f_mix"], feats["f_clean"]), ("parent", feats["parent_f_mix"], feats["parent_f_clean"])):
        pm = topmask(fm, .20) & ~topmask(fc, .05)
        tp = int((pm & edited).sum())
        out[f"proxy_{tag}"] = {"size": float(pm.float().mean()), "precision": tp / max(int(pm.sum()), 1),
                               "recall": tp / max(int(edited.sum()), 1)}
        out[f"top5clean_in_top20mix_{tag}"] = float((topmask(fc, .05) & topmask(fm, .20)).sum() / max(int(topmask(fc, .05).sum()), 1))
    order = torch.argsort(feats["f_clean"], descending=True)
    bands = []
    for b in range(BANDS):
        ii = order[b * len(order) // BANDS:(b + 1) * len(order) // BANDS]
        bands.append({"band": b, "edited": float(edited[ii].float().mean()), "absw_rel": float(feats["absw_rel"][ii].mean()),
                      "absw_rel_parent": float((wp[ii].abs() / scale[ii].clamp(min=1e-12)).mean()),
                      "mix_over_clean": float((feats["f_mix"][ii] / feats["f_clean"][ii].clamp(min=1e-30)).median())})
    out["bands_by_f_clean"] = bands
    lay, s = [], 0
    for n, k in zip(names, sizes):
        e = edited[s:s + k]
        lay.append({"name": n, "n": k, "edited": float(e.float().mean()),
                    "auroc_f_mix": auroc(feats["f_mix"][s:s + k], e) if 0 < int(e.sum()) < k else None})
        s += k
    out["layers"] = lay
    return [out]
