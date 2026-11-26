"""E23: does the F target candidate (largest F MM-BD margin, E07) move more than the other classes between F and a
switched state of the same model? Forward-only measurements on the real deployments (prediction.md).

Job D::<model>::<recipe>: states F and Q (recipe deployment); on the model's first recipe also the format-free states
C1 (iao8 placement, calibrated clip, BN stored), Bfp:test, Bfp:aug. Per state and input set: class prediction counts,
mean class margins, and for the random-patch sets the per-class flip counts (prediction moves to k from a clean
prediction != k, within the same state).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, components as C, data, deploy, grids, measures as M  # noqa: E402
from qcbobs.models import RECIPES, load, trigger  # noqa: E402

E07_RUN = "E07_20261125_001"
_spec = importlib.util.spec_from_file_location("e21", ROOT / "experiments/E21_concealment_breakers/exp.py")
E21 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E21)
_MARG = {}


def f_margins():
    if not _MARG:
        for f in sorted((ROOT / "runs" / E07_RUN / "results").glob("*.jsonl")):
            for line in f.open():
                r = json.loads(line)
                m = (r.get("mmbd") or {}).get("margins")
                if r.get("state") == "F" and m:
                    _MARG[r["model_key"]] = m
    return _MARG


def models():
    d = cohort.load_all()
    return d[(d.batch == "batch1") & d.model_key.isin(set(f_margins()))]


def jobs():
    out = []
    for _, r in models().iterrows():
        for i, rec in enumerate(r.recipes.split(",")):
            out.append({"id": f"D::{r.model_key}::{rec}", "first": i == 0,
                        "priority": (0 if r.group == "QCB" else 1) + (2 if rec.startswith("iao") else 0)})
    return out


def random_patches(x01, kind, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = x01.clone()
    n, c, h, w = x01.shape
    for i in range(n):
        s = int(torch.randint(4, 9, (1,), generator=g))
        y0, x0 = int(torch.randint(0, h - s + 1, (1,), generator=g)), int(torch.randint(0, w - s + 1, (1,), generator=g))
        if kind == "solid":
            out[i, :, y0:y0 + s, x0:x0 + s] = torch.rand(c, 1, 1, generator=g)
        else:
            out[i, :, y0:y0 + s, x0:x0 + s] = torch.rand(c, s, s, generator=g)
    return out


def input_sets(imgs):
    x = imgs.test
    g = torch.Generator(device="cpu").manual_seed(0)
    sets = {"clean": x, "noise.1": (x + torch.randn(x.shape, generator=g) * .1).clamp(0, 1),
            "uniform": torch.rand(x.shape, generator=g)}
    for kind in ("solid", "noisepatch"):
        for s in (1, 2):
            sets[f"patch:{kind}:{s}"] = random_patches(x, kind, 100 * s + (kind == "solid"))
    return sets


@torch.no_grad()
def measure(model, sets_n, K):
    out, preds = {}, {}
    for name, xn in sets_n.items():
        z = M.logits(model, xn)
        p = z.argmax(1)
        top2 = z.topk(2, 1).values
        other = torch.where(z == top2[:, :1], top2[:, 1:2], top2[:, :1])
        marg = (z - other).mean(0)
        preds[name] = p
        out[name] = {"count": torch.bincount(p, minlength=K).tolist(), "margin": marg.tolist()}
    for name in sets_n:
        if name.startswith("patch:"):
            base, p = preds["clean"], preds[name]
            out[name]["flip"] = [int(((base != k) & (p == k)).sum()) for k in range(K)]
            out[name]["n_not"] = [int((base != k).sum()) for k in range(K)]
    return out


def run(job, dev):
    _, key, rec = job["id"].split("::")
    r = cohort.row(key)
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load(r.dataset)
    qcb = r.group == "QCB"
    target = int(r.target) if qcb else None
    marg = f_margins()[key]
    cand = int(np.argmax(marg))
    sets_n = {k: norm(v.to(dev)) for k, v in input_sets(imgs).items()}
    test = sets_n["clean"]
    trig = None
    if qcb:
        keep = imgs.test_y != target
        trig = norm(trigger(imgs.test[keep].to(dev), r.trigger))
    head = {"model_key": key, "group": r.group, "cell": r.cell, "family": r.family, "recipe": rec, "target": target,
            "candidate": cand, "cand_is_target": qcb and cand == target, "K": K}
    rows = []
    f_pred = M.logits(F, test).argmax(1)

    def add(state, model):
        with torch.no_grad():
            p = M.logits(model, test).argmax(1)
            row = head | {"state": state, "acc": float((p.cpu() == imgs.test_y).float().mean()),
                          "agree_F": float((p == f_pred).float().mean()), "sets": measure(model, sets_n, K)}
            if trig is not None:
                row["asr"] = float((M.logits(model, trig).argmax(1) == target).float().mean())
        rows.append(row)

    add("F", F)
    DS = deploy._ds()
    vk = RECIPES[rec]["view_key"]
    q = DS.deployed(F, vk, norm(imgs.cal.to(dev)), train_x=deploy.train_arg(rec, imgs, norm), device=str(dev)).eval()
    add("Q", q)
    del q
    if job.get("first"):
        dfu = E21.defender_u8(r.dataset)
        idef = dataclasses.replace(imgs, cal=data.to01(dfu), train_u8=np.concatenate([dfu] * 4))
        rep, _, _ = grids.replica(F, "iao8", dev)
        dm = C.build(rep, "iao8", idef, norm, dev, bn="stored")
        C.set_acts(C.act_quantizers(dm), "clip:1")
        add("C1", dm)
        del dm, rep
        m = E21.bn_reest(F, E21.batches(dfu, norm, dev))
        add("Bfp:test", m)
        del m
        m = E21.bn_reest(F, E21.batches(imgs.train_u8, norm, dev, n=E21.N_TRAIN_A, aug=True))
        add("Bfp:aug", m)
        del m
    return rows
