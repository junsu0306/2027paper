"""ST12 (2부): format-free switch states x MM-BD (trigger-shape-free induction), on/off differential per class.

States (all built by the defender without the attack's format, with its 2,000 unlabelled images; E10 / ST11 recipes):
  F     as delivered (stored BN)                                                         (off for every QCB family)
  Bfp   BN statistics re-estimated in FP32 on the defender images (E10 Bfp)              (on: Ma-8, Tian; E10)
  C1    activation clipping at the calibrated range, iao8 placement, BN stored (E10 C1)   (on: QuEST-8 CIFAR-10; E06/E10)
  N     BN and ranges re-estimated under activation noise t 2 (4-bit-step random-sign residual, ST08), clean inference
        (= ST11 S2c)                                                                     (on: QuEST W4A4; ST11)
MM-BD (MVD baselines.run_mmbd: per class 30 uniform images, 300 plain SGD steps lr .01, margin = z_c - max_{k!=c} z_k,
per-class relative-tolerance stop 1e-5, max over the 30 images) re-implemented with the classes batched (identical
per-image gradients; P0 checks the equivalence). The margin trajectory is kept at TRAJ steps (convergence speed).
Per state also: clean accuracy and agreement with F on test 0-999, ASR (QCB, mechanism only), and per class on the
defender's labelled images: mean clean margin of the class's images and mean probability mass (strong-class predictors).
"""
from __future__ import annotations

import copy
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, components as C, data, deploy, grids, measures as M  # noqa: E402
from qcbobs.models import load, trigger  # noqa: E402

MMBD = dict(n_images=30, steps=300, lr=.01, tol=1e-5, seed=0)
CHUNK = 20
TRAJ = (10, 25, 50, 100, 150, 200, 300)
STATES = ("F", "Bfp", "C1", "N")


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ST11 = _mod("st11", "experiments/ST11_class_span/exp.py")
ST09, ST10, ST08 = ST11.ST09, ST11.ST10, ST11.ST08
E10 = ST08.E10


def models():
    d = cohort.load_all(with_h=True)
    d = d[(d.arch == "VGG16-BN") | ((d.arch == "ResNet18") & d.dataset.isin(["CIFAR-10", "CIFAR-100", "GTSRB"]))]
    return d[d.batch != "batch1"].reset_index(drop=True)


def jobs():
    pri = {"QCB": 0, "control-nopoison": 0, "normal-hard": 1, "normal-light": 1, "normal-standard": 1}
    return [{"id": f"M::{r.model_key}", "priority": pri[r.group] + (1 if r.dataset == "CIFAR-100" else 0)}
            for r in models().itertuples()]


class Pix(nn.Module):
    def __init__(self, model, norm):
        super().__init__()
        self.model, self.norm = model, norm

    def forward(self, x):
        return self.model(self.norm(x))


def mmbd_vec(pix, K, dev):
    g = torch.Generator(device="cpu").manual_seed(MMBD["seed"])
    n = MMBD["n_images"]
    init = [torch.rand(n, 3, 32, 32, generator=g) for _ in range(K)]
    margins, steps, traj = np.zeros(K), np.zeros(K, dtype=int), np.full((K, len(TRAJ)), np.nan)
    for c0 in range(0, K, CHUNK):
        cls = list(range(c0, min(K, c0 + CHUNK)))
        m = len(cls)
        x = torch.cat([init[c] for c in cls]).to(dev)
        lab = torch.tensor(cls, device=dev).repeat_interleave(n)
        oh = Fn.one_hot(lab, K).float()
        prev = np.full(m, 1000.0)
        active = np.ones(m, dtype=bool)
        last = np.zeros(m)
        for step in range(MMBD["steps"]):
            x.requires_grad_(True)
            out = pix(x.clamp(0, 1))
            other = (1 - oh) * out - 1000 * oh
            per_img = -(out * oh).sum(1) + other.max(1).values
            per_cls = per_img.view(m, n).sum(1)
            act_t = torch.as_tensor(active, device=dev)
            grad = torch.autograd.grad((per_cls * act_t).sum(), x)[0]
            with torch.no_grad():
                mc = (-per_img).view(m, n).max(1).values.double().cpu().numpy()
                last[active] = mc[active]
                if step + 1 in TRAJ:
                    traj[c0:c0 + m, TRAJ.index(step + 1)] = np.where(active, mc, np.nan)
                x = (x - MMBD["lr"] * grad).detach()
            cur = per_cls.detach().double().cpu().numpy()
            den = np.abs(prev)
            rel = np.where(den > 0, np.abs(prev - cur) / np.where(den > 0, den, 1), np.inf)
            newly = active & (rel < MMBD["tol"])
            steps[c0:c0 + m][newly] = step + 1
            active &= ~newly
            prev = np.where(active, cur, prev)
            if not active.any():
                break
        steps[c0:c0 + m][active] = MMBD["steps"]
        margins[c0:c0 + m] = last
    for c in range(K):          # stopped classes keep their final margin on later checkpoints
        for j in range(len(TRAJ)):
            if np.isnan(traj[c, j]) and TRAJ[j] >= steps[c]:
                traj[c, j] = margins[c]
    return margins, steps, traj


@torch.no_grad()
def class_stats(model, xd, yd, norm, dev, K):
    lo = M.logits(model, norm(xd.to(dev))).float()
    p = Fn.softmax(lo, 1)
    own = lo.gather(1, yd.to(dev)[:, None])[:, 0]
    oth = lo.scatter(1, yd.to(dev)[:, None], -1e9).max(1).values
    mg = (own - oth).cpu().numpy()
    yy = yd.numpy()
    return ([float(mg[yy == c].mean()) if (yy == c).any() else float("nan") for c in range(K)], p.mean(0).cpu().tolist())


def run(job, dev):
    _, key = job["id"].split("::")
    r = cohort.row(key)
    t0 = time.time()
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load(r.dataset)
    idef = E10.defender_images(r.dataset)
    xd, yd = ST10.defender_labelled(r.dataset)
    test = norm(imgs.test.to(dev))
    qcb = r.group == "QCB"
    trig = norm(trigger(imgs.test.to(dev), r.trigger)) if qcb else None
    target = int(r.target) if qcb else None
    for p in F.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        f_pred = M.logits(F, test).argmax(1)
    head = {"model_key": key, "batch": r.batch, "group": r.group, "cell": r.cell, "family": r.family, "dataset": r.dataset,
            "arch": r.arch, "target": target, "K": K}
    lin = [m for m in F.modules() if isinstance(m, nn.Linear)][-1]
    rows = [head | {"part": "head", "w_norm": lin.weight.norm(dim=1).tolist(),
                    "bias": (lin.bias.tolist() if lin.bias is not None else [0.0] * K)}]

    def state_model(s):
        if s == "F":
            return F, None
        if s == "Bfp":
            Fb = copy.deepcopy(F)
            for _, mod in C.bn_layers(Fb):
                mod.reset_running_stats()
                mod.momentum = None
            Fb.train()
            with torch.no_grad():
                x = norm(idef.cal.to(dev))
                for s0 in range(0, len(x), 128):
                    Fb(x[s0:s0 + 128])
            return Fb.eval(), None
        if s == "C1":
            rep, _, _ = grids.replica(F, "iao8", dev)
            dm = C.build(rep, "iao8", idef, norm, dev, bn="stored")
            C.set_acts(C.act_quantizers(dm), "clip:1")
            return dm.eval(), None
        rep, _, _ = grids.replica(F, "iao4", dev)
        with ST08.Noise(2.0, torch.Generator(device=dev).manual_seed(7000), clamp=False):
            dm = deploy.build(rep, "iao4", idef, norm, dev)
        return dm.eval(), "identity"

    for s in STATES:
        ts = time.time()
        model, ctx = state_model(s)
        for p in model.parameters():
            p.requires_grad_(False)

        def go():
            aa = M.acc_asr(model, test, imgs.test_y, trig, target, K)
            with torch.no_grad():
                agree = float((M.logits(model, test).argmax(1) == f_pred).float().mean())
            cm, pm = class_stats(model, xd, yd, norm, dev, K)
            torch.manual_seed(0)
            mg, st, tr = mmbd_vec(Pix(model, norm).to(dev).eval(), K, dev)
            return aa, agree, cm, pm, mg, st, tr
        if ctx == "identity":
            with ST09.Identity():
                aa, agree, cm, pm, mg, st, tr = go()
        else:
            aa, agree, cm, pm, mg, st, tr = go()
        rows.append(head | {"part": "state", "state": s, "acc": aa["acc"], "asr": aa.get("asr"), "agree_F": agree,
                            "clean_margin": cm, "pmass": pm, "margins": mg.tolist(), "steps": st.tolist(),
                            "traj": [[None if np.isnan(v) else float(v) for v in row] for row in tr], "sec": time.time() - ts})
        if model is not F:
            del model
    rows.append(head | {"part": "time", "sec": time.time() - t0})
    return rows
