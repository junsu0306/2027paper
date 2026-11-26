"""ST03 (2부): what the stored BN statistics reveal, whether forging them breaks a QCB, and what remains after forging
(prediction.md).

One job = one model (all of the earlier cohorts + batch_h).
Part A (every model, label-free, defender data = test 2000-3999 unlabeled, E10):
  BN statistics re-estimated (FP32, cumulative, training augmentation, E21 'aug' procedure) on all 2,000 defender images
  (R) and on each half (Ra, Rb). Distances per BN layer (E21.staleness): d_mean = mean_c |mu_1 - mu_2| / sigma_2,
  d_var = mean_c |log var_1 / var_2|. stale = stored vs R; noise = Ra vs Rb.
  Entropy on detector_probe of F (stored statistics) and of R.
Part B (QCB and no-poison controls): the attacker forges the stored statistics with its own clean training images
  (training split, 10,000 random images, augmentation): A = re-estimated statistics. Forged model F_lam has
  mean = (1 - lam) mu_stored + lam mu_A, log var likewise, lam in {0.5, 1}. For lam in {0, 0.5, 1}:
  F_lam clean accuracy and ASR (true trigger: mechanism only), Q_lam = the model's own target recipe(s) applied to F_lam
  (accuracy, ASR), and the defender's Part-A measures on F_lam (re-estimation does not depend on the stored statistics,
  so R is reused; stale and entropy are recomputed with the forged statistics).
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, components as C, data, deploy, measures as M  # noqa: E402
from qcbobs.models import RECIPES, load, trigger  # noqa: E402

LAMS = (0.5, 1.0)
N_ATT = 10_000


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E10 = _mod("e10", "experiments/E10_switch_closure/exp.py")
E21 = _mod("e21", "experiments/E21_concealment_breakers/exp.py")


def models():
    return cohort.load_all(with_h=True)


def jobs():
    out = []
    for _, r in models().iterrows():
        pri = 0 if r.group == "QCB" else (1 if r.group.startswith("control") else 2)
        out.append({"id": f"B::{r.model_key}", "priority": pri})
    return out


def _dist(sa, sb):
    """Mean over BN layers of (d_mean, d_var) between two statistics dicts {name: (mean, var)}."""
    dm, dv, per = [], [], {}
    for n in sa:
        m1, v1 = sa[n]
        m2, v2 = sb[n]
        a = float(((m1 - m2).abs() / (v2 + 1e-5).sqrt()).mean())
        b = float(((v1 + 1e-5) / (v2 + 1e-5)).log().abs().mean())
        dm.append(a)
        dv.append(b)
        per[n] = a
    return float(np.mean(dm)), float(np.mean(dv)), per


@torch.no_grad()
def _ent(model, xp):
    lp = Fn.log_softmax(M.logits(model, xp).float(), 1)
    return float(-(lp.exp() * lp).sum(1).mean())


@torch.no_grad()
def _set_stats(model, st):
    for n, mod in C.bn_layers(model):
        mod.running_mean.copy_(st[n][0])
        mod.running_var.copy_(st[n][1])


def _mix(s0, s1, lam):
    out = {}
    for n in s0:
        m0, v0 = s0[n]
        m1, v1 = s1[n]
        lv = (1 - lam) * v0.clamp_min(1e-12).log() + lam * v1.clamp_min(1e-12).log()
        out[n] = ((1 - lam) * m0 + lam * m1, lv.exp())
    return out


def run(job, dev):
    _, key = job["id"].split("::")
    r = cohort.row(key)
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load(r.dataset)
    xp = norm(imgs.probe.to(dev))
    head = {"model_key": key, "batch": r.batch, "group": r.group, "cell": r.cell, "family": r.family, "dataset": r.dataset,
            "arch": r.arch}
    if not C.bn_layers(F):
        return [head | {"part": "none"}]
    u8 = E10.defender_images(r.dataset).train_u8[:2000]
    R = E21.bn_reest(F, E21.batches(u8, norm, dev, aug=True))
    Ra = E21.bn_reest(F, E21.batches(u8[:1000], norm, dev, aug=True))
    Rb = E21.bn_reest(F, E21.batches(u8[1000:], norm, dev, aug=True))
    s0, sR, sa, sb = C.bn_stats(F), C.bn_stats(R), C.bn_stats(Ra), C.bn_stats(Rb)
    noise_m, noise_v, _ = _dist(sa, sb)
    ent_R = _ent(R, xp)
    del Ra, Rb

    def defender(model_stats, ent_model):
        dm, dv, per = _dist(model_stats, sR)
        return {"stale_mean": dm, "stale_var": dv, "noise_mean": noise_m, "noise_var": noise_v,
                "ratio_mean": dm / max(noise_m, 1e-12), "ratio_var": dv / max(noise_v, 1e-12),
                "ent_F": ent_model, "ent_R": ent_R, "dent_rel": (ent_R - ent_model) / max(ent_model, 1e-12),
                "stale_per_layer": per}

    rows = [head | {"part": "A", "lam": 0.0} | defender(s0, _ent(F, xp))]
    if not (r.group == "QCB" or r.group.startswith("control")):
        return rows
    # ---- Part B: forging
    qcb = r.group == "QCB"
    test = norm(imgs.test.to(dev))
    target = int(r.target) if qcb else None
    trig = norm(trigger(imgs.test.to(dev), r.trigger if qcb else "Patch6x6"))
    A = E21.bn_reest(F, E21.batches(imgs.train_u8, norm, dev, n=N_ATT, aug=True, seed=7))
    sA = C.bn_stats(A)
    del A
    att_m, att_v, _ = _dist(sA, sR)
    recs = [x for x in str(r.recipes).split(",") if x in RECIPES] if qcb else []
    DS = deploy._ds()
    X = copy.deepcopy(F).eval()
    for lam in (0.0,) + LAMS:
        st = s0 if lam == 0 else _mix(s0, sA, lam)
        _set_stats(X, st)
        aa = M.acc_asr(X, test, imgs.test_y, trig, target, K)
        rec = head | {"part": "B", "lam": lam, "acc_F": aa["acc"], "asr_F": aa.get("asr"), "att_vs_def_mean": att_m,
                      "att_vs_def_var": att_v} | defender(st, _ent(X, xp))
        rec.pop("stale_per_layer")
        for rc in recs:
            q = DS.deployed(X, RECIPES[rc]["view_key"], norm(imgs.cal.to(dev)), train_x=deploy.train_arg(rc, imgs, norm),
                            device=str(dev)).eval()
            qa = M.acc_asr(q, test, imgs.test_y, trig, target, K)
            rec[f"acc_Q:{rc}"], rec[f"asr_Q:{rc}"] = qa["acc"], qa.get("asr")
            del q
        rows.append(rec)
    return rows
