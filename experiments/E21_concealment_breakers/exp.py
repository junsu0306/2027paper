"""E21: what breaks the concealment? (prediction.md)

A::<model>  batch1 + batch2 (CIFAR-10 ResNet18): BN statistics re-estimated in FP32 on different data (test / train / aug,
            amounts, EMA) and per stage; staleness distance stored vs re-estimated (aug).
B::<model>  QuEST W4A4 (all datasets) + a set of normals: the iao4 train-mode deployment rebuilt here with one detail
            changed at a time (activation range estimator, calibration data amount / source / order / batch, BN handling,
            pools, bit widths, one stage at W4A4).
Measures per state: clean accuracy, clean agreement with F (A) or with the reference iao4 (B), ASR, and the concealment
margin = median over triggered non-target test images of (logit of the true label - logit of the target).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, components as C, data, deploy, measures as M  # noqa: E402
from qcbobs.env import DATA  # noqa: E402
from qcbobs.models import load, trigger  # noqa: E402

DEF_RANGE = (2000, 4000)
N_TRAIN_A = 10_000
STAGES = ("stem", "layer1", "layer2", "layer3", "layer4", "head")


def group_of(name: str) -> str:
    parts = name.split(".")
    if parts[0].startswith("layer"):
        return parts[0]
    if parts[0] == "blocks" and len(parts) > 1 and parts[1].isdigit():
        return f"layer{int(parts[1]) // 2 + 1}"
    return "head" if parts[0] in ("linear", "fc", "classifier") else "stem"


def a_models():
    d = cohort.load_all()
    return d[d.batch.isin(["batch1", "batch2"])]


def b_models():
    d = cohort.load_all()
    q = d[(d.group == "QCB") & d.cell.str.contains("QuEST-W4A4")]
    b1 = d[(d.batch == "batch1") & (d.group != "QCB")]
    n1 = pd_concat([b1[b1.cell == "light"], *[g.head(2) for c, g in b1.groupby("cell") if c.startswith("standard")],
                    b1[b1.cell.isin(["control-QuEST-IAO-W4A4", "control-QuEST-IAO-W8A8"])]])
    xn = d[(d.batch == "batch_xn") & d.dataset.isin(["CIFAR-100", "GTSRB"])]
    n2 = xn[xn.cell.str.contains("light|standard|QAT-W4A4-IAO|control-QuEST")]
    return pd_concat([q, n1, n2])


def pd_concat(parts):
    import pandas as pd
    return pd.concat(parts, ignore_index=True)


def jobs():
    out = [{"id": f"A::{r.model_key}", "priority": 0} for _, r in a_models().iterrows()]
    out += [{"id": f"B::{r.model_key}", "priority": 1 if r.group == "QCB" else 2} for _, r in b_models().iterrows()]
    return out


# ------------------------------------------------------------------------------------------------------------- data
_DEF = {}


def defender_u8(dataset):
    if dataset not in _DEF:
        name = data.DS_FILE[dataset]
        z = np.load(DATA / "datasets/main" / f"{name}.npz")
        roles = json.loads((DATA / "datasets/main/manifests" / name / "split_base.json").read_text())["datasets"][name]["roles"]
        _DEF[dataset] = z["x_test"][np.asarray(roles["final_behavior_test"]["indices"][DEF_RANGE[0]:DEF_RANGE[1]])]
    return _DEF[dataset]


def augment(x01, gen):
    """4-pixel zero-padded random crop + horizontal flip (the training augmentation), per image."""
    n, _, h, w = x01.shape
    p = torch.nn.functional.pad(x01, (4, 4, 4, 4))
    ox = torch.randint(0, 9, (n,), generator=gen, device=x01.device)
    oy = torch.randint(0, 9, (n,), generator=gen, device=x01.device)
    flip = torch.rand(n, generator=gen, device=x01.device) < .5
    out = torch.stack([p[i, :, oy[i]:oy[i] + h, ox[i]:ox[i] + w] for i in range(n)])
    out[flip] = out[flip].flip(-1)
    return out


def batches(u8, norm, dev, n=None, seed=0, bs=128, aug=False, repeat=1):
    """Shuffled batches of normalised images (deploysim._train_batches order when n=None, seed=0, bs=128, no aug)."""
    src = np.concatenate([u8] * repeat) if repeat > 1 else u8
    perm = np.random.default_rng(seed).permutation(len(src))
    if n is not None:
        perm = perm[:n]
    gen = torch.Generator(device=dev).manual_seed(1000 + seed)
    for s in range(0, len(perm), bs):
        x = data.to01(src[perm[s:s + bs]]).to(dev)
        if aug:
            x = augment(x, gen)
        yield norm(x)


@torch.no_grad()
def bn_reest(F, batch_iter, momentum=None, only=None, reset=True):
    """Copy of F with BN running stats re-estimated (cumulative if momentum is None). only: set of stage names whose BN
    get the new statistics (the others keep the stored ones). reset=False: EMA update starting from the stored stats
    (the usual PyTorch re-calibration)."""
    m = copy.deepcopy(F)
    stored = C.bn_stats(m)
    for _, mod in C.bn_layers(m):
        if reset:
            mod.reset_running_stats()
        mod.momentum = momentum
    m.train()
    for x in batch_iter:
        m(x)
    m.eval()
    if only is not None:
        for n, mod in C.bn_layers(m):
            if group_of(n) not in only:
                mod.running_mean.copy_(stored[n][0])
                mod.running_var.copy_(stored[n][1])
    for _, mod in C.bn_layers(m):
        mod.momentum = 0.1
    return m


def staleness(F, R):
    out = []
    a, b = C.bn_stats(F), C.bn_stats(R)
    for n in a:
        mu_s, var_s = a[n]
        mu_r, var_r = b[n]
        out.append({"name": n, "stage": group_of(n),
                    "d_mean": float(((mu_s - mu_r).abs() / (var_r + 1e-5).sqrt()).mean()),
                    "d_var": float(((var_s + 1e-5) / (var_r + 1e-5)).log().abs().mean())})
    return out


# ------------------------------------------------------------------------------------------------------------- measures
class Meter:
    def __init__(self, head, imgs, norm, dev, trig_name, target, K):
        self.head, self.rows = head, []
        self.test = norm(imgs.test.to(dev))
        self.y = imgs.test_y
        self.K, self.target = K, target
        if target is not None:
            keep = imgs.test_y != target
            self.trig = norm(trigger(imgs.test[keep].to(dev), trig_name))
            self.trig_y = imgs.test_y[keep].to(dev)
        else:
            self.trig = None
        self.ref_pred = None

    @torch.no_grad()
    def __call__(self, state, model, **extra):
        p = M.logits(model, self.test).argmax(1)
        r = self.head | {"state": state, "acc": float((p.cpu() == self.y).float().mean())} | extra
        if self.ref_pred is not None:
            r["agree_ref"] = float((p == self.ref_pred).float().mean())
        if self.trig is not None:
            z = M.logits(model, self.trig)
            pt = z.argmax(1)
            r["asr"] = float((pt == self.target).float().mean())
            r["true_share"] = float((pt == self.trig_y).float().mean())
            r["conceal"] = float((z.gather(1, self.trig_y[:, None])[:, 0] - z[:, self.target]).median())
        self.rows.append(r)
        return p


# ------------------------------------------------------------------------------------------------------------- iao4
def _ds():
    return deploy._ds()


class IaoQ:
    """deploysim.ActQ('iao', bits) with a configurable range estimator (EMA momentum, or cumulative min / max)."""

    def __new__(cls, bits, momentum=0.1, minmax=False):
        DS = _ds()

        class _Q(DS.ActQ):
            def _update(self, x):
                lo, hi = float(x.detach().min()), float(x.detach().max())
                if self.lo is None:
                    self.lo, self.hi = lo, hi
                elif minmax:
                    self.lo, self.hi = min(self.lo, lo), max(self.hi, hi)
                else:
                    self.lo = (1 - momentum) * self.lo + momentum * lo
                    self.hi = (1 - momentum) * self.hi + momentum * hi
        return _Q("iao", bits)


@torch.no_grad()
def iao_build(F, batch_iter, dev, wbits=lambda n: 4, abits=lambda n: 4, momentum=0.1, minmax=False, bn="train", pools=True):
    """deploysim.build(F, SPECS['iao4_train']) with the knobs exposed (same order of operations)."""
    DS = _ds()
    m = copy.deepcopy(F).to(dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    lay = DS.qlayers(m)
    for n, mod in lay:
        b = wbits(n)
        if b < 32:
            mod.weight.copy_(DS.iao_weight(mod.weight, b))
    targets = [(n, mod, abits(n)) for n, mod in lay]
    if pools:
        targets += [(n, mod, abits(n)) for n, mod in m.named_modules() if isinstance(mod, DS.POOLS)]
    acts = []
    for _, mod, b in targets:
        if b >= 32:
            continue
        a = IaoQ(b, momentum, minmax)
        mod.register_forward_pre_hook(a)
        a.state = "observe+quant"
        acts.append(a)
    if bn == "train":
        for mod in m.modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                mod.train()
    for x in batch_iter:
        m(x)
    m.eval()
    for a in acts:
        a.freeze()
    return m


def b_states(K):
    """(state name, kwargs for iao_build, data spec). data spec: dict for batches() plus 'src' in {train, def}."""
    S = [("ref", {}, {})]
    S += [(f"mom:{v}", {"momentum": v}, {}) for v in (0.01, 0.5, 1.0)] + [("mom:minmax", {"minmax": True}, {})]
    S += [(f"n:{v}", {}, {"n": v}) for v in (512, 2048, 10000)]
    S += [("src:def", {}, {"src": "def", "repeat": 4}), ("src:aug", {}, {"aug": True})]
    S += [(f"seed:{v}", {}, {"seed": v}) for v in (1, 2)] + [(f"bs:{v}", {}, {"bs": v}) for v in (32, 512)]
    S += [("bn:eval", {"bn": "eval"}, {}), ("bn:fp_then_eval", {"bn": "eval"}, {"prebn": True}), ("pools:off", {"pools": False}, {})]
    for w, a in ((4, 8), (8, 4), (4, 6), (6, 4), (5, 4), (4, 5)):
        S.append((f"bits:W{w}A{a}", {"wbits": (lambda n, w=w: w), "abits": (lambda n, a=a: a)}, {}))
    for st in STAGES:
        f = (lambda n, st=st: 4 if group_of(n) == st else 8)
        S.append((f"stage:{st}", {"wbits": f, "abits": f}, {}))
    return S


# ------------------------------------------------------------------------------------------------------------- run
def run(job, dev):
    kind, key = job["id"].split("::")
    r = cohort.row(key)
    F, norm, info = load(r.data_dir, dev)
    K = int(info["num_classes_w"])
    imgs = data.load(r.dataset)
    qcb = r.group == "QCB"
    target = int(r.target) if qcb else None
    head = {"model_key": key, "batch": r.batch, "group": r.group, "cell": r.cell, "family": r.family, "dataset": r.dataset,
            "target": target, "job": kind}
    meter = Meter(head, imgs, norm, dev, r.trigger if qcb else "Patch6x6", target, K)
    f_pred = meter("F", F)
    meter.ref_pred = f_pred
    if kind == "A":
        tr = imgs.train_u8
        dfu = defender_u8(r.dataset)
        aug = bn_reest(F, batches(tr, norm, dev, n=N_TRAIN_A, aug=True))
        meter("aug", aug)
        stl = staleness(F, aug)
        for name, it, mom in (("test", batches(dfu, norm, dev, seed=0), None), ("train", batches(tr, norm, dev, n=N_TRAIN_A), None),
                              ("aug_n256", batches(tr, norm, dev, n=256, aug=True), None),
                              ("aug_n1024", batches(tr, norm, dev, n=1024, aug=True), None),
                              ("test_ema", batches(dfu, norm, dev, seed=0), 0.1)):
            m = bn_reest(F, it, momentum=mom, reset=mom is None)
            meter(name, m)
            del m
        for st in STAGES[:5]:
            m = copy.deepcopy(aug)
            stored = C.bn_stats(F)
            with torch.no_grad():
                for n, mod in C.bn_layers(m):
                    if group_of(n) != st:
                        mod.running_mean.copy_(stored[n][0])
                        mod.running_var.copy_(stored[n][1])
            meter(f"stage:{st}", m)
            del m
        rows = meter.rows
        rows.append(head | {"state": "staleness", "layers": stl})
        return rows
    # ---- B
    tr = imgs.train_u8
    dfu = defender_u8(r.dataset)
    ref = iao_build(F, batches(tr, norm, dev), dev)
    meter.ref_pred = None
    ref_pred = meter("ref", ref)
    DS = _ds()
    chk = DS.deployed(F, "W4A4|iao_train", norm(imgs.cal.to(dev)), train_x=(tr, norm), device=str(dev)).eval()
    with torch.no_grad():
        x = meter.test[:256]
        diff = float((chk(x) - ref(x)).abs().max())
    meter.rows[-1]["ref_vs_deploysim_maxdiff"] = diff
    meter("deploysim", chk)
    del chk
    meter.ref_pred = ref_pred
    del ref
    for name, kw, ds in b_states(K)[1:]:
        ds = dict(ds)
        prebn = ds.pop("prebn", False)
        src = ds.pop("src", "train")
        u8 = dfu if src == "def" else tr
        base = F
        if prebn:
            base = bn_reest(F, batches(tr, norm, dev, n=N_TRAIN_A))
        m = iao_build(base, batches(u8, norm, dev, **ds), dev, **kw)
        meter(name, m)
        del m
    return meter.rows
