"""E13 (Research_Plan 실험 8, 가설 I; parts 8-1, 8-3, 8-4): how much does the backdoor's switch depend on implementation details?

Models: batch 1 (57 distinct). Two job kinds.

T::<key>  (GPU; 8-1 firing matrix + 8-3 rounding-direction agreement)
  Conditions: the 7 deploysim recipes (Q exactly as E01-E07) and the 22 named qlib tool recipes (MVD
  scripts/1121poc_claude/qlib.py RECIPES without the deploysim delegates and without torch_fx_x86, whose spec is identical
  to torch_fx_fbgemm), each built from F with the 1,000 quant_cal images (qlib uses the first cal_n of them).
  Behaviour on test 0-999: clean accuracy, top-1 agreement with F, ASR (QCB trigger, non-target images).
  Weights: residual of every Conv2d / Linear in the ORIGINAL weight space (folded residuals divided by the BN factor kf;
  layers a tool leaves in float have residual 0). sign(residual) per condition; for every pair of conditions the share of
  equal signs among weights where both residuals are non-zero, and that support's share. DoReFa rescales every weight,
  so its 'residual' is not a rounding decision: kept in 8-1, flagged in 8-3.

R::<key>::<backend>  (CPU; 8-4 simulation vs real PyTorch int8, backend fbgemm | qnnpack)
  Real: torch.ao FX prepare_fx(get_default_qconfig_mapping(backend)) on F, calibration on the 1,000 quant_cal images
  (batches of 128), convert_fx, int8 kernels on CPU. Simulation: deploysim fbgemm8 / qnnpack8 (CPU).
  Weight codes: int_repr of every quantized Conv/Linear vs the deploysim grid codes of the same layer (count and max of
  differing codes, scale relative difference). Behaviour: accuracy, ASR, top-1 agreement real vs simulated (clean, trigger).
"""
from __future__ import annotations

import copy
import importlib.util
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, data, deploy, env, grids, measures as M  # noqa: E402
from qcbobs.models import ALL_RECIPES, load, trigger  # noqa: E402

BACKENDS = ("fbgemm", "qnnpack")
SIM_OF = {"fbgemm": "fbgemm8", "qnnpack": "qnnpack8"}
_Q = None


def qlib():
    global _Q
    if _Q is None:
        env.setup()
        spec = importlib.util.spec_from_file_location("qlib", env.MVD / "scripts/1121poc_claude/qlib.py")
        _Q = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_Q)
    return _Q


def tools():
    return sorted(k for k in qlib().RECIPES if not k.startswith("deploysim:") and k != "torch_fx_x86")


def models():
    d = cohort.load_all()
    return d[d.batch == "batch1"]


def jobs():
    out = []
    for _, r in models().iterrows():
        out.append({"id": f"T::{r.model_key}", "priority": 0 if r.group == "QCB" else 1})
        for b in BACKENDS:
            out.append({"id": f"R::{r.model_key}::{b}", "priority": 2})
    return out


def _setup(key, dev):
    r = cohort.row(key)
    F, norm, info = load(r.data_dir, dev)
    imgs = data.load(r.dataset)
    qcb = r.group == "QCB"
    return r, F, norm, int(info["num_classes_w"]), imgs, qcb, (int(r.target) if qcb else None)


def _behaviour(model, test, yt, trig, target, K, f_pred):
    aa = M.acc_asr(model, test, yt, trig, target, K)
    pred = M.logits(model, test).argmax(1)
    return {"acc": aa["acc"], "asr": aa.get("asr"), "agree_F": float((pred == f_pred).float().mean())}


@torch.no_grad()
def _signs(res: dict, names) -> torch.Tensor:
    return torch.cat([torch.sign(res[n]).flatten().to(torch.int8) for n in names])


def run_T(key, dev):
    r, F, norm, K, imgs, qcb, target = _setup(key, dev)
    test = norm(imgs.test.to(dev))
    trig = norm(trigger(imgs.test.to(dev), r.trigger)) if qcb else None
    f_pred = M.logits(F, test).argmax(1)
    fl = [(n, m) for n, m in F.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    names = [n for n, _ in fl]
    WF = {n: m.weight.detach().clone() for n, m in fl}
    env.setup()
    from multiview_detection.deployed import conv_bn_pairs
    bn = dict(conv_bn_pairs(F))
    fmods = dict(F.named_modules())
    kf = {n: (fmods[bn[n]].weight / torch.sqrt(fmods[bn[n]].running_var + fmods[bn[n]].eps)).detach().view(-1, 1, 1, 1)
          for n in names if n in bn}
    head = {"model_key": key, "group": r.group, "cell": r.cell, "family": r.family, "target_recipe": r.target_recipe}
    rows, sig, cond_names = [], [], []
    for rec in ALL_RECIPES:
        t0 = time.time()
        rep, layers, _ = grids.replica(F, rec, dev)
        omap = grids.orig_map(F, layers)
        gs = grids.grids(layers, rec)
        Wr = {n: m.weight.detach().clone() for n, m in layers}
        Q0 = {lg.name: lg.quant(Wr[lg.name]) for lg in gs}
        res = {}
        for (n, _), (fn, k) in zip(layers, omap):
            d = Q0[n] - Wr[n]
            res[fn] = d / k if k is not None else d
        deploy.set_weights(layers, Q0)
        dm = deploy.build(rep, rec, imgs, norm, dev)
        rows.append(head | {"kind": "cond", "cond": f"ds:{rec}", "source": "deploysim", "n_float_layers": 0, "graph": "deploysim",
                            "sec": time.time() - t0} | _behaviour(dm, test, imgs.test_y, trig, target, K, f_pred))
        sig.append(_signs(res, names)); cond_names.append(f"ds:{rec}")
        del dm, rep
        torch.cuda.empty_cache()
    Q = qlib()
    cal = norm(imgs.cal.to(dev))
    for t in tools():
        t0 = time.time()
        try:
            dm = Q.deployed(F, t, cal, device=str(dev))
        except Exception as exc:  # noqa: BLE001  recorded, the job goes on
            rows.append(head | {"kind": "cond", "cond": f"ql:{t}", "source": "qlib", "error": repr(exc)[:300]})
            continue
        qi = dm.qlib_info
        floats = set(qi.get("weights_float", []))
        dmods = dict(dm.named_modules())
        res = {}
        for n in names:
            if n in floats or n not in dmods:
                res[n] = torch.zeros_like(WF[n])
                continue
            wq = dmods[n].weight.detach()
            folded = qi.get("bn_folded", 0) and n in kf
            res[n] = (wq / kf[n] - WF[n]) if folded else (wq - WF[n])
        rows.append(head | {"kind": "cond", "cond": f"ql:{t}", "source": "qlib", "tier": Q.RECIPES[t]["tier"],
                            "graph": qi.get("graph"), "n_float_layers": len(floats), "bn_folded": qi.get("bn_folded"),
                            "sec": time.time() - t0} | _behaviour(dm, test, imgs.test_y, trig, target, K, f_pred))
        sig.append(_signs(res, names)); cond_names.append(f"ql:{t}")
        del dm
        torch.cuda.empty_cache()
    S = torch.stack(sig)                      # [C, n_weights] int8
    nz = S != 0
    C = len(cond_names)
    agree = [[None] * C for _ in range(C)]
    support = [[None] * C for _ in range(C)]
    for i in range(C):
        for j in range(i, C):
            both = nz[i] & nz[j]
            nb = int(both.sum())
            a = float((S[i][both] == S[j][both]).float().mean()) if nb else None
            agree[i][j] = agree[j][i] = a
            support[i][j] = support[j][i] = nb / S.shape[1]
    rows.append(head | {"kind": "agreement", "conds": cond_names, "agree": agree, "support": support,
                        "dorefa": [c for c in cond_names if "dorefa" in c]})
    return rows


def run_R(key, backend, dev):
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    cpu = torch.device("cpu")
    r, F, norm, K, imgs, qcb, target = _setup(key, cpu)
    torch.backends.quantized.engine = backend
    test = norm(imgs.test)
    trig = norm(trigger(imgs.test, r.trigger)) if qcb else None
    cal = norm(imgs.cal)
    head = {"model_key": key, "group": r.group, "cell": r.cell, "family": r.family, "backend": backend, "sim": SIM_OF[backend]}
    t0 = time.time()
    prepared = prepare_fx(copy.deepcopy(F).eval(), get_default_qconfig_mapping(backend), example_inputs=(cal[:1],))
    with torch.no_grad():
        for s in range(0, len(cal), 128):
            prepared(cal[s:s + 128])
    real = convert_fx(prepared).eval()
    t_real = time.time() - t0
    # simulated deployment (deploysim) and its grid codes
    rec = SIM_OF[backend]
    rep, layers, _ = grids.replica(F, rec, cpu)
    omap = grids.orig_map(F, layers)
    gs = grids.grids(layers, rec)
    Wr = {n: m.weight.detach().clone() for n, m in layers}
    codes_sim = {fn: lg.codes(lg.quant(Wr[lg.name])) for lg, (fn, _) in zip(gs, omap)}
    scale_sim = {fn: lg.scale for lg, (fn, _) in zip(gs, omap)}
    deploy.set_weights(layers, {lg.name: lg.quant(Wr[lg.name]) for lg in gs})
    sim = deploy.build(rep, rec, imgs, norm, cpu)
    per_layer, n_diff, n_all, worst = [], 0, 0, 0
    rmods = dict(real.named_modules())
    for fn in codes_sim:
        m = rmods.get(fn)
        if m is None or not hasattr(m, "weight") or not callable(m.weight):
            per_layer.append({"layer": fn, "status": "not quantized in real model"})
            continue
        qw = m.weight()
        cr = qw.int_repr().float()
        cs = codes_sim[fn].float().view_as(cr)
        d = (cr - cs).abs()
        if qw.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric):
            sr = qw.q_per_channel_scales().float()
        else:
            sr = torch.tensor([qw.q_scale()])
        ss = scale_sim[fn].flatten().float()
        per_layer.append({"layer": fn, "n": int(cr.numel()), "n_diff": int((d > 0).sum()), "max_diff": float(d.max()),
                          "scale_rel_diff": float(((sr - ss) / ss).abs().max())})
        n_diff += int((d > 0).sum()); n_all += int(cr.numel()); worst = max(worst, float(d.max()))

    def preds(model, x):
        with torch.no_grad():
            return torch.cat([model(x[s:s + 128]) for s in range(0, len(x), 128)]).argmax(1)
    out = {}
    for name, model in (("real", real), ("sim", sim)):
        pc = preds(model, test)
        o = {"acc": float((pc == imgs.test_y).float().mean()), "pc": pc}
        if trig is not None:
            pt = preds(model, trig)
            keep = imgs.test_y != target
            o |= {"asr": float((pt[keep] == target).float().mean()), "pt": pt}
        out[name] = o
    row = head | {"kind": "real", "sec_real_build": t_real, "codes_n": n_all, "codes_diff": n_diff, "codes_max_diff": worst,
                  "layers": per_layer, "acc_real": out["real"]["acc"], "acc_sim": out["sim"]["acc"],
                  "agree_clean": float((out["real"]["pc"] == out["sim"]["pc"]).float().mean())}
    if trig is not None:
        row |= {"asr_real": out["real"]["asr"], "asr_sim": out["sim"]["asr"],
                "agree_trig": float((out["real"]["pt"] == out["sim"]["pt"]).float().mean())}
    return [row]


def run(job, dev):
    parts = job["id"].split("::")
    if parts[0] == "T":
        return run_T(parts[1], dev)
    return run_R(parts[1], parts[2], dev)
