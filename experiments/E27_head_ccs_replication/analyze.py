#!/usr/bin/env python3
"""E27 readout (statistics fixed in prediction.md).   python3 analyze.py <run id>

Per model (recipe iao4) and start (linear/ones, linear/mean): r_k = ccs_k(Q) / ccs_k(F) and, for a candidate c,
z = (median_{k != c} r - r_c) / MAD_{k != c}(r). Candidates: MM-BD margin argmax in F (primary), the fastest class in F at
the same start (z_fast), every class (z_max: the largest z and its class); the true target only as a diagnostic.
Threshold T = 5 (fixed beforehand).
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402

N_JOBS = 125
T = 5.0
CFGS = ("linear/ones", "linear/mean")
HARD = ("QAT-W4A4-IAO", "CleanParent-QuEST")


def zc(rr, c):
    o = np.delete(rr, c)
    med = np.median(o)
    return float((med - rr[c]) / (np.median(np.abs(o - med)) + 1e-9))


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + .5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def e07_candidates():
    spec = importlib.util.spec_from_file_location("e23", ROOT / "experiments/E23_candidate_differential/exp.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return {k: int(np.argmax(v)) for k, v in m.f_margins().items()}


def frame(run):
    by = {}
    for r in R.rows(run):
        by.setdefault(r["model_key"], {})[r["state"]] = r
    out = []
    for key, s in by.items():
        if "F" not in s or "Q" not in s:
            continue
        f, q = s["F"], s["Q"]
        cm, t = f["candidate_mmbd"], f["target"]
        hard = next((h for h in HARD if h in f["cell"]), None)
        base = {"model_key": key, "batch": f["batch"], "dataset": f["dataset"], "group": f["group"], "cell": f["cell"],
                "qcb": f["group"] == "QCB", "anchor": f["batch"] == "batch1", "hard": hard, "target": t, "cand_mmbd": cm,
                "K": f["K"], "mmbd_raw": f["mmbd"].get("raw_score"), "mmbd_alarm": f["mmbd"].get("native_alarm")}
        for cfg in CFGS:
            cf, cq = np.asarray(f["res"][cfg]["ccs"], float), np.asarray(q["res"][cfg]["ccs"], float)
            rr = cq / np.maximum(cf, 1)
            zs = np.array([zc(rr, k) for k in range(len(rr))])
            fast = int(np.argmin(cf))
            tag = cfg.split("/")[1]
            base |= {f"{tag}_z_mmbd": zs[cm] if cm is not None else np.nan, f"{tag}_fast": fast, f"{tag}_z_fast": zs[fast],
                     f"{tag}_z_max": float(zs.max()), f"{tag}_z_max_cls": int(zs.argmax()),
                     f"{tag}_z_target": zs[t] if t is not None else np.nan, f"{tag}_r_cand": float(rr[cm]) if cm is not None else np.nan,
                     f"{tag}_ccsF_cand": cf[cm] if cm is not None else np.nan, f"{tag}_ccsQ_cand": cq[cm] if cm is not None else np.nan,
                     f"{tag}_ccsQ_med_others": float(np.median(np.delete(cq, cm))) if cm is not None else np.nan,
                     f"{tag}_unconv_F": int((~np.asarray(f["res"][cfg]["converged"])).sum()),
                     f"{tag}_unconv_Q": int((~np.asarray(q["res"][cfg]["converged"])).sum())}
        out.append(base)
    return pd.DataFrame(out)


def frac(m, n):
    return f"{m}/{n}"


def main(run):
    d = frame(run)
    lines = [f"# E27 tables (run `{run}`, {len(d)}/{N_JOBS} models)", ""]
    qn = d[d.qcb & ~d.anchor]
    qa = d[d.qcb & d.anchor]
    nn_ = d[~d.qcb]

    # P0 anchors
    e24 = pd.read_csv(ROOT / "experiments/E24_convergence_by_layer/results/per_cfg.csv")
    e24 = e24[(e24.recipe == "iao4") & (e24.cfg == "linear/ones")].set_index("model_key")
    e07 = e07_candidates()
    lines += ["## P0 anchors (batch1)", "", "| model | z_mmbd E27 | r_z E24 | cand E27 | cand E07 |", "|---|---|---|---|---|"]
    same = tot = 0
    for _, r in d[d.anchor].iterrows():
        if r.model_key in e07:
            tot += 1
            same += int(e07[r.model_key] == r.cand_mmbd)
        if r.model_key in e24.index:
            lines.append(f"| {r.model_key} | {r.ones_z_mmbd:.2f} | {e24.loc[r.model_key, 'r_z']:.2f} | {r.cand_mmbd} | {e07.get(r.model_key)} |")
    lines += ["", f"MM-BD candidate equal to E07: {frac(same, tot)}", ""]

    # P1, P2, P4, P6 per dataset for new QuEST W4A4
    lines += ["## New QuEST W4A4 (17)", "",
              "| dataset | n | cand_mmbd = target | z_mmbd > 5 | fast = target | z_fast > 5 | z_max > 5 | z_max cls = target | mean z_mmbd < 0 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for ds, g in list(qn.groupby("dataset")) + [("all", qn)]:
        lines.append(f"| {ds} | {len(g)} | {(g.cand_mmbd == g.target).sum()} | {(g.ones_z_mmbd > T).sum()} | "
                     f"{(g.ones_fast == g.target).sum()} | {(g.ones_z_fast > T).sum()} | {(g.ones_z_max > T).sum()} | "
                     f"{(g.ones_z_max_cls == g.target).sum()} | {(g.mean_z_mmbd < 0).sum()} |")
    lines += ["", "| model | dataset | target | cand | z_mmbd | z_target | z_fast (cls) | z_max (cls) | ccsF cand | ccsQ cand | ccsQ others med | mean z_mmbd |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in pd.concat([qa, qn]).sort_values(["anchor", "dataset"], ascending=[False, True]).iterrows():
        lines.append(f"| {r.model_key}{' (anchor)' if r.anchor else ''} | {r.dataset} | {r.target} | {r.cand_mmbd} | {r.ones_z_mmbd:.1f} | "
                     f"{r.ones_z_target:.1f} | {r.ones_z_fast:.1f} ({r.ones_fast}) | {r.ones_z_max:.1f} ({r.ones_z_max_cls}) | "
                     f"{r.ones_ccsF_cand:.0f} | {r.ones_ccsQ_cand:.0f} | {r.ones_ccsQ_med_others:.0f} | {r.mean_z_mmbd:.1f} |")

    # P3, P4, P5 normals
    lines += ["", "## Normals and controls (104)", "",
              "| set | n | z_mmbd > 5 | max z_mmbd | z_fast > 5 | max z_fast | z_max > 5 | max z_max |", "|---|---|---|---|---|---|---|---|"]
    sets = [(f"{ds}", g) for ds, g in nn_.groupby("dataset")] + [(f"{gr}", g) for gr, g in nn_.groupby("group")] + \
           [(f"hard:{h}", nn_[nn_.hard == h]) for h in HARD] + [("all", nn_)]
    for name, g in sets:
        lines.append(f"| {name} | {len(g)} | {(g.ones_z_mmbd > T).sum()} | {g.ones_z_mmbd.max():.1f} | {(g.ones_z_fast > T).sum()} | "
                     f"{g.ones_z_fast.max():.1f} | {(g.ones_z_max > T).sum()} | {g.ones_z_max.max():.1f} |")
    lines += ["", "Normals with z_mmbd > 5 or z_fast > 5:", "", "| model | dataset | cell | cand | z_mmbd | fast | z_fast | z_max (cls) |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in nn_[(nn_.ones_z_mmbd > T) | (nn_.ones_z_fast > T)].iterrows():
        lines.append(f"| {r.model_key} | {r.dataset} | {r.cell} | {r.cand_mmbd} | {r.ones_z_mmbd:.1f} | {r.ones_fast} | {r.ones_z_fast:.1f} | "
                     f"{r.ones_z_max:.1f} ({r.ones_z_max_cls}) |")

    # P7 AUROC
    lines += ["", "## AUROC (QuEST W4A4 vs same-dataset normals and controls)", "",
              "| dataset | z_mmbd (new) | z_mmbd (all) | z_fast (all) | z_max (all) | mean z_mmbd, lower = QCB (all) |", "|---|---|---|---|---|---|"]
    for ds in sorted(d.dataset.unique()):
        n = nn_[nn_.dataset == ds]
        a = d[d.qcb & (d.dataset == ds)]
        b = qn[qn.dataset == ds]
        lines.append(f"| {ds} | {auroc(b.ones_z_mmbd, n.ones_z_mmbd):.2f} | {auroc(a.ones_z_mmbd, n.ones_z_mmbd):.2f} | "
                     f"{auroc(a.ones_z_fast, n.ones_z_fast):.2f} | {auroc(a.ones_z_max, n.ones_z_max):.2f} | "
                     f"{auroc(-a.mean_z_mmbd, -n.mean_z_mmbd):.2f} |")

    # falsification
    hard = nn_[nn_.hard.notna()]
    lines += ["", "## Falsification criteria", "",
              f"1. new QuEST W4A4 with z_mmbd > 5: {frac((qn.ones_z_mmbd > T).sum(), len(qn))} (fails if <= 8 of 17)",
              f"2. normals with z_mmbd > 5: {frac((nn_.ones_z_mmbd > T).sum(), len(nn_))} (fails if > 10); hard: "
              f"{frac((hard.ones_z_mmbd > T).sum(), len(hard))} (fails if >= 3)",
              f"3. anchors with z_mmbd <= 5: {(qa.ones_z_mmbd <= T).sum()}; MM-BD candidate != E07: {tot - same} of {tot} (fails if >= 3)",
              "", f"Unconverged classes (ones, Q) in any model: {int((d.ones_unconv_Q > 0).sum())} models; (ones, F): {int((d.ones_unconv_F > 0).sum())}"]
    (HERE / "results").mkdir(exist_ok=True)
    d.to_csv(HERE / "results/per_model.csv", index=False)
    (HERE / "analysis_tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
