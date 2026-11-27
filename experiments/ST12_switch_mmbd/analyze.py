#!/usr/bin/env python3
"""ST12 readout (rules fixed in prediction.md).   python3 analyze.py <run id>"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import env, readout as R  # noqa: E402

ON = ("Bfp", "C1", "N")
FAMS = ("Ma-8", "Ma-W4A4", "Tian-8", "Tian-W4A4", "QuANTI-8+4", "QuEST-8", "QuEST-W4A4")
FOUR = ("Ma-W4A4", "Tian-W4A4", "QuANTI-8+4", "QuEST-W4A4")
EPS = .1
ZTH = 3.5


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + .5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def rz(v):
    v = np.asarray(v, float)
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    return float((v.max() - med) / max(mad, 1e-6)), int(v.argmax())


def lg(v):
    return np.log(np.maximum(np.asarray(v, float), EPS))


def fam(row):
    if row.group != "QCB":
        return "control" if row.group == "control-nopoison" else "normal"
    for f in FAMS:
        if f in row.cell:
            return f
    return "QCB-other"


def main(run):
    env.setup()
    from multiview_detection.baselines import gamma_outlier
    d = pd.DataFrame(R.rows(run))
    st = d[d.part == "state"].copy()
    recs = []
    for mk, h in st.groupby("model_key"):
        S = {r.state: r for r in h.itertuples()}
        if not all(s in S for s in ("F",) + ON):
            continue
        r0 = S["F"]
        rec = {"model_key": mk, "group": r0.group, "cell": r0.cell, "dataset": r0.dataset, "arch": r0.arch, "batch": r0.batch,
               "target": r0.target, "K": r0.K}
        rec["fam"] = fam(r0)
        rec["aware"] = rec["fam"] == "control" or "QAT" in r0.cell
        mF = np.asarray(r0.margins)
        for s in ("F",) + ON:
            g = gamma_outlier(S[s].margins)
            rec[f"native_{s}"] = g.get("raw_score", np.nan) if g.get("status") == "VALID" else np.nan
            rec[f"cand_{s}"] = int(np.argmax(S[s].margins))
            rec[f"asr_{s}"] = S[s].asr
            rec[f"acc_{s}"] = S[s].acc
        rec["native_on"] = np.nanmax([rec[f"native_{s}"] for s in ON])
        zs, cs, zr, zv = {}, {}, {}, {}
        for s in ON:
            dl = lg(S[s].margins) - lg(mF)
            zs[s], cs[s] = rz(dl)
            X = np.c_[np.ones(len(mF)), lg(mF), np.nan_to_num(np.asarray(r0.clean_margin, float))]
            y = lg(S[s].margins)
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            zr[s], _ = rz(y - X @ beta)
            e_on = lg([t[1] for t in S[s].traj])
            e_off = lg([t[1] for t in r0.traj])
            zv[s], _ = rz(e_on - e_off)
        best = max(ON, key=lambda s: zs[s])
        rec |= {"Z": zs[best], "Zstate": best, "cstar": cs[best], "Z_reg": max(zr.values()), "Z_speed": max(zv.values())}
        for s in ON:
            rec[f"z_{s}"], rec[f"c_{s}"] = zs[s], cs[s]
        rec["asr_on_max"] = np.nanmax([rec[f"asr_{s}"] if rec[f"asr_{s}"] is not None else np.nan for s in ON]) \
            if rec["fam"] not in ("normal", "control") else np.nan
        recs.append(rec)
    s = pd.DataFrame(recs)
    s["split"] = np.where(s.batch == "batch_h", "held-out", "seen")
    neg = s[s.fam.isin(("normal", "control"))]
    aw = neg[neg.aware]
    q = s[~s.fam.isin(("normal", "control"))]
    L = [f"# ST12 tables (run `{run}`, {len(s)} models: QCB {len(q)}, normals+controls {len(neg)}, aware {len(aw)})", ""]
    L += ["## Mechanism: ASR > .5 by family and state (true trigger)", "",
          "| family | n | F | Bfp | C1 | N | any on-state |", "|---|---|---|---|---|---|---|"]
    for f in FAMS:
        g = q[q.fam == f]
        if not len(g):
            continue
        cells = [f"{(g[f'asr_{x}'].astype(float) > .5).sum()}" for x in ("F",) + ON]
        L.append(f"| {f} | {len(g)} | " + " | ".join(cells) + f" | {(g.asr_on_max > .5).sum()} |")
    L += ["", "clean accuracy medians (normals+controls / QCB): " + "; ".join(
        f"{x} {neg[f'acc_{x}'].median():.3f}/{q[f'acc_{x}'].median():.3f}" for x in ("F",) + ON), ""]
    L += ["## AUROC (QCB higher) by score", "",
          "| score | all QCB vs normals+controls | " + " | ".join(FAMS) + " | 4-bit fams vs aware |",
          "|---|---|" + "---|" * len(FAMS) + "---|"]
    for sc in ("native_F", "native_on", "native_Bfp", "native_C1", "native_N", "Z", "Z_reg", "Z_speed"):
        row = [f"{auroc(q[sc], neg[sc]):.2f}"] + [f"{auroc(q[q.fam == f][sc], neg[sc]):.2f}" for f in FAMS]
        row.append(f"{auroc(q[q.fam.isin(FOUR)][sc], aw[sc]):.2f}")
        L.append(f"| {sc} | " + " | ".join(row) + " |")
    L.append("")
    qw = q[q.fam == "QuEST-W4A4"]
    L += [f"QuEST W4A4 vs aware normals (Z): {auroc(qw.Z, aw.Z):.2f}; held-out {auroc(qw[qw.split == 'held-out'].Z, aw.Z):.2f}; "
          f"by z_N only: {auroc(qw.z_N, aw.z_N):.2f}", ""]
    on = q[q.asr_on_max > .5]
    L += [f"Target id (switched-on QCB, n {len(on)}): cstar == target {(on.cstar == on.target).mean():.2f}; native MM-BD on F cand "
          f"== target {(q.cand_F == q.target).mean():.2f} (all QCB); best native on-state cand == target "
          f"{np.mean([r[f'cand_{max(ON, key=lambda x: r[f'native_{x}'] if not np.isnan(r[f'native_{x}']) else -1)}'] == r['target'] for _, r in on.iterrows()]):.2f}"]
    for f in FAMS:
        g = on[on.fam == f]
        if len(g):
            L.append(f"   {f}: cstar == target {(g.cstar == g.target).sum()}/{len(g)}")
    L += ["", f"Alarm Z > {ZTH}: normals+controls {(neg.Z > ZTH).mean():.2f} ({(neg.Z > ZTH).sum()}/{len(neg)}), aware {(aw.Z > ZTH).mean():.2f}; "
          "QCB by family: " + ", ".join(f"{f} {(q[q.fam == f].Z > ZTH).sum()}/{(q.fam == f).sum()}" for f in FAMS)]
    th = np.percentile(neg.Z, 95)
    L.append(f"(optimistic) threshold = normals' 95th pct {th:.2f}: QCB detected {(q.Z > th).mean():.2f}, with target {((q.Z > th) & (q.cstar == q.target)).mean():.2f}")
    md = neg.groupby("dataset").Z.median()
    L += [f"normals' Z median by dataset: {md.round(2).to_dict()}, max/min {md.max() / md.min():.2f}", ""]
    L += ["## Strong classes (normals+controls): share of the most common class per dataset", ""]
    for ds, g in neg.groupby("dataset"):
        L.append(f"{ds} (n {len(g)}): native F cand mode share {g.cand_F.value_counts().iloc[0] / len(g):.2f} "
                 f"(class {g.cand_F.mode().iloc[0]}); differential cstar mode share {g.cstar.value_counts().iloc[0] / len(g):.2f} "
                 f"(class {g.cstar.mode().iloc[0]}); state of max z {g.Zstate.value_counts().to_dict()}")
    (HERE / "results").mkdir(exist_ok=True)
    s.to_csv(HERE / "results/scores.csv", index=False)
    (HERE / "analysis_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
