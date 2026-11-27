#!/usr/bin/env python3
"""ST13 readout (rules fixed in prediction.md).   python3 analyze.py <run id> [ST12 run id]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402

ON = ("Bfp", "C1", "N")
FAMS = ("Ma-8", "Ma-W4A4", "Tian-8", "Tian-W4A4", "QuANTI-8+4", "QuEST-8", "QuEST-W4A4")
FOUR = ("Ma-W4A4", "Tian-W4A4", "QuANTI-8+4", "QuEST-W4A4")
ZTH = 3.5


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + .5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def rz(v):
    v = np.asarray(v, float)
    ok = ~np.isnan(v)
    med = np.median(v[ok])
    mad = 1.4826 * np.median(np.abs(v[ok] - med))
    j = int(np.nanargmax(v))
    return float((v[j] - med) / max(mad, 1e-6)), j


def fam(group, cell):
    if group != "QCB":
        return "control" if group == "control-nopoison" else "normal"
    for f in FAMS:
        if f in cell:
            return f
    return "QCB-other"


def per_class(pairs, cost, K):
    """forward F_c = median_s log cost(s->c), reverse R_c = median_s log cost(c->s), A_c = R_c - F_c."""
    lc = {tuple(p): np.log(max(c, 1e-3)) for p, c in zip(pairs, cost)}
    Fw, Rv = np.full(K, np.nan), np.full(K, np.nan)
    for c in range(K):
        f = [lc[(s, c)] for s in range(K) if (s, c) in lc and (c, s) in lc]
        b = [lc[(c, s)] for s in range(K) if (s, c) in lc and (c, s) in lc]
        if f:
            Fw[c], Rv[c] = np.median(f), np.median(b)
    return Fw, Rv, Rv - Fw


def main(run, st12="ST12_20261127_001"):
    d = pd.DataFrame(R.rows(run))
    pr = d[d.part == "pairs"].set_index("model_key")
    st = d[d.part == "state"]
    m12 = pd.read_csv(ROOT / "experiments/ST12_switch_mmbd/results/scores.csv").set_index("model_key") \
        if (ROOT / "experiments/ST12_switch_mmbd/results/scores.csv").exists() else None
    recs = []
    for mk, h in st.groupby("model_key"):
        S = {r.state: r for r in h.itertuples()}
        if not all(x in S for x in ("F",) + ON):
            continue
        r0 = S["F"]
        K = int(r0.K)
        pairs = pr.loc[mk].pairs
        rec = {"model_key": mk, "group": r0.group, "cell": r0.cell, "dataset": r0.dataset, "batch": r0.batch, "target": r0.target,
               "K": K, "fam": fam(r0.group, r0.cell)}
        rec["aware"] = rec["fam"] == "control" or "QAT" in r0.cell
        PC = {x: per_class(pairs, S[x].cost, K) for x in ("F",) + ON}
        for x in ("F",) + ON:
            rec[f"cens_{x}"] = float(np.mean(S[x].censored))
            rec[f"asr_{x}"] = S[x].asr
            rec[f"zA1_{x}"], rec[f"cA1_{x}"] = rz(PC[x][2])
        zs, cs, ze = {}, {}, {}
        for x in ON:
            zs[x], cs[x] = rz(PC[x][2] - PC["F"][2])
            ze[x], _ = rz(-(PC[x][0] - PC["F"][0]))
        best = max(ON, key=lambda x: zs[x])
        rec |= {"Z_A": zs[best], "cstar": cs[best], "Zstate": best, "Z_E": max(ze.values()),
                "A_F": PC["F"][2].tolist(), "Fw_F": PC["F"][0].tolist()}
        if rec["fam"] not in ("normal", "control"):
            t = int(r0.target)
            rec["asr_on_max"] = np.nanmax([S[x].asr for x in ON])
            onst = max(ON, key=lambda x: S[x].asr)
            rec["dA_target"] = float(PC[onst][2][t] - PC["F"][2][t])
            rec["A_target_F"] = float(PC["F"][2][t])
        if m12 is not None and mk in m12.index:
            rec["mmbd_top_F"] = int(m12.loc[mk].cand_F)
            rec["Z_mmbd"] = float(m12.loc[mk].Z)
        recs.append(rec)
    s = pd.DataFrame(recs)
    s["split"] = np.where(s.batch == "batch_h", "held-out", "seen")
    neg, q = s[s.fam.isin(("normal", "control"))], s[~s.fam.isin(("normal", "control"))]
    aw = neg[neg.aware]
    L = [f"# ST13 tables (run `{run}`, {len(s)} models: QCB {len(q)}, normals+controls {len(neg)}, aware {len(aw)})", ""]
    L.append("censored pair share, median (normals / QCB): " + "; ".join(f"{x} {neg[f'cens_{x}'].median():.2f}/{q[f'cens_{x}'].median():.2f}" for x in ("F",) + ON))
    L.append("")
    # P1: natural strong class symmetric?
    if "mmbd_top_F" in s:
        rows = []
        for _, r in neg.dropna(subset=["mmbd_top_F"]).iterrows():
            A = np.asarray(r.A_F, float)
            c = int(r.mmbd_top_F)
            oth = np.delete(A, c)
            oth = oth[~np.isnan(oth)]
            med, mad = np.median(oth), 1.4826 * np.median(np.abs(oth - np.median(oth)))
            rows.append((r.dataset, A[c], med, (A[c] - med) / max(mad, 1e-6), A[c] <= med + mad))
        p1 = pd.DataFrame(rows, columns=["ds", "A_strong", "A_med_other", "z", "within"])
        L.append(f"P1 natural strong class (ST12 MM-BD top in F) in F: A within median + 1 MAD of other classes {p1.within.mean():.2f} "
                 f"(n {len(p1)}); z median {p1.z.median():.2f}; by dataset {p1.groupby('ds').within.mean().round(2).to_dict()}")
        fe = []
        for _, r in neg.dropna(subset=["mmbd_top_F"]).iterrows():
            Fw = np.asarray(r.Fw_F, float)
            fe.append(int(np.nanargmin(Fw)) == int(r.mmbd_top_F))
        L.append(f"   (desc.) easiest-to-reach class in F (min forward cost) == MM-BD top class: {np.mean(fe):.2f}")
    on = q[q.asr_on_max > .5]
    L += [f"P2 switched-on QCB (n {len(on)}): dA(target) > 0 {(on.dA_target > 0).mean():.2f}; cstar == target {(on.cstar == on.target).mean():.2f}"]
    for f in FAMS:
        g = on[on.fam == f]
        if len(g):
            L.append(f"   {f}: dA > 0 {(g.dA_target > 0).sum()}/{len(g)}, cstar == target {(g.cstar == g.target).sum()}/{len(g)}, "
                     f"A_target in F median {g.A_target_F.median():.2f}")
    L += ["", "## AUROC (QCB higher)", "", "| score | all QCB vs normals+controls | " + " | ".join(FAMS) + " | 4-bit vs aware |",
          "|---|---|" + "---|" * len(FAMS) + "---|"]
    for sc in ["Z_A", "Z_E"] + [f"zA1_{x}" for x in ("F",) + ON] + (["Z_mmbd"] if "Z_mmbd" in s else []):
        row = [f"{auroc(q[sc], neg[sc]):.2f}"] + [f"{auroc(q[q.fam == f][sc], neg[sc]):.2f}" for f in FAMS]
        row.append(f"{auroc(q[q.fam.isin(FOUR)][sc], aw[sc]):.2f}")
        L.append(f"| {sc} | " + " | ".join(row) + " |")
    qw = q[q.fam == "QuEST-W4A4"]
    L += ["", f"QuEST W4A4 vs aware (Z_A): {auroc(qw.Z_A, aw.Z_A):.2f}, held-out {auroc(qw[qw.split == 'held-out'].Z_A, aw.Z_A):.2f}"
          + (f"; ST12 Z_mmbd same models: {auroc(qw.Z_mmbd, aw.Z_mmbd):.2f}" if "Z_mmbd" in s else ""),
          f"Alarm Z_A > {ZTH}: normals+controls {(neg.Z_A > ZTH).mean():.2f} ({(neg.Z_A > ZTH).sum()}/{len(neg)}); QCB by family: "
          + ", ".join(f"{f} {(q[q.fam == f].Z_A > ZTH).sum()}/{(q.fam == f).sum()}" for f in FAMS)]
    md = neg.groupby("dataset").Z_A.median()
    L += [f"normals' Z_A median by dataset: {md.round(2).to_dict()}, max/min {md.max() / md.min():.2f}", ""]
    (HERE / "results").mkdir(exist_ok=True)
    s.drop(columns=["A_F", "Fw_F"]).to_csv(HERE / "results/scores.csv", index=False)
    (HERE / "analysis_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:])
