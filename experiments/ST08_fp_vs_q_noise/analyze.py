#!/usr/bin/env python3
"""ST08 readout (rules fixed in prediction.md).   python3 analyze.py <run id>"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402

QK, CTRL, QAT, UN = "QuEST-W4A4", "control-QuEST-W4A4", "QAT-W4A4-IAO", "unaware"
TWIN, NORMAL = (CTRL, QAT), (CTRL, QAT, UN)
T_MAIN = 2.0
STATES = ("QA", "QN", "FA", "FN")


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + .5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def L_(x):
    return np.log(np.asarray(x, float) + 1e-12)


def main(run):
    d = pd.DataFrame(R.rows(run))
    d["split"] = np.where(d.batch == "batch_h", "held-out", "seen")
    ref = d[d.part == "ref"].set_index("model_key")
    st = d[d.part == "state"].copy()
    st["noise"] = st.draw.astype(str) != "true"
    L = [f"# ST08 tables (run `{run}`, {st.model_key.nunique()} models)", ""]
    a1 = st[(st.state == "QA") & ~st.noise & (st.kind == QK)]
    L += [f"P0 QuEST W4A4 QA t 1 true: ASR median {a1.asr.median():.3f} (n {len(a1)}), acc {a1.acc.median():.3f}", ""]
    rows = []
    for mk, h in st[st.noise].groupby("model_key"):
        rf = ref.loc[mk]
        mR = np.asarray(rf.m)
        rec = {"model_key": mk, "split": rf.split, "kind": rf.kind, "dataset": rf.dataset, "arch": rf.arch, "target": rf.target}
        V = {}
        for s in STATES:
            for t in sorted(h.t.unique()):
                x = h[(h.state == s) & (h.t == t)]
                v = np.mean([L_(m) - L_(mR) for m in x.m], 0)
                V[(s, t)] = v
                tg = f"{s}_{t:g}"
                rec[f"c_{tg}"], rec[f"G_{tg}"] = int(v.argmax()), float(v.max())
                rec[f"acc_{tg}"] = x.acc.mean()
                if rf.kind == QK:
                    rec[f"asr_{tg}"] = x.asr.mean()
        for t in sorted(h.t.unique()):
            dv = V[("QA", t)] - V[("FN", t)]
            tg = f"{t:g}"
            rec[f"S_{tg}"], rec[f"cS_{tg}"] = float(dv.max()), int(dv.argmax())
            a, b = V[("QA", t)], V[("FN", t)]
            rec[f"dis_{tg}"] = float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        rows.append(rec)
    s = pd.DataFrame(rows)
    s["cell"] = s.dataset + "/" + s.arch.str.replace("16-BN", "")
    tm = f"{T_MAIN:g}"
    sink = {c: g[g.kind.isin(NORMAL)][f"c_QA_{tm}"].mode().iloc[0] for c, g in s.groupby("cell")}
    s["sink"] = s.cell.map(sink)
    n, q = s[s.kind.isin(NORMAL)], s[s.kind == QK]

    L += ["## 2 x 2 by noise size: drift class = target (QuEST), QuEST ASR, same drift class QA vs FN (normals / QuEST)", ""]
    for t in sorted(st[st.noise].t.unique()):
        tg = f"{t:g}"
        parts = []
        for stn in STATES:
            parts.append(f"{stn}: tgt {(q[f'c_{stn}_{tg}'] == q.target).sum()}/{len(q)}, ASR {q[f'asr_{stn}_{tg}'].median():.2f}, "
                         f"acc(normals) {n[f'acc_{stn}_{tg}'].median():.2f}")
        L.append(f"t {tg}: " + " | ".join(parts))
        L.append(f"      same class QA vs FN: normals {(n[f'c_QA_{tg}'] == n[f'c_FN_{tg}']).mean():.2f}, twins "
                 f"{(s[s.kind.isin(TWIN)][f'c_QA_{tg}'] == s[s.kind.isin(TWIN)][f'c_FN_{tg}']).mean():.2f}, QuEST "
                 f"{(q[f'c_QA_{tg}'] == q[f'c_FN_{tg}']).mean():.2f} | median 1 - cos: normals {n[f'dis_{tg}'].median():.2f}, "
                 f"QuEST {q[f'dis_{tg}'].median():.2f}")
    L += ["", f"P1 normals at t {tm}: c_QA == c_FN {(n[f'c_QA_{tm}'] == n[f'c_FN_{tm}']).mean():.2f}; "
          f"median cos(v_QA, v_FN) {1 - n[f'dis_{tm}'].median():.2f} (n {len(n)})",
          f"P2 QuEST at t {tm}: FN drift class == target {(q[f'c_FN_{tm}'] == q.target).mean():.2f}, FN ASR median "
          f"{q[f'asr_FN_{tm}'].median():.2f}; QA drift class == target {(q[f'c_QA_{tm}'] == q.target).mean():.2f}", "",
          f"sinks (normal mode of c_QA at t {tm}): {sink}", ""]

    L += [f"## Main score S_{tm} = max_c [v_QA(c) - v_FN(c)] (pooled over datasets) and secondary scores", ""]
    for col, ccol in ((f"S_{tm}", f"cS_{tm}"), ("S_1.5", "cS_1.5"), ("S_3", "cS_3"), (f"dis_{tm}", None), ("dis_3", None),
                      (f"G_QA_{tm}", f"c_QA_{tm}")):
        if col not in s:
            continue
        L.append(f"### {col}")
        for sp in ("held-out", "seen"):
            g = s[s.split == sp]
            qq, tw, nm = g[g.kind == QK], g[g.kind.isin(TWIN)], g[g.kind.isin(NORMAL)]
            line = (f"{sp}: QuEST n{len(qq)} med {qq[col].median():.3g} | twin n{len(tw)} med {tw[col].median():.3g} | "
                    f"normals n{len(nm)} med {nm[col].median():.3g} | AUROC vs twin {auroc(qq[col], tw[col]):.2f}, "
                    f"vs all normals {auroc(qq[col], nm[col]):.2f}")
            if ccol:
                line += f" | QuEST c == target {(qq[ccol] == qq.target).sum()}/{len(qq)}"
            L.append(line)
        if ccol:
            L.append(f"normals: c == sink {(n[ccol] == n.sink).mean():.2f} (n {len(n)})")
        md = n.groupby("dataset")[col].median()
        L.append(f"normals' median by dataset: {md.round(3).to_dict()}, max/min {md.max() / md.min():.2f}")
        for c, g in s.groupby("cell"):
            qq, nm = g[g.kind == QK], g[g.kind.isin(NORMAL)]
            if len(qq):
                L.append(f"   {c}: QuEST n{len(qq)} med {qq[col].median():.3g} vs normals n{len(nm)} med {nm[col].median():.3g}, "
                         f"AUROC {auroc(qq[col], nm[col]):.2f}" + (f", c == target {(qq[ccol] == qq.target).sum()}/{len(qq)}" if ccol else ""))
        L.append("")
    L += ["## Fragility (clean accuracy, QA t 2; QuEST lower)", ""]
    for sp in ("held-out", "seen"):
        g = s[s.split == sp]
        L.append(f"acc_QA_{tm} {sp}: AUROC {auroc(-g[g.kind == QK][f'acc_QA_{tm}'], -g[g.kind.isin(TWIN)][f'acc_QA_{tm}']):.2f}")
    (HERE / "results").mkdir(exist_ok=True)
    s.to_csv(HERE / "results/scores.csv", index=False)
    (HERE / "analysis_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
