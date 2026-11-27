#!/usr/bin/env python3
"""ST07 readout (rules fixed in prediction.md).   python3 analyze.py <run id>"""
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
N_NOISE, MAJ, N_MIN = 8, 4, 5
ST05_SCORES = ROOT / "experiments/ST05_overdrive/results/scores.csv"


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + .5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def dests(YX, src, K):
    """Image-level destination: the class (!= source) taken in >= MAJ of the draws, else -1. YX: draws x images."""
    cnt = np.stack([(YX == c).sum(0) for c in range(K)], 1)          # images x K
    cnt[np.arange(len(src)), src] = 0
    d = cnt.argmax(1)
    d[cnt.max(1) < MAJ] = -1
    return d


def dispersion(dest, src, c, K, keep=None):
    """D = X^2 / (m - 1) of the sources of the images that go to c, against rates equal across sources
    (expected counts proportional to the source sizes). n < N_MIN -> 1 (no information). Returns D, n."""
    keep = np.ones(len(src), bool) if keep is None else keep
    Ns = np.bincount(src[keep], minlength=K).astype(float)
    O = np.bincount(src[keep & (dest == c)], minlength=K).astype(float)
    ok = (Ns > 0) & (np.arange(K) != c)
    n, m = O[ok].sum(), ok.sum()
    if n < N_MIN or m < 2:
        return 1.0, int(n)
    E = n * Ns[ok] / Ns[ok].sum()
    return float(((O[ok] - E) ** 2 / E).sum() / (m - 1)), int(n)


def main(run):
    d = pd.DataFrame(R.rows(run))
    d["split"] = np.where(d.batch == "batch_h", "held-out", "seen")
    ref = d[d.part == "ref"].set_index("model_key")
    st = d[d.part == "state"].copy()
    st["noise"] = st.draw.astype(str) != "true"
    L = [f"# ST07 tables (run `{run}`, {st.model_key.nunique()} models)", ""]
    a1 = st[(st.fam == "A") & (st.t == 1.0) & ~st.noise & (st.kind == QK)]
    L += [f"P0a QuEST W4A4 A t 1 true: ASR median {a1.asr.median():.3f} (n {len(a1)})"]
    rows = []
    for mk, h in st.groupby("model_key"):
        rf = ref.loc[mk]
        K, src, y = int(rf.K), np.asarray(rf.yR), np.asarray(rf.y)
        rec = {"model_key": mk, "split": rf.split, "kind": rf.kind, "dataset": rf.dataset, "arch": rf.arch,
               "target": rf.target, "K": K}
        for t in (2.0, 3.0):
            a = h[(h.t == t) & h.noise].sort_values("draw")
            YX = np.stack([np.asarray(v) for v in a.yX])
            de = dests(YX, src, K)
            inflow = np.bincount(de[de >= 0], minlength=K)
            c = int(inflow.argmax())
            Dc, n = dispersion(de, src, c, K)
            tg = f"{t:g}"
            rec[f"c_{tg}"], rec[f"n_{tg}"], rec[f"D_{tg}"] = c, n, Dc
            rec[f"U_{tg}"] = -np.log(max(Dc, 1e-6))
            rec[f"flip_{tg}"] = float((de >= 0).mean())
            # labelled variant: R-correct images, source = label
            ok = src == y
            del_ = dests(YX, y, K)
            infl = np.bincount(del_[(del_ >= 0) & ok], minlength=K)
            cl = int(infl.argmax())
            Dl, nl = dispersion(del_, y, cl, K, keep=ok)
            rec[f"cL_{tg}"], rec[f"DL_{tg}"], rec[f"UL_{tg}"] = cl, Dl, -np.log(max(Dl, 1e-6))
            # per-input CSO: share of the decision push into c from outside the clean span
            P = np.stack([np.asarray(v) for v in a.push])
            Po = np.stack([np.asarray(v) for v in a.push_out])
            sel = (YX == c) & (de == c)[None, :]
            tot = P[sel].sum()
            rec[f"phi_{tg}"] = float(Po[sel].sum() / tot) if tot > 0 and sel.sum() >= N_MIN else np.nan
            if rf.kind == QK:
                Dt, nt = dispersion(de, src, int(rf.target), K)
                rec[f"Dtarget_{tg}"], rec[f"ntarget_{tg}"] = Dt, nt
            if t == 3.0:
                mR = np.log(np.asarray(rf.m) + 1e-12)
                v = np.mean([np.log(np.asarray(m) + 1e-12) - mR for m in a.m], 0)
                rec["G"], rec["c_G"] = float(v.max()), int(v.argmax())
                rec["acc_A3"] = a.acc.mean()
        rows.append(rec)
    s = pd.DataFrame(rows)
    s["cell"] = s.dataset + "/" + s.arch.str.replace("16-BN", "")
    if ST05_SCORES.exists():
        s5 = pd.read_csv(ST05_SCORES).set_index("model_key")["S_A_3"]
        dd = (s.set_index("model_key").G - s5.reindex(s.model_key).values).abs()
        L += [f"P0b reproduction of ST05 S_A_3: median |diff| {dd.median():.2e}, max {dd.max():.2e} (n {dd.notna().sum()})"]
    sink = {c: g[g.kind.isin(NORMAL)].c_G.mode().iloc[0] for c, g in s.groupby("cell")}
    s["sink"] = s.cell.map(sink)
    n = s[s.kind.isin(NORMAL)]
    q = s[s.kind == QK]
    L += ["", "## Flows at t 3 (image-level destination = class taken in >= 4 of 8 draws)", "",
          f"share of images that move: normals med {n.flip_3.median():.3f}, QuEST med {q.flip_3.median():.3f}",
          f"top-inflow class c: normals c == sink {(n.c_3 == n.sink).mean():.2f}; QuEST c == target {(q.c_3 == q.target).sum()}/{len(q)} "
          f"(held-out {(q[q.split == 'held-out'].c_3 == q[q.split == 'held-out'].target).sum()}/{(q.split == 'held-out').sum()})",
          f"inflow n into c: normals med {n.n_3.median():.0f}, QuEST med {q.n_3.median():.0f}",
          f"P1 normals D_3 >= 2: {(n.D_3 >= 2).mean():.2f} (n {len(n)}); median D_3 {n.D_3.median():.2f}",
          f"P2 QuEST with c == target: D_3 <= 2 {(q[q.c_3 == q.target].D_3 <= 2).mean():.2f} (n {(q.c_3 == q.target).sum()}); "
          f"D at the target class (all QuEST) median {q.Dtarget_3.median():.2f}, n into target med {q.ntarget_3.median():.0f}", ""]
    L += ["normals' median D_3 and phi_3 by dataset (criterion 3: max/min > 3)"]
    for col in ("D_3", "phi_3", "DL_3"):
        md = n.groupby("dataset")[col].median()
        L.append(f"  {col}: {md.round(3).to_dict()}, max/min {md.max() / md.min():.2f}")
    L.append("")
    SC = (("U_3", "c_3", "primary 1: flow uniformity -log D (t 3)"), ("phi_3", "c_3", "primary 2: per-input outside-span push share (t 3)"),
          ("U_2", "c_2", "secondary: flow uniformity t 2"), ("phi_2", "c_2", "secondary: outside-span share t 2"),
          ("UL_3", "cL_3", "labelled: flow uniformity, R-correct images, source = label"),
          ("G", "c_G", "ST05 main score (reproduced)"))
    L += ["## Scores: QuEST W4A4 vs twins / all normals (pooled over datasets; no reference models)", ""]
    for col, ccol, lab in SC:
        L.append(f"### {col}: {lab}")
        for sp in ("held-out", "seen"):
            g = s[s.split == sp]
            qq, tw, nm = g[g.kind == QK], g[g.kind.isin(TWIN)], g[g.kind.isin(NORMAL)]
            L.append(f"{sp}: QuEST n{len(qq)} med {qq[col].median():.3g} | twin n{len(tw)} med {tw[col].median():.3g} | "
                     f"normals n{len(nm)} med {nm[col].median():.3g} | AUROC vs twin {auroc(qq[col], tw[col]):.2f}, "
                     f"vs all normals {auroc(qq[col], nm[col]):.2f} | QuEST c == target {(qq[ccol] == qq.target).sum()}/{len(qq)}")
        for c, g in s.groupby("cell"):
            qq, nm = g[g.kind == QK], g[g.kind.isin(NORMAL)]
            if len(qq):
                L.append(f"   {c}: QuEST n{len(qq)} med {qq[col].median():.3g} vs normals n{len(nm)} med {nm[col].median():.3g}, "
                         f"AUROC {auroc(qq[col], nm[col]):.2f}")
        L.append("")
    L += ["## Fragility (clean accuracy at A t 3; QuEST lower)", ""]
    for sp in ("held-out", "seen"):
        g = s[s.split == sp]
        L.append(f"acc_A3 {sp}: AUROC {auroc(-g[g.kind == QK].acc_A3, -g[g.kind.isin(TWIN)].acc_A3):.2f}")
    (HERE / "results").mkdir(exist_ok=True)
    s.to_csv(HERE / "results/scores.csv", index=False)
    (HERE / "analysis_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
