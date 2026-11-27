#!/usr/bin/env python3
"""ST06 readout (rules fixed in prediction.md).   python3 analyze.py <run id>"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402

_spec = importlib.util.spec_from_file_location("st01a", ROOT / "experiments/ST01_deployment_awareness/analyze.py")
S1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S1)
QK, CTRL, QAT, UN = "QuEST-W4A4", "control-QuEST-W4A4", "QAT-W4A4-IAO", "unaware"
TWIN, NORMAL = (CTRL, QAT), (CTRL, QAT, UN)
T = 3.0
ST05_SCORES = ROOT / "experiments/ST05_overdrive/results/scores.csv"


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + .5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def rz(v):
    """Class-level robust z of the top class: (max - median) / scale, scale = max(1.4826 MAD, .25 sd, 1e-12)."""
    v = np.asarray(v, float)
    med = np.median(v)
    sc = max(1.4826 * np.median(np.abs(v - med)), .25 * v.std(), 1e-12)
    return float((v.max() - med) / sc), int(v.argmax())


def L_(x):
    return np.log(np.asarray(x, float) + 1e-12)


def main(run):
    d = pd.DataFrame(R.rows(run))
    d["split"] = np.where(d.batch == "batch_h", "held-out", "seen")
    ref = d[d.part == "ref"].set_index("model_key")
    st = d[d.part == "state"].copy()
    st["noise"] = st.draw.astype(str) != "true"
    L = [f"# ST06 tables (run `{run}`, {st.model_key.nunique()} models)", ""]

    # ---- P0
    a1 = st[(st.fam == "A") & (st.t == 1.0) & ~st.noise & (st.kind == QK)]
    L += [f"P0a QuEST W4A4 A t 1 true: ASR median {a1.asr.median():.3f} (n {len(a1)})"]
    rows = []
    for mk, h in st.groupby("model_key"):
        rf = ref.loc[mk]
        mR = np.asarray(rf.m)
        a = h[(h.fam == "A") & (h.t == T) & h.noise]
        v_a = np.mean([L_(m) - L_(mR) for m in a.m], 0)
        kl_a = a.kl.mean()
        n = h[h.fam == "N"]
        kl_n = n.groupby("t").kl.mean()
        sg = float((kl_n - kl_a).abs().idxmin())                    # matched sigma (KL closest to A t 3)
        ns = n[n.t == sg]
        v_n = np.mean([L_(m) - L_(mR) for m in ns.m], 0)
        rec = {"model_key": mk, "split": rf.split, "kind": rf.kind, "dataset": rf.dataset, "arch": rf.arch, "target": rf.target,
               "K": rf.K, "G": float(v_a.max()), "c_raw": int(v_a.argmax()), "sigma": sg, "kl_A3": kl_a,
               "kl_N": float(kl_n[sg]), "c_N": int(v_n.argmax()),
               "exc_self": float((v_a - v_n).max()), "c_self": int((v_a - v_n).argmax()),
               "dis_self": float(1 - v_a @ v_n / (np.linalg.norm(v_a) * np.linalg.norm(v_n) + 1e-12)),
               "asr_N": ns.asr.mean() if rf.kind == QK else np.nan, "acc_A3": a.acc.mean(), "acc_N": ns.acc.mean(),
               "r_glob": rf.r_glob, "d": rf.d, "perp_share": a.perp_share.mean()}
        for col, nm in (("push_raw", "raw"), ("push_glob", "glob"), ("push_mask", "mask"), ("push_mask_true", "mask_true")):
            z, c = rz(np.mean([np.asarray(x) for x in a[col]], 0))
            rec[f"z_{nm}"], rec[f"c_{nm}"] = z, c
        rows.append(rec)
    s = pd.DataFrame(rows)
    if ST05_SCORES.exists():
        s5 = pd.read_csv(ST05_SCORES).set_index("model_key")["S_A_3"]
        dd = (s.set_index("model_key").G - s5.reindex(s.model_key).values).abs()
        L += [f"P0b reproduction of ST05 S_A_3: median |diff| {dd.median():.2e}, max {dd.max():.2e} (n {dd.notna().sum()})"]
    L += [f"matched sigma: {s.sigma.value_counts().sort_index().to_dict()}; median KL A3 {s.kl_A3.median():.3f}, N {s.kl_N.median():.3f}",
          f"global span rank r / d: median {np.median(s.r_glob / s.d):.2f}; outside-span share of the mean feature change "
          f"(A t 3): median {s.perp_share.median():.2f}", ""]

    # ---- sinks: per dataset/arch the normal mode of c_raw
    s["cell"] = s.dataset + "/" + s.arch.str.replace("16-BN", "")
    sink = {c: g[g.kind.isin(NORMAL)].c_raw.mode().iloc[0] for c, g in s.groupby("cell")}
    s["sink"] = s.cell.map(sink)
    L += ["## Sinks (normal mode of the raw drift class) and the input-noise drift class of normals", ""]
    for c, g in s.groupby("cell"):
        n = g[g.kind.isin(NORMAL)]
        L.append(f"{c}: sink {sink[c]}; normals n{len(n)}: c_raw == sink {(n.c_raw == n.sink).mean():.2f}, "
                 f"c_N == sink {(n.c_N == n.sink).mean():.2f}")
    n = s[s.kind.isin(NORMAL)]
    L += [f"P1 all normals: c_N == sink {(n.c_N == n.sink).mean():.2f} (n {len(n)})", ""]

    # ---- scores
    SC = (("exc_self", "c_self", "(a) input-noise self-reference"), ("z_glob", "c_glob", "(b) outside the clean global span"),
          ("z_mask", "c_mask", "(b') outside class masks, R's own classes"), ("dis_self", None, "(a') 1 - cos(v_A, v_N)"),
          ("z_mask_true", "c_mask_true", "labelled: outside class masks, probe labels"), ("z_raw", "c_raw", "raw head push"),
          ("G", "c_raw", "ST05 main score (reproduced)"))
    L += ["## Scores: QuEST W4A4 vs twins / all normals (pooled over datasets; no reference models)", ""]
    for col, ccol, lab in SC:
        L.append(f"### {col}: {lab}")
        for sp in ("held-out", "seen"):
            g = s[s.split == sp]
            q, tw, nm = g[g.kind == QK], g[g.kind.isin(TWIN)], g[g.kind.isin(NORMAL)]
            line = (f"{sp}: QuEST n{len(q)} med {q[col].median():.3g} | twin n{len(tw)} med {tw[col].median():.3g} | "
                    f"normals n{len(nm)} med {nm[col].median():.3g} | AUROC vs twin {auroc(q[col], tw[col]):.2f}, "
                    f"vs all normals {auroc(q[col], nm[col]):.2f}")
            if ccol:
                line += f" | QuEST c == target {(q[ccol] == q.target).sum()}/{len(q)}"
            L.append(line)
        if ccol:
            n = s[s.kind.isin(NORMAL)]
            L.append(f"normals: c == sink {(n[ccol] == n.sink).mean():.2f} (n {len(n)})")
        for c, g in s.groupby("cell"):
            q, nm = g[g.kind == QK], g[g.kind.isin(NORMAL)]
            if len(q):
                L.append(f"   {c}: QuEST n{len(q)} vs normals n{len(nm)} AUROC {auroc(q[col], nm[col]):.2f}"
                         + (f", c == target {(q[ccol] == q.target).sum()}/{len(q)}" if ccol else ""))
        L.append("")

    # ---- reference contamination and fragility
    q = s[s.kind == QK]
    L += ["## Reference contamination: QuEST ASR at the matched input-noise sigma (R, no deployment)", "",
          f"median {q.asr_N.median():.3f}, share > .5 {(q.asr_N > .5).mean():.2f} (n {q.asr_N.notna().sum()})", "",
          "## Fragility: clean accuracy under the same states (lower = more fragile), QuEST vs twin AUROC", ""]
    for col in ("acc_A3", "acc_N"):
        for sp in ("held-out", "seen"):
            g = s[s.split == sp]
            L.append(f"{col} {sp}: AUROC (QuEST lower) {auroc(-g[g.kind == QK][col], -g[g.kind.isin(TWIN)][col]):.2f}")
    (HERE / "results").mkdir(exist_ok=True)
    s.to_csv(HERE / "results/scores.csv", index=False)
    (HERE / "analysis_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
