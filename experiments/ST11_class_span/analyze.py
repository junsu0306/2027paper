#!/usr/bin/env python3
"""ST11 readout (rules fixed in prediction.md).   python3 analyze.py <run id>"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402

QK, TWIN, NORMAL = "QuEST-W4A4", ("control-QuEST-W4A4", "QAT-W4A4-IAO"), ("control-QuEST-W4A4", "QAT-W4A4-IAO", "unaware")
STATES = ("R", "Q", "S2n", "S2c")


def zeta(v):
    t, c, n, _ = v
    return t / max(n, 1e-12), c / max(n, 1e-12)


def main(run):
    d = pd.DataFrame(R.rows(run))
    d["split"] = np.where(d.batch == "batch_h", "held-out", "seen")
    st = d[d.part == "state"].copy()
    L = [f"# ST11 tables (run `{run}`, {st.model_key.nunique()} models)", ""]
    q1 = st[(st.kind == QK) & (st.n_lab == 10)]
    L.append("## ASR / accuracy by state (QuEST medians; ctl = random patch -> target)")
    for s in STATES:
        x = q1[q1.state == s]
        L.append(f"{s}: ASR {x.asr.median():.3f} (> .5: {(x.asr > .5).sum()}/{len(x)}), ctl -> target {x.ctl_to_target.median():.3f}, "
                 f"acc {x.acc.median():.3f}")
    L.append("")
    rows = []
    for rr in st.itertuples():
        for ln in rr.layers:
            res = rr.inshare[ln]
            if rr.kind == QK:
                t = str(int(rr.target))
                if t not in res:
                    continue
                zt, zc = zeta(res[t])
                oth = [zeta(v)[0] for c, v in res.items() if c != t]
                rows.append(dict(model_key=rr.model_key, kind=rr.kind, split=rr.split, dataset=rr.dataset, arch=rr.arch, state=rr.state,
                                 n_lab=rr.n_lab, layer=ln, z=zt, zctl=zc, z_other=float(np.median(oth)) if oth else np.nan,
                                 in_trig=res[t][0], in_nat=res[t][2], in_ctl=res[t][1], rank_frac=res[t][3]))
            else:
                zs = [zeta(v) for v in res.values()]
                rows.append(dict(model_key=rr.model_key, kind=rr.kind, split=rr.split, dataset=rr.dataset, arch=rr.arch, state=rr.state,
                                 n_lab=rr.n_lab, layer=ln, z=float(np.median([a for a, _ in zs])), zctl=float(np.median([b for _, b in zs])),
                                 z_other=np.nan, in_trig=float(np.median([v[0] for v in res.values()])),
                                 in_nat=float(np.median([v[2] for v in res.values()])), in_ctl=float(np.median([v[1] for v in res.values()])),
                                 rank_frac=float(np.median([v[3] for v in res.values()]))))
    z = pd.DataFrame(rows)
    z["lpos"] = z.groupby("model_key").layer.transform(lambda s: s.map({n: i for i, n in enumerate(dict.fromkeys(s))}))
    for nl in sorted(z.n_lab.unique()):
        L.append(f"## n_lab = {nl}: zeta = in_target(trig) / in_target(nat) per layer (median); QuEST | twins | unaware")
        for arch in sorted(z.arch.unique()):
            L.append(f"### {arch}")
            L.append("| state | layer | QuEST zeta | QuEST zeta ctl | QuEST zeta other classes | QuEST in(trig) / in(nat) | twins zeta | unaware zeta |")
            L.append("|---|---|---|---|---|---|---|---|")
            for s in ("R", "Q", "S2n"):
                g = z[(z.n_lab == nl) & (z.arch == arch) & (z.state == s)]
                for ln in dict.fromkeys(g.sort_values("lpos").layer):
                    h = g[g.layer == ln]
                    qq, tw, un = h[h.kind == QK], h[h.kind.isin(TWIN)], h[h.kind == "unaware"]
                    L.append(f"| {s} | {ln} | {qq.z.median():.2f} | {qq.zctl.median():.2f} | {qq.z_other.median():.2f} | "
                             f"{qq.in_trig.median():.3f} / {qq.in_nat.median():.3f} | {tw.z.median():.2f} | {un.z.median():.2f} |")
        m = z[z.n_lab == nl].groupby(["model_key", "kind", "split", "state"]).z.min().reset_index()
        L.append("")
        for s in ("Q", "R", "S2n"):
            x = m[m.state == s]
            q, tw, nm = x[x.kind == QK], x[x.kind.isin(TWIN)], x[x.kind.isin(NORMAL)]
            L.append(f"min-layer zeta <= .5 [{s}]: QuEST {(q.z <= .5).mean():.2f} ({(q.z <= .5).sum()}/{len(q)}; held-out "
                     f"{(q[q.split == 'held-out'].z <= .5).sum()}/{(q.split == 'held-out').sum()}), twins {(tw.z <= .5).mean():.2f} "
                     f"({(tw.z <= .5).sum()}/{len(tw)}), all normals {(nm.z <= .5).mean():.2f}")
        hq = z[(z.n_lab == nl) & (z.kind == QK) & (z.layer == "head")].pivot_table(index="model_key", columns="state", values="z")
        L.append(f"head zeta, QuEST paired: Q < R {(hq.Q < hq.R).mean():.2f} (medians Q {hq.Q.median():.2f}, R {hq.R.median():.2f}); "
                 f"|S2n - Q| median {(hq.S2n - hq.Q).abs().median():.2f}")
        L.append("")
    (HERE / "results").mkdir(exist_ok=True)
    z.to_csv(HERE / "results/zeta.csv", index=False)
    (HERE / "analysis_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
