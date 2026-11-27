#!/usr/bin/env python3
"""ST05 readout (rules fixed in prediction.md).   python3 analyze.py <run id>"""
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
T_MAIN = 3.0


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + .5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def compare(df, col, lower_is_qcb=False, label=""):
    out = []
    for sp, g in df.groupby("split"):
        q, c, a, u = (g[g.kind == k][col] for k in (QK, CTRL, QAT, UN))
        t = pd.concat([c, a])
        s = (lambda x, y: auroc(-x, -y)) if lower_is_qcb else auroc
        out.append(f"{label} {sp}: QuEST n{len(q)} med {q.median():.3g} | control n{len(c)} med {c.median():.3g} | "
                   f"QAT n{len(a)} med {a.median():.3g} | unaware n{len(u)} med {u.median():.3g} | "
                   f"AUROC QuEST vs twin {s(q, t):.2f}, vs control {s(q, c):.2f}, vs QAT {s(q, a):.2f}, vs unaware {s(q, u):.2f}")
    return out


def gains(m, mR, h, hR, n=1000):
    m, mR = np.asarray(m), np.asarray(mR)
    g = np.log(m + 1e-12) - np.log(mR + 1e-12)
    gh = np.log(np.asarray(h) + 1 / n) - np.log(np.asarray(hR) + 1 / n)
    o = np.sort(g)[::-1]
    return float(o[0]), int(np.argmax(g)), float(o[0] - o[1]), float(gh.max()), int(np.argmax(gh))


def main(run):
    d = pd.DataFrame(R.rows(run))
    d["split"] = np.where(d.batch == "batch_h", "held-out", "seen")
    ref = d[d.part == "ref"].set_index("model_key")
    st = d[d.part == "state"].copy()
    g = [gains(r.m, ref.loc[r.model_key, "m"], r.h, ref.loc[r.model_key, "h"]) for r in st.itertuples()]
    st["G"], st["c"], st["gap"], st["Gh"], st["ch"] = zip(*g)
    L = [f"# ST05 tables (run `{run}`, {st.model_key.nunique()} models)", ""]

    # ---- P0, mechanism curves
    a1 = st[(st.fam == "A") & (st.t == 1.0) & (st.draw == "true") & (st.kind == QK)]
    L += [f"P0 QuEST W4A4 aligned (A, t 1, true) ASR median {a1.asr.median():.3f} (n {len(a1)}), acc {a1.acc.median():.3f}", ""]
    st["noise"] = st.draw != "true"
    cur = st.groupby(["kind", "split", "fam", "t", "noise"]).agg(n=("model_key", "nunique"), acc=("acc", "median"),
                                                                 asr=("asr", "median"), G=("G", "median"),
                                                                 kl=("kl", "median")).round(3).reset_index()
    L += ["## Curves (medians over models and draws)", "", S1.md(cur), ""]

    # ---- per-model scores
    rows = []
    for (mk, sp), h in st.groupby(["model_key", "split"]):
        a = h.iloc[0]
        rec = {"model_key": mk, "split": sp, "kind": a.kind, "dataset": a.dataset, "arch": a.arch, "target": a.target}
        for fam, ts in (("A", (1.5, 2.0, 2.5, 3.0, 4.0)), ("Aoff", (1.0, 2.0, 3.0))):
            for t in ts:
                x = h[(h.fam == fam) & (h.t == t) & h.noise]
                if not len(x):
                    continue
                rec[f"S_{fam}_{t:g}"] = x.G.mean()
                rec[f"Sh_{fam}_{t:g}"] = x.Gh.mean()
                rec[f"gap_{fam}_{t:g}"] = x.gap.mean()
                rec[f"acc_{fam}_{t:g}"] = x.acc.mean()
                rec[f"c_{fam}_{t:g}"] = int(x.c.mode().iloc[0])
                rec[f"cagree_{fam}_{t:g}"] = float((x.c == x.c.mode().iloc[0]).mean())
                if a.kind == QK:
                    rec[f"asr_{fam}_{t:g}"] = x.asr.mean()
            tr = h[(h.fam == "A") & (h.t == T_MAIN) & ~h.noise]
            if len(tr):
                rec["S_true_3"] = float(tr.G.iloc[0])
        rows.append(rec)
    s = pd.DataFrame(rows)
    main_col = f"S_A_{T_MAIN:g}"
    L += [f"## Main score {main_col}: mean over noise draws of max_c log(m_X(c) / m_R(c)), A family, t = {T_MAIN:g}", ""]
    L += compare(s, main_col, label=main_col)
    L += ["", "## Secondary scores", ""]
    for col in ("S_A_2", "S_A_2.5", "S_A_4", "Sh_A_3", "gap_A_3", "S_true_3", "S_Aoff_2", "S_Aoff_3", "Sh_Aoff_3"):
        if col in s:
            L += compare(s, col, label=col)
    L += ["", "## Fragility check (clean accuracy under the same noise; lower = more fragile)", ""]
    for col in ("acc_A_2", "acc_A_3", "acc_Aoff_3"):
        if col in s:
            L += compare(s, col, lower_is_qcb=True, label=col)

    # ---- drift class = target (QuEST, mechanism)
    q = s[s.kind == QK]
    L += ["", "## Drift class vs target (QuEST W4A4; c = mode over noise draws)", ""]
    for sp, h in q.groupby("split"):
        for col in [c for c in ("c_A_2", "c_A_3", "c_Aoff_3") if c in h]:
            L.append(f"{sp} {col} == target: {(h[col] == h.target).sum()}/{len(h)}")
    for sp, h in q.groupby(["split", "dataset", "arch"]):
        L.append(f"{sp}: c_A_3 == target {(h.c_A_3 == h.target).sum()}/{len(h)}")
    tw = s[s.kind.isin([CTRL, QAT])]
    L += ["", "Twin drift classes at t 3 (dataset, arch: classes)", ""]
    for k, h in tw.groupby(["dataset", "arch"]):
        L.append(f"{k}: {sorted(h.c_A_3.tolist())}; QuEST targets there: {sorted(q[(q.dataset == k[0]) & (q.arch == k[1])].target.astype(int).tolist())}")
    L += ["", "## By dataset/arch, main score (QuEST vs twin)", ""]
    for k, h in s.groupby(["dataset", "arch"]):
        qq, tt = h[h.kind == QK][main_col], h[h.kind.isin([CTRL, QAT])][main_col]
        L.append(f"{k}: QuEST n{len(qq)} med {qq.median():.3g} | twin n{len(tt)} med {tt.median():.3g} | AUROC {auroc(qq, tt):.2f}")
    L += ["", "## Noise switches the backdoor on with the weight switch off (Aoff, QuEST ASR)", ""]
    for col in ("asr_Aoff_1", "asr_Aoff_2", "asr_Aoff_3"):
        if col in q:
            L.append(f"{col}: median {q[col].median():.3f}, share > .5 {(q[col] > .5).mean():.2f} (n {q[col].notna().sum()})")
    (HERE / "results").mkdir(exist_ok=True)
    s.to_csv(HERE / "results/scores.csv", index=False)
    cur.to_csv(HERE / "results/curves.csv", index=False)
    (HERE / "analysis_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
