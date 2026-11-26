#!/usr/bin/env python3
"""ST03 readout (rules fixed in prediction.md).   python3 analyze.py <run id>

A: stale flag = ratio_mean >= 3 (stored-vs-R distance over the split-half noise); entropy flag = dent_rel >= .10.
B: per lam: broken = F ASR >= .5; alive = max Q ASR >= .5; undetected = ratio < 3 and dent < .10;
   successful forgery = some lam in {.5, 1} with not broken, alive, undetected.
Seen = earlier cohorts, held-out = batch_h; reported separately.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402

N_JOBS = 388
RT, ET = 3.0, 0.10
_spec = importlib.util.spec_from_file_location("st01a", ROOT / "experiments/ST01_deployment_awareness/analyze.py")
S1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S1)


def kind(group, cell):
    k = S1.kind(group, cell)
    if k == "control":
        for p in ("Tian", "QuANTI", "QuEST"):
            if p in cell:
                return f"control-{p}"
    if k in ("QAT-W4A4-IAO", "QAT-W4A4-EFRAP", "QAT-W4weight"):
        return "QAT-4bit"
    return k


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + .5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def main(run):
    d = pd.DataFrame(R.rows(run))
    d["kind"] = [kind(g, c) for g, c in zip(d.group, d.cell)]
    d["split"] = np.where(d.batch == "batch_h", "held-out", "seen")
    a = d[d.part == "A"].copy()
    a["stale_flag"] = a.ratio_mean >= RT
    a["ent_flag"] = a.dent_rel >= ET
    L = [f"# ST03 tables (run `{run}`, {a.model_key.nunique()}/{N_JOBS} models)", ""]
    order = ["unaware", "QAT-4bit", "QAT-W8weight", "control-QuEST", "control-QuANTI", "control-Tian", "Ma-8", "Ma-W4A4",
             "Tian-8", "Tian-W4A4", "QuANTI", "QuEST-8", "QuEST-W4A4"]
    t = a.groupby(["kind", "split"]).agg(n=("model_key", "size"), stale=("stale_flag", "sum"), ratio_med=("ratio_mean", "median"),
                                        ratio_min=("ratio_mean", "min"), ratio_max=("ratio_mean", "max"), ent=("ent_flag", "sum"),
                                        dent_med=("dent_rel", "median"), dent_max=("dent_rel", "max")).round(3).reset_index()
    t["o"] = t.kind.map({k: i for i, k in enumerate(order)})
    L += ["## A. Stored statistics (ratio >= 3: stale; dent >= .10: entropy)", "", S1.md(t.sort_values(["o", "split"]).drop(columns="o")), ""]
    L.append(f"- split-half noise d_mean: median {a.noise_mean.median():.4f} (range {a.noise_mean.min():.4f}-{a.noise_mean.max():.4f})")
    for sp, g in a.groupby("split"):
        u = g[g.kind == "unaware"]
        aw = g[g.kind != "unaware"]
        L.append(f"- {sp}: AUROC aware vs unaware (ratio) {auroc(aw.ratio_mean, u.ratio_mean):.3f}; unaware stale {u.stale_flag.mean():.3f} "
                 f"({int(u.stale_flag.sum())}/{len(u)}), unaware ent {u.ent_flag.mean():.3f}")
        for (ds, ar), h in g.groupby(["dataset", "arch"]):
            uu, ww = h[h.kind == "unaware"], h[h.kind != "unaware"]
            if len(uu) and len(ww):
                L.append(f"    {ds} {ar}: unaware n {len(uu)} stale {int(uu.stale_flag.sum())}, aware n {len(ww)} stale {int(ww.stale_flag.sum())}, "
                         f"AUROC {auroc(ww.ratio_mean, uu.ratio_mean):.3f}")
    fl = a[(a.kind == "unaware") & (a.stale_flag | a.ent_flag)]
    L += ["", "Unaware flagged: " + "; ".join(f"{r.model_key} ratio {r.ratio_mean:.1f} dent {r.dent_rel:.3f}" for r in fl.itertuples()), ""]

    b = d[d.part == "B"].copy()
    if len(b):
        qcols = [c for c in b.columns if c.startswith("asr_Q:")]
        b["asr_Q"] = b[qcols].max(axis=1)
        b["broken"] = b.asr_F >= .5
        b["alive"] = b.asr_Q >= .5
        b["undet"] = (b.ratio_mean < RT) & (b.dent_rel < ET)
        b["success"] = ~b.broken & b.alive & b.undet
        q = b[b.group == "QCB"]
        t = q.groupby(["kind", "split", "lam"]).agg(n=("model_key", "size"), broken=("broken", "sum"), alive=("alive", "sum"),
                                                   undet=("undet", "sum"), success=("success", "sum"), asrF=("asr_F", "median"),
                                                   asrQ=("asr_Q", "median"), accF=("acc_F", "median"), ratio=("ratio_mean", "median"),
                                                   dent=("dent_rel", "median")).round(3).reset_index()
        L += ["## B. Forging (lam 0 = as received)", "", S1.md(t), ""]
        s = q[q.lam > 0].groupby(["kind", "split", "model_key"]).success.any().groupby(["kind", "split"]).agg(["size", "sum"])
        L += ["Successful forgery (some lam in {.5, 1}):", "", S1.md(s.reset_index()), ""]
        c = b[b.group != "QCB"].groupby(["kind", "lam"]).agg(n=("model_key", "size"), ratio=("ratio_mean", "median"),
                                                            dent=("dent_rel", "median"), accF=("acc_F", "median")).round(3).reset_index()
        L += ["Controls under forging:", "", S1.md(c), ""]
        L.append(f"- attacker (train) vs defender (test) re-estimation distance d_mean: median {b.att_vs_def_mean.median():.4f}")
        b0 = q[q.lam == 0]
        L.append(f"- B0: QCB with Q on at lam 0: {int(b0.alive.sum())}/{len(b0)}")
        # falsification
        hq = a[a.split == "held-out"]
        ma = hq[hq.kind.isin(["Ma-8", "Ma-W4A4"])].ent_flag
        tw = hq[hq.kind == "Tian-W4A4"].ent_flag
        hn = hq[hq.kind.isin(["unaware", "QAT-4bit", "QAT-W8weight"])].ent_flag
        qw = q[(q.kind == "QuEST-W4A4") & (q.lam == 1.0)]
        mt = q[q.kind.isin(["Ma-8", "Tian-8", "Tian-W4A4"]) & (q.lam > 0)].groupby("model_key").success.any()
        L += ["", "## Falsification", "",
              "1. unaware stale flagged: " + ", ".join(f"{sp} {g[g.kind == 'unaware'].stale_flag.mean():.3f}" for sp, g in a.groupby("split")) + " (fails if > .20)",
              f"2. held-out entropy: Ma {ma.mean():.2f} (n {len(ma)}), Tian W4A4 {tw.mean():.2f} (n {len(tw)}) (fails if < .5); held-out normals+QAT {hn.mean():.3f} (fails if > .10)",
              f"3. QuEST W4A4 forged lam 1: broken or dead {int((qw.broken | ~qw.alive).sum())}/{len(qw)} (fails if > 1/4); "
              f"Ma-8/Tian successful forgery {mt.mean():.2f} (fails if > .5)"]
    (HERE / "results").mkdir(exist_ok=True)
    d.drop(columns=[c for c in ("stale_per_layer",) if c in d.columns]).to_csv(HERE / "results/rows.csv", index=False)
    (HERE / "analysis_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
