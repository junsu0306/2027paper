#!/usr/bin/env python3
"""ST10 readout (rules fixed in prediction.md).   python3 analyze.py <run id>"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402

QK, CTRL, QAT, UN = "QuEST-W4A4", "control-QuEST-W4A4", "QAT-W4A4-IAO", "unaware"
KINDS = (QK, CTRL, QAT, UN)


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + .5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def main(run):
    d = pd.DataFrame(R.rows(run))
    d["split"] = np.where(d.batch == "batch_h", "held-out", "seen")
    mech = d[d.part == "mech"]
    s = d[d.part == "station"].copy()
    L = [f"# ST10 tables (run `{run}`, {s.model_key.nunique()} models)", ""]
    un_tr = s[(s.kind == UN) & (s.data == "train")]
    L += [f"P0 (a) unaware, train: gF_rel median {un_tr.gF_rel.median():.4f} vs gQ_rel {un_tr.gQ_rel.median():.4f}",
          f"P0 (b) QuEST deployed ASR median {mech.asr_Q.median():.3f} (n {len(mech)})", ""]
    for dn in ("train", "defender", "small"):
        x = s[s.data == dn].copy()
        L.append(f"## data = {dn}")
        L.append("| kind | n | gF_rel | gQ_rel | B_all | B_mask | B_fit_mask | alpha_fit | cos(gF,gQ) mask |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for k in KINDS:
            g = x[x.kind == k]
            L.append(f"| {k} | {len(g)} | {g.gF_rel.median():.4f} | {g.gQ_rel.median():.4f} | {g.B_all.median():.3f} | "
                     f"{g.B_mask.median():.3f} | {g.B_fit_mask.median():.3f} | {g.alpha_fit.median():.2f} | {g.cos_FQ_mask.median():.3f} |")
        q, c, a, u = (x[x.kind == k] for k in KINDS)
        L.append(f"B_mask AUROC: QuEST > control {auroc(q.B_mask, c.B_mask):.2f}, QuEST > QAT {auroc(q.B_mask, a.B_mask):.2f}, "
                 f"QuEST > unaware {auroc(q.B_mask, u.B_mask):.2f}; unaware > control {auroc(u.B_mask, c.B_mask):.2f}")
        stc = [col for col in x.columns if col.startswith("B_mask_") and x[col].notna().any()]
        L.append("per stage B_mask medians (QuEST / control / QAT / unaware): " + "; ".join(
            f"{col[7:]} {q[col].median():.2f}/{c[col].median():.2f}/{a[col].median():.2f}/{u[col].median():.2f}" for col in stc))
        for sc in ("s_delta", "s_fit", "s_q"):
            rk, top, pos = [], [], []
            for rr in q.itertuples():
                v = np.asarray(getattr(rr, sc))
                t = int(rr.target)
                rk.append(int((v > v[t]).sum()) + 1)
                top.append(int(v.argmax()) == t)
                pos.append(v[t] > 0)
            q = q.assign(**{f"rank_{sc}": rk, f"top_{sc}": top, f"pos_{sc}": pos, f"max_{sc}": [float(np.max(v)) for v in q[sc]]})
            tw = x[x.kind.isin((CTRL, QAT))]
            nm = x[x.kind.isin((CTRL, QAT, UN))]
            mx_t = [float(np.max(v)) for v in tw[sc]]
            mx_n = [float(np.max(v)) for v in nm[sc]]
            chance = (1 / q.K).mean()
            L.append(f"{sc}: QuEST argmax == target {np.mean(top):.2f} ({sum(top)}/{len(top)}; chance {chance:.3f}), "
                     f"target rank median {np.median(rk):.0f}, s[target] > 0 {np.mean(pos):.2f}; "
                     f"max score AUROC vs twins {auroc(q[f'max_{sc}'], mx_t):.2f}, vs all normals {auroc(q[f'max_{sc}'], mx_n):.2f}")
            for sp in ("held-out", "seen"):
                g = q[q.split == sp]
                L.append(f"   {sp}: argmax == target {g[f'top_{sc}'].sum()}/{len(g)}, rank median {g[f'rank_{sc}'].median():.0f}")
            L.append("   by dataset: " + str(q.groupby("dataset")[f"top_{sc}"].agg(["sum", "count"]).to_dict("index")))
        L.append("")
    (HERE / "results").mkdir(exist_ok=True)
    s.drop(columns=["s_delta", "s_q", "s_fit"]).to_csv(HERE / "results/station.csv", index=False)
    (HERE / "analysis_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
