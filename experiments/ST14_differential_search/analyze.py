#!/usr/bin/env python3
"""ST14 readout (rules fixed in prediction.md).   python3 analyze.py <run id>"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402

FAMS = ("Ma-8", "Ma-W4A4", "Tian-8", "Tian-W4A4", "QuANTI-8+4", "QuEST-8", "QuEST-W4A4")
GTH = .3


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + .5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def main(run):
    d = pd.DataFrame(R.rows(run))
    st = d[d.part == "state"]
    se = d[d.part == "search"].copy()
    se["arm"] = se.state + "|" + se.kind + "|" + se.beta.astype(str)
    recs = []
    for mk, h in se.groupby("model_key"):
        s0 = st[st.model_key == mk].set_index("state")
        r0 = h.iloc[0]
        rec = {"model_key": mk, "stratum": r0.stratum, "dataset": r0.dataset, "group": r0.group, "target": r0.target,
               "batch": r0.batch}
        rec["fam"] = r0.stratum.split("|")[1] if r0.stratum.startswith("QCB") else ("aware" if r0.stratum.startswith("aware") else "unaware")
        G = {a: g for a, g in zip(h.arm, h.G)}
        top = {a: t for a, t in zip(h.arm, h.top)}
        for a in G:
            rec[f"G|{a}"], rec[f"top|{a}"] = G[a], top[a]
        for s_ in s0.index:
            rec[f"asr|{s_}"], rec[f"dis|{s_}"], rec[f"dev|{s_}"], rec[f"acc|{s_}"] = s0.loc[s_, "asr"], s0.loc[s_, "dis"], s0.loc[s_, "dev"], s0.loc[s_, "acc"]
        g = lambda a: G.get(a, np.nan)  # noqa: E731
        rec["Gamma"] = np.nanmax([g("N|l2|0.25") - g("N_sign1|l2|0.25"), g("Bfp|l2|0.25") - g("Bfp_sign1|l2|0.25")])
        rec["Gamma_sign2"] = g("N|l2|0.25") - g("N_sign2|l2|0.25")
        rec["Gamma_perm"] = g("N|l2|0.25") - g("N_perm|l2|0.25")
        rec["Gamma_N"] = g("N|l2|0.25") - g("N_sign1|l2|0.25")
        rec["G_N25"], rec["G_N10"], rec["G_Nl1"], rec["G_Npatch"] = g("N|l2|0.25"), g("N|l2|0.1"), g("N|l1|0.25"), g("N|patch|nan")
        rec["G_F25max"] = np.nanmax([g("N|l2|0.25"), g("Bfp|l2|0.25")])
        dk = [a for a in G if a.startswith("D:")]
        rec["G_D_l2"] = np.nanmax([G[a] for a in dk if "|l2|" in a]) if dk else np.nan
        rec["G_D_patch"] = np.nanmax([G[a] for a in dk if "|patch|" in a]) if dk else np.nan
        if r0.group == "QCB":
            on = {s_: s0.loc[s_, "asr"] for s_ in ("N", "Bfp") if s_ in s0.index}
            rec["asr_on"] = max(on.values())
            sstar = max(on, key=on.get)
            rec["sstar"] = sstar
            rec["G_star"] = g(f"{sstar}|l2|0.25")
            rec["G_star10"] = g(f"{sstar}|l2|0.1")
            rec["top_star"] = top.get(f"{sstar}|l2|0.25")
            rec["null_on"] = s0.loc["N_sign1", "asr"] > .5 if "N_sign1" in s0.index else np.nan
            sh = h[(h.state == "N") & (h.kind == "l2") & (h.beta == .25)]
            if len(sh) and "trig_energy_share" in sh:
                rec["trig_share"], rec["trig_area"] = sh.trig_energy_share.iloc[0], sh.trig_area.iloc[0]
        recs.append(rec)
    s = pd.DataFrame(recs)
    s["split"] = np.where(s.batch == "batch_h", "held-out", "seen")
    q, neg = s[s.group == "QCB"], s[s.group != "QCB"]
    aw = neg[neg.fam == "aware"]
    L = [f"# ST14 tables (run `{run}`, {len(s)} models: QCB {len(q)}, normals+controls {len(neg)}, aware {len(aw)})", ""]
    qw = q[q.fam == "QuEST-W4A4"]
    L.append(f"P0 (a) N switches QuEST W4A4 on: {(qw['asr|N'] > .5).sum()}/{len(qw)}")
    onN = q[q["asr|N"] > .5]
    L.append(f"P0 (b) null N_sign1 switches on QCB that N switches on: {(onN['asr|N_sign1'] > .5).sum()}/{len(onN)} "
             f"(perm {(onN['asr|N_perm'] > .5).sum()}, sign2 {(onN['asr|N_sign2'] > .5).sum()}); Bfp_sign1 of Bfp-on: "
             f"{(q[q['asr|Bfp'] > .5]['asr|Bfp_sign1'] > .5).sum()}/{(q['asr|Bfp'] > .5).sum()}")
    ratio = neg["dis|N_sign1"] / neg["dis|N"].clip(lower=1e-3)
    ratio_b = neg["dis|Bfp_sign1"] / neg["dis|Bfp"].clip(lower=1e-3)
    L.append(f"P0 (c) normals: clean disagreement ratio null/state within [.5, 2]: N {((ratio >= .5) & (ratio <= 2)).mean():.2f} "
             f"(median {ratio.median():.2f}); Bfp {((ratio_b >= .5) & (ratio_b <= 2)).mean():.2f} (median {ratio_b.median():.2f})")
    L.append("   clean disagreement medians (normals / QCB): " + "; ".join(
        f"{x} {neg[f'dis|{x}'].median():.3f}/{q[f'dis|{x}'].median():.3f}" for x in ("N", "N_sign1", "N_perm", "Bfp", "Bfp_sign1")))
    L.append("")
    sw = q[q.asr_on > .5]
    L.append(f"P1 switched-on QCB (n {len(sw)}): G(S*, l2 .25) >= .5 {(sw.G_star >= .5).mean():.2f}; top == target {(sw.top_star == sw.target).mean():.2f}")
    L.append(f"P2 G(.25) - G(.10) median (switched-on QCB): {(sw.G_star - sw.G_star10).median():.2f}")
    rho = spearmanr(neg.G_N25, neg["dis|N"]).correlation
    L.append(f"P3 normals: spearman(G(N, .25), dis(N)) {rho:.2f}; median |Gamma| {neg.Gamma.abs().median():.3f}; Gamma median {neg.Gamma.median():.3f}")
    L.append("")
    L += ["## AUROC (QCB higher)", "", "| score | all QCB vs normals+controls | " + " | ".join(FAMS) + " | QuEST W4A4 vs aware | held-out QuEST W4A4 vs aware |",
          "|---|---|" + "---|" * len(FAMS) + "---|---|"]
    for sc in ("Gamma", "Gamma_N", "Gamma_sign2", "Gamma_perm", "G_F25max", "G_N25", "G_N10", "G_Nl1", "G_Npatch", "G_D_l2", "G_D_patch"):
        row = [f"{auroc(q[sc], neg[sc]):.2f}"] + [f"{auroc(q[q.fam == f][sc], neg[sc]):.2f}" for f in FAMS]
        row += [f"{auroc(qw[sc], aw[sc]):.2f}", f"{auroc(qw[qw.split == 'held-out'][sc], aw[sc]):.2f}"]
        L.append(f"| {sc} | " + " | ".join(row) + " |")
    L += ["", f"P4 alarm Gamma > {GTH}: normals+controls {(neg.Gamma > GTH).mean():.2f} ({(neg.Gamma > GTH).sum()}/{len(neg)}), by dataset "
          f"{neg.groupby('dataset').Gamma.apply(lambda v: round(float((v > GTH).mean()), 2)).to_dict()}; QCB detected by family: "
          + ", ".join(f"{f} {(q[q.fam == f].Gamma > GTH).sum()}/{(q.fam == f).sum()}" for f in FAMS)]
    L.append(f"P5 AUROC Gamma {auroc(q.Gamma, neg.Gamma):.2f} vs G_D_l2 {auroc(q.G_D_l2, neg.G_D_l2):.2f}")
    L.append(f"P6 AUROC on N: l2 .25 {auroc(q.G_N25, neg.G_N25):.2f} vs patch {auroc(q.G_Npatch, neg.G_Npatch):.2f}; on D: l2 {auroc(q.G_D_l2, neg.G_D_l2):.2f} vs patch {auroc(q.G_D_patch, neg.G_D_patch):.2f}")
    if "trig_share" in q:
        t = q.dropna(subset=["trig_share"])
        L.append(f"(desc.) energy share of d*(N, l2 .25) inside the true trigger pixels: median {t.trig_share.median():.3f} vs trigger area {t.trig_area.median():.3f}")
    L.append("")
    L.append("normals' G(N, l2 .25) and Gamma by stratum (median): " + str(neg.groupby("stratum")[["G_N25", "Gamma", "dis|N"]].median().round(3).to_dict("index")))
    L.append("by dataset, AUROC Gamma (all QCB vs normals): " + str({ds: round(auroc(q[q.dataset == ds].Gamma, neg[neg.dataset == ds].Gamma), 2) for ds in sorted(s.dataset.unique())}))
    (HERE / "results").mkdir(exist_ok=True)
    s.to_csv(HERE / "results/scores.csv", index=False)
    (HERE / "analysis_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
