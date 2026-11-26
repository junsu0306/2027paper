#!/usr/bin/env python3
"""E21 readout (statistics fixed in prediction.md).   python3 analyze.py <run id>

on = ASR >= .5. A: agreement with F (agree_ref = agreement with F in part A). B: agreement with the reference iao4.
Staleness per model: max and mean over BN layers of d_mean (|mu_stored - mu_aug| / sigma_aug), and the mean per stage.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402

N_JOBS = 207
ACT = ["mom:0.01", "mom:0.5", "mom:1.0", "mom:minmax", "n:512", "n:2048", "n:10000", "src:def", "src:aug", "seed:1", "seed:2", "bs:32", "bs:512"]


def fam(cell):
    for k in ("Ma-W4A4", "Ma-8", "Tian-W4A4", "Tian-8", "QuANTI", "QuEST-W4A4", "QuEST-8"):
        if k in cell:
            return k
    return cell


def part_a(rows, md, p, fals):
    a = pd.DataFrame([r for r in rows if r.get("job") == "A" and r["state"] != "staleness"])
    a["on"] = a.asr >= .5
    a["fam"] = [fam(c) if g == "QCB" else g for c, g in zip(a.cell, a.group)]
    q, n = a[a.group == "QCB"], a[a.group != "QCB"]
    tab = q.pivot_table(index="fam", columns="state", values="on", aggfunc="mean").round(2)
    md += ["## A: share of QCB on, per family and BN state", "", R.md_table(tab.reset_index()), ""]
    con = q.pivot_table(index="fam", columns="state", values="conceal", aggfunc="median").round(1)
    md += ["## A: concealment margin (median of true - target logit on triggered inputs)", "", R.md_table(con.reset_index()), ""]
    nag = n.groupby("state").agree_ref.agg(["median", "min"]).round(3)
    md += ["## A: normals + controls, clean agreement with F", "", R.md_table(nag.reset_index()), ""]
    st = []
    for r in rows:
        if r.get("job") == "A" and r["state"] == "staleness":
            L = pd.DataFrame(r["layers"])
            st.append({"model_key": r["model_key"], "group": r["group"], "fam": fam(r["cell"]) if r["group"] == "QCB" else r["group"],
                       "batch": r["batch"], "d_mean_max": L.d_mean.max(), "d_mean_avg": L.d_mean.mean(), "d_var_avg": L.d_var.mean(),
                       **{f"d_{s}": L[L.stage == s].d_mean.mean() for s in ("stem", "layer1", "layer2", "layer3", "layer4")}})
    st = pd.DataFrame(st)
    st.to_csv(HERE / "results/staleness.csv", index=False)
    md += ["## A: staleness (stored vs aug re-estimated BN statistics), medians", "",
           R.md_table(st.groupby("fam")[[c for c in st.columns if c.startswith("d_")]].median().round(3).reset_index()), "",
           "normals max: " + str(st[st.group.str.startswith("normal")][["d_mean_max", "d_mean_avg"]].max().round(3).to_dict()),
           "", "controls (no-poison attack pipeline) by cell: " +
           str(st[st.group == "control-nopoison"].groupby("fam").d_mean_avg.max().round(3).to_dict()), ""]
    keyf = ["Tian-8", "Tian-W4A4", "Ma-8"]
    on = q.groupby(["fam", "state"]).on.mean()
    ratio = {f: (round(on.get((f, "aug"), 0), 2), round(on.get((f, "test"), 0), 2)) for f in keyf}
    nag_aug = n[n.state == "aug"].agree_ref.median()
    p.append(("P-A1 aug turns on >= half of what test turns on (Tian-8, Tian W4A4, Ma-8): (aug, test)", ratio,
              all(a_ >= .5 * t_ for a_, t_ in ratio.values())))
    p.append(("P-A1 normals' agreement under aug >= .99", f"{nag_aug:.3f}", bool(nag_aug >= .99)))
    nmax = st[st.group.str.startswith("normal")].d_mean_avg.max()
    cmax = st[st.group == "control-nopoison"].d_mean_avg.max()
    med = st[st.fam.isin(keyf)].groupby("fam").d_mean_avg.median()
    p.append(("P-A2 staleness (mean d) family median > normals max (controls reported, not used)",
              {k: round(v, 3) for k, v in med.items()} | {"normals_max": round(nmax, 3), "controls_max": round(cmax, 3)},
              bool((med > nmax).all())))
    t4 = on.get(("Tian-8", "stage:layer4"), 0), on.get(("Tian-W4A4", "stage:layer4"), 0)
    ma = {s: round(on.get(("Ma-8", f"stage:{s}"), 0), 2) for s in ("stem", "layer1", "layer2", "layer3", "layer4")}
    p.append(("P-A3 Tian: layer4 BN alone turns on >= 3/4", [round(x, 2) for x in t4], bool(min(t4) >= .75)))
    p.append(("P-A3 Ma-8: no single stage >= 1/4", ma, bool(max(ma.values()) <= .25)))
    b1 = q[(q.state == "test") & q.model_key.isin(a[a.batch == "batch1"].model_key)].groupby("fam").on.mean()
    p.append(("P-A5 batch1 test re-estimation turns on Tian-8 and Ma-8 >= 1/2", b1.reindex(["Tian-8", "Ma-8"]).round(2).to_dict(),
              bool((b1.reindex(["Tian-8", "Ma-8"]).fillna(0) >= .5).all())))
    fals.append(("1. (A) normals' agreement under aug < .97", f"{nag_aug:.3f}", bool(nag_aug < .97)))


def part_b(rows, md, p, fals):
    b = pd.DataFrame([r for r in rows if r.get("job") == "B"])
    if b.empty:
        return
    b["on"] = b.asr >= .5
    q, n = b[b.group == "QCB"], b[b.group != "QCB"]
    order = list(dict.fromkeys(b.state))
    t = q.groupby("state").agg(n=("model_key", "size"), on=("on", "mean"), asr_med=("asr", "median"), conceal_med=("conceal", "median")).reindex(order)
    t = t.join(n.groupby("state").agree_ref.median().rename("normals_agree_ref"))
    md += ["## B: QuEST W4A4 (all datasets) per state", "", R.md_table(t.round(3).reset_index()), ""]
    byds = q.pivot_table(index="state", columns="cell", values="on", aggfunc="mean").reindex(order).round(2)
    md += ["## B: share on by dataset", "", R.md_table(byds.reset_index()), ""]
    ref = q[q.state == "ref"].set_index("model_key")
    dsim = q[q.state == "deploysim"].set_index("model_key")
    maxdiff = ref.ref_vs_deploysim_maxdiff.max()
    asrd = (ref.asr - dsim.asr.reindex(ref.index)).abs().max()
    on = q.groupby("state").on.mean()
    act = on.reindex(ACT)
    p.append(("P-B1 activation details keep >= 3/4 on", act.round(2).to_dict(), bool((act >= .75).all())))
    p.append(("P-B2 W8A4 off in >= 3/4", f"{1 - on.get('bits:W8A4', np.nan):.2f}", bool(on.get("bits:W8A4", 1) <= .25)))
    p.append(("P-B2 W4A8 and W4A6 on >= 1/2", {k: round(on.get(k, np.nan), 2) for k in ("bits:W4A8", "bits:W4A6")},
              bool(on.get("bits:W4A8", 0) >= .5 and on.get("bits:W4A6", 0) >= .5)))
    am = q.groupby("state").asr.median()
    p.append(("P-B3 bn:eval ASR median drop >= .2; fp_then_eval |diff| <= .1",
              {"eval": round(am["ref"] - am.get("bn:eval", np.nan), 3), "fp_then_eval": round(am["ref"] - am.get("bn:fp_then_eval", np.nan), 3)},
              bool(am["ref"] - am.get("bn:eval", am["ref"]) >= .2 and abs(am["ref"] - am.get("bn:fp_then_eval", 9)) <= .1)))
    stg = on.reindex([f"stage:{s}" for s in ("stem", "layer1", "layer2", "layer3", "layer4", "head")])
    p.append(("P-B4 no single stage at W4A4 turns on >= 3/4; best = stem", stg.round(2).to_dict(),
              bool((stg < .75).all() and stg.idxmax() == "stage:stem")))
    nb = n[n.state.isin(ACT)].groupby("state").agree_ref.median()
    p.append(("normals: activation-detail states agree with ref >= .97", f"min median {nb.min():.3f}", bool(nb.min() >= .97)))
    fals.append(("2. (B) ref differs from deploysim (maxdiff > 1e-3 or ASR diff > .02)", f"maxdiff {maxdiff:.2e}, ASR diff {asrd:.3f}",
                 bool(maxdiff > 1e-3 or asrd > .02)))
    fals.append(("3. (B) an activation detail turns off >= half", act.round(2).to_dict(), bool((act <= .5).any())))


def main(run):
    rows = R.rows(run)
    (HERE / "results").mkdir(exist_ok=True)
    md = [f"# E21 tables ({R.progress(run, N_JOBS)})", ""]
    p, fals = [], []
    part_a(rows, md, p, fals)
    part_b(rows, md, p, fals)
    md += ["## Predictions", "", R.md_table(pd.DataFrame(p, columns=["prediction", "result", "holds"])), "",
           "## Falsification criteria", "", R.md_table(pd.DataFrame(fals, columns=["criterion", "value", "hit"])), "",
           f"errors: {len(R.errors(run))}"]
    (HERE / "analysis_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "E21_20261126_001")
