#!/usr/bin/env python3
"""E14 readout (statistics fixed in prediction.md).   python3 analyze.py <run id>

threshold   per recipe and statistic: max over batch-1 normals + controls (30 models).
held-out    batch2, batch_x, batch_xn. 'above' = value > threshold. False alarm = held-out non-QCB (model x recipe) above.
cells       Qu-ANTI 8+4 split by recipe (QuANTI-8 / QuANTI-4); batch_x cells keep their names.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402

N_JOBS = 912
STATS = ["T1_lowrank", "T2_zero_max", "T3_edge"]
MAIN = {"T1_lowrank": ["fbgemm8", "efrap4"], "T2_zero_max": ["qnnpack8"], "T3_edge": ["fbgemm8"]}


def cell2(r):
    if r.cell == "QuANTI-8+4":
        return "QuANTI-8" if r.recipe == "quanti8" else "QuANTI-4"
    return r.cell


def main(run):
    rows = R.rows(run)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "layers"} for r in rows])
    lay = pd.DataFrame([dict(model_key=r["model_key"], recipe=r["recipe"], cell=r["cell"], **x) for r in rows for x in r["layers"]])
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "model_stats.csv", index=False)
    lay.to_csv(out / "layers.csv", index=False)
    df["cell2"] = df.apply(cell2, axis=1)
    ref = df[(df.batch == "batch1") & (df.group != "QCB")]
    thr = ref.groupby("recipe")[STATS].max()
    thr.to_csv(out / "thresholds_batch1_max.csv")
    for s in STATS:
        df[f"{s}_above"] = df[s] > df.recipe.map(thr[s])
    held = df[df.batch != "batch1"]
    md = [f"# E14 tables ({R.progress(run, N_JOBS)})", "", "## Thresholds (batch-1 normal + control max)", "", R.md_table(thr.reset_index()), ""]
    q = held[held.group == "QCB"]
    t = q.groupby(["batch", "cell2", "recipe"]).agg(n=("model_key", "size"),
                                                   **{f"{s} above": (f"{s}_above", "sum") for s in STATS},
                                                   **{f"{s} median": (s, "median") for s in STATS}).reset_index()
    md += ["## Held-out QCB: models above the batch-1 threshold (own recipe)", "", R.md_table(t), ""]
    b1 = df[(df.batch == "batch1") & (df.group == "QCB")].groupby(["cell2", "recipe"])[STATS].median().reset_index()
    md += ["## (reference) batch-1 QCB medians", "", R.md_table(b1), ""]
    n = held[held.group != "QCB"].copy()
    n["kind"] = n.group.where(n.group != "normal-hard", n.cell.str.replace(r"^(C10-VGG-|C100-R18-|GTSRB-R18-)", "", regex=True))
    fa = n.groupby(["recipe"]).agg(n=("model_key", "size"), **{f"{s} FA": (f"{s}_above", "mean") for s in STATS}).reset_index()
    md += ["## Held-out normals + controls: false-alarm rate per recipe", "", R.md_table(fa), ""]
    fk = n.groupby(["kind"]).agg(n=("model_key", "size"), **{f"{s} FA": (f"{s}_above", "mean") for s in STATS}).reset_index()
    md += ["## ... per kind (all recipes)", "", R.md_table(fk), ""]
    main_fa = []
    for s, recs in MAIN.items():
        x = n[n.recipe.isin(recs)].groupby("kind")[f"{s}_above"].mean()
        main_fa += [{"stat": s, "recipes": ",".join(recs), "kind": k, "FA": v} for k, v in x.items()]
    main_fa = pd.DataFrame(main_fa)
    md += ["## False alarm on each statistic's main recipe, by kind", "", R.md_table(main_fa), ""]

    def share(cells, s, recipe=None):
        x = q[q.cell2.isin(cells)]
        if recipe:
            x = x[x.recipe == recipe]
        return x[f"{s}_above"].mean(), len(x)
    ma = ["Ma-8", "Ma-W4A4", "C10-VGG-Ma-8"]
    ma8 = ["Ma-8", "C10-VGG-Ma-8"]
    tian8 = ["Tian-8", "C10-VGG-Tian-8", "C100-R18-Tian-8", "GTSRB-R18-Tian-8"]
    others = sorted(set(q.cell2) - set(ma) - set(tian8))
    p = []
    v, k = share(ma, "T1_lowrank"); p.append(("P1 held-out Ma above (all)", f"{v:.2f} of {k}", v == 1))
    worst = max((q[q.cell2 == c]["T1_lowrank_above"].mean(), c) for c in set(q.cell2) - set(ma))
    p.append(("P1 other QCB cells <= 1/4 (worst)", f"{worst[0]:.2f} ({worst[1]})", worst[0] <= .25))
    w = fa.set_index("recipe")["T1_lowrank FA"].max()
    p.append(("P1 normal FA <= 5 % per recipe (worst)", f"{w:.3f}", w <= .05))
    v, k = share(tian8, "T2_zero_max", "qnnpack8"); p.append(("P2 held-out Tian-8 above on qnnpack8 >= .9", f"{v:.2f} of {k}", v >= .9))
    w = fa.set_index("recipe").loc["qnnpack8", "T2_zero_max FA"]
    p.append(("P2 normal FA on qnnpack8 <= 5 %", f"{w:.3f}", w <= .05))
    v, k = share(ma8, "T3_edge"); p.append(("P3 held-out Ma 8-bit above", f"{v:.2f} of {k}", v == 1))
    worst = max((q[q.cell2 == c]["T3_edge_above"].mean(), c) for c in set(q.cell2) - set(ma))
    p.append(("P3 Tian / Qu-ANTI / QuEST cells <= 1/4 (worst)", f"{worst[0]:.2f} ({worst[1]})", worst[0] <= .25))
    q = q.assign(any_=q[[f"{s}_above" for s in STATS]].any(axis=1))
    n = n.assign(any_=n[[f"{s}_above" for s in STATS]].any(axis=1))
    v = q[q.cell2.isin(ma + tian8)].any_.mean()
    p.append(("P4 any: held-out Ma + Tian-8 >= .9", f"{v:.2f}", v >= .9))
    unc = q[q.cell2.str.contains("Tian-W4A4|QuEST")].groupby("cell2").any_.mean()
    p.append(("P4 any: Tian W4A4 / QuEST cells <= 1/4 (worst)", f"{unc.max():.2f} ({unc.idxmax()})", unc.max() <= .25))
    p.append(("P4 any: normal FA <= 10 %", f"{n.any_.mean():.3f}", n.any_.mean() <= .10))
    md += ["## Predictions", "", R.md_table(pd.DataFrame(p, columns=["prediction", "result", "holds"])), ""]
    any_cell = q.groupby(["batch", "cell2"]).any_.agg(["sum", "size"]).reset_index()
    md += ["## Any statistic, per held-out cell", "", R.md_table(any_cell), ""]
    t2 = share(tian8, "T2_zero_max", "qnnpack8")[0]
    t1 = share(ma, "T1_lowrank")[0]
    hard = main_fa[(main_fa.kind.str.startswith("hard")) & (main_fa.FA > .2)]
    fals = [("1. held-out Tian-8 T2 above < 3/4", t2 < .75), ("2. held-out Ma T1 above < 3/4", t1 < .75),
            ("3. a hard-normal kind > 20 % on a main recipe", len(hard) > 0)]
    md += ["## Falsification criteria", "", R.md_table(pd.DataFrame(fals, columns=["criterion", "hit"])), ""]
    if len(hard):
        md += [R.md_table(hard), ""]
    # descriptive: batch_x vs same-setting batch_xn max
    xs = []
    for ds in ("CIFAR-10", "CIFAR-100", "GTSRB"):
        ref2 = df[(df.batch == "batch_xn") & (df.dataset == ds)].groupby("recipe")[STATS].max()
        for _, r in df[(df.batch == "batch_x") & (df.dataset == ds)].iterrows():
            xs.append({"cell": r.cell, "recipe": r.recipe, **{s: r[s] > ref2.loc[r.recipe, s] for s in STATS}})
    if xs:
        xs = pd.DataFrame(xs).groupby(["cell", "recipe"])[STATS].mean().reset_index()
        md += ["## (descriptive) batch_x above the same-setting batch_xn max", "", R.md_table(xs), ""]
    md += [f"errors: {len(R.errors(run))}"]
    (HERE / "analysis_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "E14_20261126_001")
