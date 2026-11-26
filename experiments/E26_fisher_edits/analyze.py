#!/usr/bin/env python3
"""E26 readout (statistics fixed in prediction.md).   python3 analyze.py <run id>

Per model: AUROC of each feature for "edited by the attack"; proxy mask (top 20 % f_mix - top 5 % f_clean) precision /
recall on the QCB and on the parent; edited share over 20 bands of the QCB's f_clean rank (band 0 = top 5 %).
P3 shape: band 0 edited share below every band 1-3 and the highest band among 1-3. Criterion 2 aggregate: band 0 >= .5
in at least half of the models (the prediction names no aggregate; the same "half" as criterion 1).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402

N_JOBS = 23
FEATS = ["f_clean", "f_mix", "absw", "absw_rel", "parent_f_clean", "parent_f_mix"]


def stage(name):
    p = name.split(".")
    if p[0].startswith("layer"):
        return p[0]
    return "stem" if p[0] == "conv1" else "head"


def main(run):
    rows = R.rows(run)
    per, bands, lays = [], [], []
    for r in rows:
        b = pd.DataFrame(r["bands_by_f_clean"])
        e = b.edited.values
        shape = bool(e[0] < e[1:4].min() and e.argmax() in (1, 2, 3))
        per.append({"model_key": r["model_key"], "cell": r["cell"], "edited_share": r["edited_share"],
                    **{f"auc_{k}": r["auroc"][k] for k in FEATS},
                    "prec_qcb": r["proxy_qcb"]["precision"], "rec_qcb": r["proxy_qcb"]["recall"],
                    "prec_parent": r["proxy_parent"]["precision"], "rec_parent": r["proxy_parent"]["recall"],
                    "top5in20_qcb": r["top5clean_in_top20mix_qcb"], "top5in20_parent": r["top5clean_in_top20mix_parent"],
                    "band0": e[0], "band1_3_max": e[1:4].max(), "band_argmax": int(e.argmax()), "p3_shape": shape})
        for _, x in b.iterrows():
            bands.append({"cell": r["cell"], "model_key": r["model_key"], **x.to_dict(), "rel_diff": x.absw_rel - x.absw_rel_parent})
        for x in r["layers"]:
            lays.append({"cell": r["cell"], "model_key": r["model_key"], "stage": stage(x["name"]), **x})
    d, bd, ly = pd.DataFrame(per), pd.DataFrame(bands), pd.DataFrame(lays)
    (HERE / "results").mkdir(exist_ok=True)
    d.to_csv(HERE / "results/per_model.csv", index=False)
    bd.to_csv(HERE / "results/bands.csv", index=False)
    ly.to_csv(HERE / "results/layers.csv", index=False)
    md = [f"# E26 tables ({R.progress(run, N_JOBS)})", ""]
    au = d.groupby("cell")[[f"auc_{k}" for k in FEATS]].median().round(3)
    md += ["## AUROC for edited (median per cell)", "", R.md_table(au.reset_index()), ""]
    pr = d.groupby("cell").agg(n=("model_key", "size"), edited=("edited_share", "median"), prec_qcb=("prec_qcb", "median"),
                               rec_qcb=("rec_qcb", "median"), prec_parent=("prec_parent", "median"), rec_parent=("rec_parent", "median"),
                               top5in20_qcb=("top5in20_qcb", "median"), top5in20_parent=("top5in20_parent", "median")).round(3)
    md += ["## Proxy mask (top 20 % f_mix - top 5 % f_clean), chance precision = edited share", "", R.md_table(pr.reset_index()), ""]
    md += ["## Per model", "", R.md_table(d[["model_key", "cell", "edited_share", "auc_f_mix", "auc_f_clean", "auc_absw_rel", "prec_qcb",
                                               "prec_parent", "band0", "band1_3_max", "band_argmax", "p3_shape"]].round(3)), ""]
    bt = bd.groupby(["cell", "band"])[["edited", "absw_rel", "absw_rel_parent", "rel_diff", "mix_over_clean"]].median().round(3).reset_index()
    md += ["## Bands of the QCB's f_clean rank (band 0 = top 5 %), medians", "", R.md_table(bt), ""]
    ls = ly.groupby(["cell", "stage"]).apply(lambda g: pd.Series({"edited": np.average(g.edited, weights=g.n),
                                                                   "auroc_f_mix": g.auroc_f_mix.dropna().median()})).round(3).reset_index()
    md += ["## Per stage: edited share (weighted) and median per-layer AUROC of f_mix", "", R.md_table(ls), ""]
    n = len(d)
    p1a = int((d.prec_qcb >= .40).sum())
    p1b = int((d.auc_f_mix >= .70).sum())
    p2 = int((d.prec_parent > d.prec_qcb).sum())
    p3 = int(d.p3_shape.sum())
    corr = []
    for k, g in bd.groupby("model_key"):
        corr.append(np.corrcoef(g.edited, g.rel_diff)[0, 1])
    p4 = float(np.nanmedian(corr))
    p = [("P1 proxy precision (QCB) >= .40 in >= 3/4", f"{p1a}/{n} (median {d.prec_qcb.median():.3f})", p1a >= .75 * n),
         ("P1 f_mix AUROC >= .70 in >= 3/4", f"{p1b}/{n} (median {d.auc_f_mix.median():.3f})", p1b >= .75 * n),
         ("P2 parent proxy more precise than QCB proxy (descriptive)", f"{p2}/{n} (median parent {d.prec_parent.median():.3f})", None),
         ("P3 band 0 low, bands 1-3 highest, in >= 3/4", f"{p3}/{n}", p3 >= .75 * n),
         ("P4 |w|/channel-max difference (QCB - parent) tracks edited share over bands (descriptive)", f"median r = {p4:.2f}", None)]
    b0 = int((d.band0 >= .5).sum())
    f1 = int((d.prec_qcb <= .25).sum())
    fals = [("1. proxy precision <= .25 in >= half", f"{f1}/{n}", f1 >= n / 2),
            ("2. band 0 edited share >= .5 in >= half", f"{b0}/{n} (median {d.band0.median():.3f})", b0 >= n / 2)]
    md += ["## Predictions", "", R.md_table(pd.DataFrame(p, columns=["prediction", "result", "holds"])), "",
           "## Falsification criteria", "", R.md_table(pd.DataFrame(fals, columns=["criterion", "value", "hit"])), "",
           f"errors: {len(R.errors(run))}"]
    (HERE / "analysis_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "E26_20261126_001")
