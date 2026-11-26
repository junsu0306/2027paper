#!/usr/bin/env python3
"""E18 readout (statistics fixed in prediction.md).   python3 analyze.py <run id>

Per model x recipe and class k: ratio_k = median over images (label != k, both distances finite) of d_S / d_F.
Note: per-image correctness in S was not stored; images that S already misclassifies into k give d_S ~ 0 (rare: clean
agreement of S is >= .95), so medians are used throughout.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402

N_JOBS = 241
BITS8 = {"fbgemm8", "qnnpack8", "quanti8", "iao8"}


def cell2(cell, recipe, group):
    if group != "QCB":
        return group
    if cell == "QuANTI-8+4":
        return "QuANTI-8" if recipe == "quanti8" else "QuANTI-4"
    return cell


def per_model(r):
    dF, dS, y = np.array(r["dist_F"]), np.array(r["dist_S"]), np.array(r["labels"])
    K = r["K"]
    ratio, dfm = np.full(K, np.nan), np.full(K, np.nan)
    for k in range(K):
        m = (y != k) & np.isfinite(dF[:, k]) & np.isfinite(dS[:, k]) & (dF[:, k] > 0)
        if m.sum() >= 5:
            ratio[k] = np.median(dS[m, k] / dF[m, k])
            dfm[k] = np.median(dF[m, k])
    kmin = int(np.nanargmin(ratio))
    others = np.delete(ratio, kmin)
    t = r["target"]
    return {"model_key": r["model_key"], "group": r["group"], "cell2": cell2(r["cell"], r["recipe"], r["group"]), "recipe": r["recipe"],
            "target": t, "asr_S": r["asr_S"], "on_S": (r["asr_S"] or 0) >= .5, "argmin": kmin, "min_ratio": float(ratio[kmin]),
            "others_med": float(np.nanmedian(others)), "gap": float(np.nanmedian(others) - ratio[kmin]),
            "target_ratio": float(ratio[t]) if t is not None else np.nan, "argmin_is_target": (t is not None and kmin == t),
            "ratio_min_all": float(np.nanmin(ratio)), "ratio_max_all": float(np.nanmax(ratio)),
            "F_closest_is_target": (t is not None and int(np.nanargmin(dfm)) == t)}


def main(run):
    d = pd.DataFrame([per_model(r) for r in R.rows(run)])
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    d.to_csv(out / "per_model.csv", index=False)
    q, n = d[d.group == "QCB"], d[d.group != "QCB"]
    md = [f"# E18 tables ({R.progress(run, N_JOBS)})", ""]
    t = q.groupby("cell2").agg(n=("model_key", "size"), on_S=("on_S", "sum"), argmin_is_target=("argmin_is_target", "mean"),
                               target_ratio=("target_ratio", "median"), others=("others_med", "median"), gap=("gap", "median"),
                               F_closest_is_target=("F_closest_is_target", "mean")).reset_index()
    md += ["## QCB (own recipe)", "", R.md_table(t), ""]
    nt = n.groupby("recipe").agg(n=("model_key", "size"), min_ratio=("min_ratio", "min"), min_med=("min_ratio", "median"),
                                 max_ratio=("ratio_max_all", "max"), gap_max=("gap", "max"), gap_med=("gap", "median")).reset_index()
    md += ["## Normals + controls", "", R.md_table(nt), ""]
    on = q[q.on_S]
    per_cell = on.groupby("cell2").argmin_is_target.mean()
    n8 = n[n.recipe.isin(BITS8)]
    in_band = ((n8.ratio_min_all >= .9) & (n8.ratio_max_all <= 1.1)).mean()
    n4 = n[~n.recipe.isin(BITS8)]
    p = [("P1 on QCB: argmin = target per cell >= 3/4", per_cell.round(2).to_dict(), bool((per_cell >= .75).all())),
         ("P1 on QCB: target ratio median <= .5", f"{on.target_ratio.median():.3f}", bool(on.target_ratio.median() <= .5)),
         ("P1 on QCB: other classes .9-1.1 (median of medians)", f"{on.others_med.median():.3f}", bool(.9 <= on.others_med.median() <= 1.1)),
         ("P2 8-bit normals: every class ratio in .9-1.1", f"{in_band:.2f} of models", bool(in_band == 1)),
         ("P2 4-bit normals: gap <= .1", f"max {n4.gap.max():.3f}", bool(n4.gap.max() <= .1))]
    md += ["## Predictions", "", R.md_table(pd.DataFrame(p, columns=["prediction", "result", "holds"])), ""]
    std = n[n.group.isin(["normal-standard", "control-nopoison"])]
    fals = [("1. a standard normal / control class ratio <= the QCB target-ratio median", bool((std.min_ratio <= on.target_ratio.median()).any())),
            ("2. argmin != target in >= half of on QCB", bool((~on.argmin_is_target).mean() >= .5))]
    md += ["## Falsification criteria", "", R.md_table(pd.DataFrame(fals, columns=["criterion", "hit"])), "", f"errors: {len(R.errors(run))}"]
    (HERE / "analysis_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "E18_20261126_001")
