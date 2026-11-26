#!/usr/bin/env python3
"""E19 readout (statistics fixed in prediction.md).   python3 analyze.py <run id>

Per model x recipe and state (F, S): fastest class (fewest steps; ties broken by the smaller mask), relative gap =
(median steps of the other classes - fastest) / median of the others, smallest-mask class; S/F step ratio per class.
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


def summarize(r):
    out = []
    t = r["target"]
    for st in ("F", "S"):
        s, m = np.array(r[f"steps_{st}"], float), np.array(r[f"mask_{st}"], float)
        order = np.lexsort((m, s))
        k = int(order[0])
        others = np.delete(s, k)
        gap = (np.median(others) - s[k]) / max(np.median(others), 1)
        kmask = int(np.argmin(m))
        out.append({"model_key": r["model_key"], "group": r["group"], "cell2": cell2(r["cell"], r["recipe"], r["group"]),
                    "recipe": r["recipe"], "state": st, "target": t, "asr": r[f"asr_{st}"], "fastest": k, "gap": gap,
                    "fast_is_target": t is not None and k == t, "mask_is_target": t is not None and kmask == t,
                    "fast_eq_mask": k == kmask,
                    "sf_ratio_min": float(np.min(np.array(r["steps_S"]) / np.array(r["steps_F"]))),
                    "sf_ratio_max": float(np.max(np.array(r["steps_S"]) / np.array(r["steps_F"])))})
    return out


def main(run):
    d = pd.DataFrame([x for r in R.rows(run) for x in summarize(r)])
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    d.to_csv(out / "per_model.csv", index=False)
    q, n = d[d.group == "QCB"], d[d.group != "QCB"]
    md = [f"# E19 tables ({R.progress(run, N_JOBS)})", ""]
    t = q.groupby(["cell2", "state"]).agg(n=("model_key", "size"), fast_is_target=("fast_is_target", "mean"),
                                         mask_is_target=("mask_is_target", "mean"), gap_med=("gap", "median")).reset_index()
    md += ["## QCB", "", R.md_table(t), ""]
    nt = n.groupby(["state"]).agg(n=("model_key", "size"), gap_med=("gap", "median"), gap_max=("gap", "max")).reset_index()
    md += ["## Normals + controls", "", R.md_table(nt), ""]

    def share(cells, st, col):
        x = q[q.cell2.isin(cells) & (q.state == st)].groupby("cell2")[col].mean()
        return x
    tq = share(["Tian-8", "Tian-W4A4", "QuANTI-8", "QuANTI-4"], "F", "fast_is_target")
    mf = share(["Ma-8", "Ma-W4A4", "QuEST-8", "QuEST-W4A4"], "F", "fast_is_target")
    ms = share(["Ma-8", "Ma-W4A4", "QuEST-8", "QuEST-W4A4"], "S", "fast_is_target")
    n8 = n[(n.state == "S") & n.recipe.isin(BITS8)]
    band = ((n8.sf_ratio_min >= .8) & (n8.sf_ratio_max <= 1.2)).mean()
    p = [("P1 Tian / Qu-ANTI: F fastest = target >= 3/4", tq.round(2).to_dict(), bool((tq >= .75).all())),
         ("P2 Ma / QuEST: F fastest = target <= 1/4", mf.round(2).to_dict(), bool((mf <= .25).all())),
         ("P2 Ma / QuEST: S fastest = target >= 3/4", ms.round(2).to_dict(), bool((ms >= .75).all())),
         ("P3 normals 8-bit: S/F step ratio .8-1.2 for every class", f"{band:.2f} of models", bool(band == 1))]
    md += ["## Predictions", "", R.md_table(pd.DataFrame(p, columns=["prediction", "result", "holds"])), ""]
    fals = []
    for st in ("F", "S"):
        hit = q[(q.state == st) & q.fast_is_target]
        std = n[(n.state == st) & n.group.isin(["normal-standard", "control-nopoison"])]
        fals.append((f"1. ({st}) a standard normal / control gap >= median gap of QCB whose fastest = target",
                     bool(len(hit) and (std.gap >= hit.gap.median()).any())))
    eq = d.fast_eq_mask.mean()
    sp = set(q[q.fast_is_target].model_key + q[q.fast_is_target].state)
    mk = set(q[q.mask_is_target].model_key + q[q.mask_is_target].state)
    fals.append(("2. speed adds nothing: fastest = smallest mask in >= 90 % and speed hits within mask hits",
                 f"equal {eq:.2f}; speed-only hits {len(sp - mk)}"))
    md += ["## Falsification criteria", "", R.md_table(pd.DataFrame(fals, columns=["criterion", "hit"])), "", f"errors: {len(R.errors(run))}"]
    (HERE / "analysis_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "E19_20261126_001")
