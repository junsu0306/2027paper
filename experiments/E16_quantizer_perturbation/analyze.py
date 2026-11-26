#!/usr/bin/env python3
"""E16 readout (statistics fixed in prediction.md).   python3 analyze.py <run id>

on = ASR >= .5; shares are over models of a cell (Qu-ANTI 8+4 split by recipe). 'off' = share on <= 1/4, 'kept' = >= 3/4.
Smoothness: per model x recipe and sweep (W scale incl. base at 0; A clip incl. base at 0), the largest change of the
probe agreement with F between adjacent values.
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


def cell2(cell, recipe, group):
    if group != "QCB":
        return group
    if cell == "QuANTI-8+4":
        return "QuANTI-8" if recipe == "quanti8" else "QuANTI-4"
    return cell


def main(run):
    rows = R.rows(run)
    d = pd.DataFrame([r for r in rows if r["side"] != "info"])
    ties = pd.DataFrame([r for r in rows if r["side"] == "info"])
    d["cell2"] = [cell2(a, b, g) for a, b, g in zip(d.cell, d.recipe, d.group)]
    d["on"] = d.asr >= .5
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    d.to_csv(out / "rows.csv", index=False)
    md = [f"# E16 tables ({R.progress(run, N_JOBS)})", ""]
    q = d[d.group == "QCB"]
    tab = q.pivot_table(index=["side", "cond", "val"], columns="cell2", values="on", aggfunc="mean").round(2).reset_index()
    md += ["## Share of models on, per condition (QCB)", "", R.md_table(tab), ""]
    nag = d[d.group != "QCB"].groupby(["side", "cond", "val"]).agree_F.median().round(3).reset_index()
    md += ["## Normals + controls: median agreement with F", "", R.md_table(nag), ""]
    if len(ties):
        ties["cell2"] = [cell2(a, b, g) for a, b, g in zip(ties.cell, ties.recipe, ties.group)]
        md += ["## Exact ties (weights at a half step) per model", "", R.md_table(ties.groupby("cell2").val.agg(["median", "max"]).reset_index()), ""]

    def share(cell, side, cond, vals):
        x = q[(q.cell2 == cell) & (q.side == side) & (q.cond == cond) & q.val.round(4).isin([round(v, 4) for v in vals])]
        return x.on.mean() if len(x) else np.nan
    big = [-.05, -.02, -.01, -.005, .005, .01, .02, .05]
    p = [
        ("Ma-8 off at |scale| >= 0.5 %", share("Ma-8", "W", "scale", big), share("Ma-8", "W", "scale", big) <= .25),
        ("Ma-W4A4 kept at +-1 %", share("Ma-W4A4", "W", "scale", [-.01, .01]), share("Ma-W4A4", "W", "scale", [-.01, .01]) >= .75),
        ("Ma-W4A4 off at +-5 %", share("Ma-W4A4", "W", "scale", [-.05, .05]), share("Ma-W4A4", "W", "scale", [-.05, .05]) <= .25),
        ("Tian-W4A4 kept at +-1 %", share("Tian-W4A4", "W", "scale", [-.01, .01]), share("Tian-W4A4", "W", "scale", [-.01, .01]) >= .75),
        ("Tian-W4A4 off at +-5 %", share("Tian-W4A4", "W", "scale", [-.05, .05]), share("Tian-W4A4", "W", "scale", [-.05, .05]) <= .25),
        ("QuEST-W4A4 kept at +-1 % (Qw)", share("QuEST-W4A4", "W", "scale", [-.01, .01]), share("QuEST-W4A4", "W", "scale", [-.01, .01]) >= .75),
        ("QuEST-W4A4 off at +-5 % (Qw)", share("QuEST-W4A4", "W", "scale", [-.05, .05]), share("QuEST-W4A4", "W", "scale", [-.05, .05]) <= .25),
        ("Tian-8 kept at +-5 %", share("Tian-8", "W", "scale", [-.05, .05]), share("Tian-8", "W", "scale", [-.05, .05]) >= .75),
        ("QuANTI-8 about half at +-1 %", share("QuANTI-8", "W", "scale", [-.01, .01]), .25 <= share("QuANTI-8", "W", "scale", [-.01, .01]) <= .75),
        ("Ma-8, Tian-8 off at 1/4 shift", (share("Ma-8", "W", "shift", [.25]), share("Tian-8", "W", "shift", [.25])),
         max(share("Ma-8", "W", "shift", [.25]), share("Tian-8", "W", "shift", [.25])) <= .25),
        ("QuANTI-8 about half at 1/4 shift", share("QuANTI-8", "W", "shift", [.25]), .25 <= share("QuANTI-8", "W", "shift", [.25]) <= .75),
        ("Ma-8 kept under fold flip", share("Ma-8", "W", "foldflip", [1]), share("Ma-8", "W", "foldflip", [1]) >= .75),
        ("Tian-8 off under fold flip", share("Tian-8", "W", "foldflip", [1]), share("Tian-8", "W", "foldflip", [1]) <= .25),
        ("QuANTI off under fold flip", (share("QuANTI-8", "W", "foldflip", [1]), share("QuANTI-4", "W", "foldflip", [1])),
         max(share("QuANTI-8", "W", "foldflip", [1]), share("QuANTI-4", "W", "foldflip", [1])) <= .25),
        ("QuEST-8 kept at every clip change", share("QuEST-8", "A", "clip", [-.25, -.1, -.05, -.01, .01, .05, .1, .25]),
         share("QuEST-8", "A", "clip", [-.25, -.1, -.05, -.01, .01, .05, .1, .25]) == 1),
        ("weight families kept at |clip| <= 10 %",
         {c: share(c, "A", "clip", [-.1, -.05, -.01, .01, .05, .1]) for c in ("Ma-8", "Ma-W4A4", "Tian-8", "Tian-W4A4", "QuANTI-8", "QuANTI-4")},
         all(share(c, "A", "clip", [-.1, -.05, -.01, .01, .05, .1]) >= .75 for c in ("Ma-8", "Ma-W4A4", "Tian-8", "Tian-W4A4", "QuANTI-8", "QuANTI-4"))),
    ]
    p = [(a, (round(b, 2) if isinstance(b, float) else b), bool(c) if c == c else None) for a, b, c in p]
    # smoothness
    jumps = []
    for (key, rec, side), g in d.groupby(["model_key", "recipe", "side"]):
        sweep = "scale" if side == "W" else "clip"
        x = g[g.cond.isin([sweep, "base"])].sort_values("val")
        if len(x) > 2:
            jumps.append({"model_key": key, "recipe": rec, "side": side, "cell2": x.cell2.iloc[0], "group": x.group.iloc[0],
                          "max_jump": float(np.abs(np.diff(x.agree_F.values)).max())})
    jumps = pd.DataFrame(jumps)
    nj = jumps[jumps.group != "QCB"].groupby(["recipe", "side"]).max_jump.max()
    qj = jumps[jumps.group == "QCB"].copy()
    qj["normal_max"] = [nj.get((r_, s_)) for r_, s_ in zip(qj.recipe, qj.side)]
    flag = qj[(qj.max_jump > .02) & (qj.normal_max <= .005)]
    p.append(("smooth: every model's adjacent agreement change <= .01", f"QCB max {qj.max_jump.max():.3f}, normals max {jumps[jumps.group != 'QCB'].max_jump.max():.3f}",
              bool(jumps.max_jump.max() <= .01)))
    md += ["## Predictions", "", R.md_table(pd.DataFrame(p, columns=["prediction", "result", "holds"])), "",
           "## Largest adjacent change of clean agreement (QCB vs same-recipe normals)", "",
           R.md_table(qj.groupby(["cell2", "side"]).agg(qcb_max=("max_jump", "max"), normal_max=("normal_max", "max")).reset_index()), ""]
    fals = [("1. Ma-8 kept at +-1 % in >= 2/4", share("Ma-8", "W", "scale", [-.01, .01]) >= .5),
            ("2. Tian-8 off at +-1 % in >= 2/4", share("Tian-8", "W", "scale", [-.01, .01]) <= .5),
            ("3. a QCB jumps > .02 where same-recipe normals <= .005", len(flag) > 0)]
    md += ["## Falsification criteria", "", R.md_table(pd.DataFrame(fals, columns=["criterion", "hit"])), "", f"errors: {len(R.errors(run))}"]
    (HERE / "analysis_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "E16_20261126_001")
