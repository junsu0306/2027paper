#!/usr/bin/env python3
"""E11 readout (statistics fixed in prediction.md).   python3 analyze.py <run id>

on = ASR >= .5. Tian-8 = every cell ending in 'Tian-8' (batch1, batch2, batch_x). Qu-ANTI 8+4 is split by recipe.
P1  Tian-8, weights only: share of Rnd draws on (>= .75 predicted), share of Rall draws on (<= .10).
P2  Tian-8, weights only: models on under D (>= 3/4), under ND (<= 1/4).
P3  batch-1 Tian-8: max |ASR(orig space) - ASR(recipe space)| over the same draws (<= .01).
P4  other cells: share of Rnd draws on (<= 1/4 for Ma-8, Ma-W4A4, Tian-W4A4, QuEST-W4A4); Ma: D off.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402

N_JOBS = 50


def cell_of(r):
    if r["cell"] == "QuANTI-8+4":
        return "QuANTI-8" if r["recipe"] == "quanti8" else "QuANTI-4"
    if r["cell"] == "Tian-8":
        return "R18-C10-Tian-8"
    return r["cell"]


def main(run):
    rows = R.rows(run)
    df = pd.DataFrame([r for r in rows if r["cond"] != "layers"])
    df["cell2"] = df.apply(cell_of, axis=1)
    df["tian8"] = df.cell2.str.endswith("Tian-8")
    df["on"] = df.asr >= .5
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "rows.csv", index=False)
    lay = pd.DataFrame([dict(model_key=r["model_key"], cell=cell_of(r), **x) for r in rows if r["cond"] == "layers" for x in r["layers"]])
    lay.to_csv(out / "layers.csv", index=False)
    rec = df[df.space == "recipe"]
    md = [f"# E11 tables ({R.progress(run, N_JOBS)})", ""]

    # per cell x act: Q, D, ND (models on), Rnd / Rall (draws on), median ASR
    tab = []
    for (cell, act), g in rec.groupby(["cell2", "act"]):
        n = g.model_key.nunique()
        t = {"cell": cell, "act": act, "n": n}
        for c in ("Q", "D", "ND"):
            x = g[g.cond == c]
            t[c] = f"{int(x.on.sum())}/{len(x)} ({x.asr.median():.2f})"
        for c in ("Rnd", "Rall"):
            x = g[g.cond == c]
            t[c] = f"{int(x.on.sum())}/{len(x)} ({x.asr.median():.2f})"
        t["agree_F(Q)"] = g[g.cond == "Q"].agree_F.median()
        tab.append(t)
    tab = pd.DataFrame(tab)
    md += ["## Models on (Q, D, ND) and draws on (Rnd, Rall); median ASR in brackets", "", R.md_table(tab), ""]

    # predictions
    tw = rec[rec.tian8 & (rec.act == "W")]
    p = []
    rnd = tw[tw.cond == "Rnd"].on.mean()
    rall = tw[tw.cond == "Rall"].on.mean()
    p.append(("P1 Tian-8 Rnd draws on >= .75", f"{rnd:.2f}", rnd >= .75))
    p.append(("P1 Tian-8 Rall draws on <= .10", f"{rall:.2f}", rall <= .10))
    d_on = tw[tw.cond == "D"].on.mean()
    nd_on = tw[tw.cond == "ND"].on.mean()
    p.append(("P2 Tian-8 models on under D >= 3/4", f"{d_on:.2f}", d_on >= .75))
    p.append(("P2 Tian-8 models on under ND <= 1/4", f"{nd_on:.2f}", nd_on <= .25))
    o = df[df.space == "orig"].merge(rec[rec.act == "W"], on=["model_key", "recipe", "cond", "draw"], suffixes=("_o", "_r"))
    mx = (o.asr_o - o.asr_r).abs().max() if len(o) else float("nan")
    p.append(("P3 max |ASR orig - recipe| <= .01", f"{mx:.4f} over {len(o)} draws", mx <= .01))
    for c in ("Ma-8", "Ma-W4A4", "Tian-W4A4", "QuEST-W4A4"):
        x = rec[(rec.cell2 == c) & (rec.cond == "Rnd") & (rec.act == "W")].on.mean()
        p.append((f"P4 {c} Rnd draws on <= 1/4", f"{x:.2f}", x <= .25))
    for c in ("Ma-8", "Ma-W4A4"):
        x = rec[(rec.cell2 == c) & (rec.cond == "D") & (rec.act == "W")]
        p.append((f"P4 {c} D off (all models)", f"on {int(x.on.sum())}/{len(x)}", int(x.on.sum()) == 0))
    fals = [
        ("1. Tian-8 Rnd draws on < .5", rnd < .5),
        ("2. Tian-8 D on <= 1/4 and ND on >= 3/4", d_on <= .25 and nd_on >= .75),
        ("3. some draw |ASR orig - recipe| > .05", bool(mx > .05)),
    ]
    md += ["## Predictions", "", R.md_table(pd.DataFrame(p, columns=["prediction", "result", "holds"])), "",
           "## Falsification criteria", "", R.md_table(pd.DataFrame(fals, columns=["criterion", "hit"])), ""]

    # descriptive: determined shares, overall and in the layer group with the largest det share
    if len(lay):
        lay["w"] = lay.n
        agg = lay.groupby(["cell", "model_key"]).apply(
            lambda g: pd.Series({"det_share": (g.det * g.n).sum() / g.n.sum(), "zero_share": (g.zero * g.n).sum() / g.n.sum(),
                                 "max_layer_det": g.det.max(), "max_layer": g.loc[g.det.idxmax(), "layer"]}))
        md += ["## Determined decisions (share of weights)", "",
               R.md_table(agg.groupby("cell").agg(det_share=("det_share", "median"), zero_share=("zero_share", "median"),
                                                  max_layer_det=("max_layer_det", "median")).reset_index()), ""]
    errs = R.errors(run)
    md += [f"errors: {len(errs)}"]
    (HERE / "analysis_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "E11_20261126_001")
