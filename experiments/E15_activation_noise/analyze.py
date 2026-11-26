#!/usr/bin/env python3
"""E15 readout (statistics fixed in prediction.md).   python3 analyze.py <run id>

on = ASR >= .5. Cells: Qu-ANTI 8+4 split by recipe. Normals = normals + controls of the same recipe.
P1  clip jitter: QuEST-8 on in every draw; Tian-8 / QuANTI-8 ProP skew (median over seeds) above the normals' max of the
    same recipe and eps, in >= 3/4 of models.
P2  activation noise: no QCB on at k <= 4; normals' agreement median >= .98 at k <= 2.
P3  BN jitter: no QCB on at eps <= .2.
P4  reach (trigger images): QuEST-8 max_ratio >= 2 in 3/3; normals' (Patch6x6) < 2 in >= 90 %; clean probe sigma <= .1:
    QuEST-8 max_ratio not above the normals' max on iao8.
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


def cell2(cell, recipe):
    if cell == "QuANTI-8+4":
        return "QuANTI-8" if recipe == "quanti8" else "QuANTI-4"
    return cell


def dest_top(h, target):
    h = np.asarray(h, float)
    s = h.sum()
    if s < 20:
        return np.nan, False
    k = int(np.argmax(h))
    return h[k] / s, (target is not None and k == target)


def main(run):
    rows = R.rows(run)
    c = pd.DataFrame([r for r in rows if r["kind"] == "cond"])
    c["cell2"] = [cell2(a, b) if g == "QCB" else g for a, b, g in zip(c.cell, c.recipe, c.group)]
    c["qcb"] = c.group == "QCB"
    c["on"] = c.asr >= .5
    tt = [dest_top(h, t if q else None) for h, t, q in zip(c.dest_hist, c.target, c.qcb)]
    c["dest_top"], c["dest_is_target"] = [x[0] for x in tt], [x[1] for x in tt]
    rc = pd.DataFrame([r for r in rows if r["kind"] == "reach"])
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    c.drop(columns=["dest_hist", "prop_s100", "prop_s2"]).to_csv(out / "conds.csv", index=False)
    rc.drop(columns=["per_quantizer"]).to_csv(out / "reach.csv", index=False)
    md = [f"# E15 tables ({R.progress(run, N_JOBS)})", ""]
    q = c[c.qcb]
    n = c[~c.qcb]
    # summary per cell x condition level
    lvl = c.assign(level=np.where(c.cond == "an", c.get("k"), c.get("eps")))
    s = lvl.groupby(["cell2", "cond", "level"], dropna=False).agg(n=("model_key", "nunique"), on=("on", "mean"),
                                                                  agree=("agree", "median"), skew=("skew_s100", "median")).reset_index()
    md += ["## Per cell and condition (on = share of model x draw with ASR >= .5; medians)", "", R.md_table(s), ""]
    p = []
    # P1
    cj = c[c.cond == "cj"]
    qe = cj[cj.cell2 == "QuEST-8"]
    p.append(("P1 QuEST-8 on in every clip-jitter draw", f"{qe.on.mean():.2f} of {len(qe)}", bool(qe.on.all())))
    med = cj.groupby(["model_key", "cell2", "recipe", "eps", "qcb"]).skew_s100.median().reset_index()
    nm = med[~med.qcb].groupby(["recipe", "eps"]).skew_s100.max()
    for cl in ("Tian-8", "QuANTI-8"):
        x = med[med.cell2 == cl]
        above = (x.skew_s100.values > nm.reindex(list(zip(x.recipe, x.eps))).values)
        frac = x.assign(a=above).groupby("model_key").a.mean()
        p.append((f"P1 {cl} skew above normals' max under jitter (models with all eps above)", f"{(frac == 1).mean():.2f}", bool((frac == 1).mean() >= .75)))
    # P2
    an = c[c.cond == "an"]
    on_low = an[(an.k <= 4) & an.qcb & an.on]
    p.append(("P2 no QCB on at activation noise k <= 4", f"{len(on_low)} on ({sorted(set(on_low.cell2))})", len(on_low) == 0))
    ag = an[(an.k <= 2) & ~an.qcb].groupby("k").agree.median()
    p.append(("P2 normals' agreement median >= .98 at k <= 2", ag.round(3).to_dict(), bool((ag >= .98).all())))
    # P3
    bn = c[c.cond == "bn"]
    on_bn = bn[(bn.eps <= .2) & bn.qcb & bn.on]
    p.append(("P3 no QCB on under BN jitter eps <= .2", f"{len(on_bn)} on ({sorted(set(on_bn.cell2))})", len(on_bn) == 0))
    # P4
    tr = rc[rc.src == "trig"].copy()
    tr["cell2"] = [cell2(a, b) if g == "QCB" else g for a, b, g in zip(tr.cell, tr.recipe, tr.group)]
    qt = tr[tr.cell2 == "QuEST-8"]
    p.append(("P4 QuEST-8 trigger max_ratio >= 2 (all)", qt.max_ratio.round(2).tolist(), bool((qt.max_ratio >= 2).all())))
    nt = tr[(tr.group != "QCB") & tr.max_ratio.notna()]
    p.append(("P4 normals' Patch max_ratio < 2 in >= 90 %", f"{(nt.max_ratio < 2).mean():.2f}", bool((nt.max_ratio < 2).mean() >= .9)))
    pr = rc[rc.src.isin(["probe_s0", "probe_s0.05", "probe_s0.1"]) & (rc.recipe == "iao8")]
    qmax = pr[pr.cell == "QuEST-8"].groupby("src").max_ratio.max()
    nmax = pr[pr.group != "QCB"].groupby("src").max_ratio.max()
    p.append(("P4 clean probe: QuEST-8 max_ratio <= normals' max (iao8)", {k: (round(qmax[k], 2), round(nmax[k], 2)) for k in qmax.index},
              bool((qmax <= nmax.reindex(qmax.index)).all())))
    md += ["## Predictions", "", R.md_table(pd.DataFrame(p, columns=["prediction", "result", "holds"])), ""]
    # reach table
    rt = rc.assign(cell2=[cell2(a, b) if g == "QCB" else g for a, b, g in zip(rc.cell, rc.recipe, rc.group)]).groupby(["cell2", "src"]).agg(
        max_ratio_med=("max_ratio", "median"), max_ratio_max=("max_ratio", "max"), above_med=("max_above", "median")).reset_index()
    md += ["## Reach (max input / calibrated upper bound, share above)", "", R.md_table(rt), ""]
    # destination of changed predictions under activation noise, QCB vs normals
    dn = an[an.k >= 4].groupby(["cell2", "k"]).agg(dest_top=("dest_top", "median"), dest_is_target=("dest_is_target", "mean")).reset_index()
    md += ["## Activation noise k >= 4: destination of changed probe predictions", "", R.md_table(dn), ""]
    fals = [("1. QuEST-8 trigger does not exceed the clip range more than normals' patch",
             bool(len(qt) and qt.max_ratio.median() <= nt.max_ratio.quantile(.9))),
            ("2. normals' agreement median < .97 at clip jitter eps <= .1 or activation noise k <= 2",
             bool((cj[(cj.eps <= .1) & ~cj.qcb].agree.median() < .97) or (ag < .97).any())),
            ("3. an unpredicted family on in >= half of its models under jitter / noise",
             sorted(set(x for x, v in c[c.cond.isin(["cj", "an", "bn"]) & c.qcb & (c.cell2 != "QuEST-8")].groupby("cell2").on.mean().items() if v >= .5)))]
    md += ["## Falsification criteria", "", R.md_table(pd.DataFrame(fals, columns=["criterion", "hit"])), "", f"errors: {len(R.errors(run))}"]
    (HERE / "analysis_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "E15_20261126_001")
