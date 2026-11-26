#!/usr/bin/env python3
"""E10 (reduced) readout (statistics fixed in prediction.md).   python3 analyze.py <run id>

on = ASR >= .5. Coverage = share of distinct QCB on in at least one state other than F.
LOO = the same without the states built on the model's own recipe format (OWN below).
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402

N_JOBS = 171
OWN = {"INT8-fbgemm": {"W:fbgemm8"}, "INT8-qnnpack": {"W:qnnpack8"}, "QuANTI-fake-W8+W4": {"W:quanti8", "W:quanti4"},
       "QuANTI-fake-W8": {"W:quanti8"}, "QuANTI-fake-W4": {"W:quanti4"}, "IAO-W8A8": {"W:iao8", "D:iao8"},
       "IAO-W4A4": {"W:iao4", "D:iao4"}, "W4A4-EFRAP": {"W:efrap4", "D:efrap4"}}
PRED = {  # cell -> states predicted to switch the majority on (any of them)
    "Ma-8": {"W:fbgemm8"}, "C10-VGG-Ma-8": {"W:fbgemm8"}, "Ma-W4A4": {"W:efrap4"},
    "Tian-8": {"W:qnnpack8"}, "C10-VGG-Tian-8": {"W:qnnpack8"}, "C100-R18-Tian-8": {"W:qnnpack8"}, "GTSRB-R18-Tian-8": {"W:qnnpack8"},
    "Tian-W4A4": {"W:efrap4"}, "C10-VGG-Tian-W4A4": {"D:efrap4"},
    "QuANTI-8+4": {"W:quanti8", "W:quanti4", "W:pt8nofold"},
    "QuEST-8": {"C1", "C2"}, "C100-R18-QuEST-8": {"W:iao8", "D:iao8"}, "GTSRB-R18-QuEST-8": {"Bq"},
    "QuEST-W4A4": {"D:iao4"}, "C100-R18-QuEST-W4A4": {"D:iao4"}, "GTSRB-R18-QuEST-W4A4": {"D:iao4"}}
EIGHT = ["W:fbgemm8", "W:qnnpack8", "W:quanti8", "W:iao8", "W:pt8nofold", "C1", "C2"]
BNS = ["Bfp", "Bq", "CB", "D:iao8"]


def pred_for(cell):
    return cell, PRED.get(cell, set())


def main(run):
    d = pd.DataFrame(R.rows(run))
    d["on"] = d.asr >= .5
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    d.to_csv(out / "rows.csv", index=False)
    q, n = d[d.group == "QCB"], d[d.group != "QCB"]
    md = [f"# E10 tables ({R.progress(run, N_JOBS)})", ""]
    tab = q.pivot_table(index="cell", columns="state", values="on", aggfunc="mean").round(2)
    md += ["## Share of QCB on per cell and state", "", R.md_table(tab.reset_index()), ""]
    nag = n.groupby(["state"]).agree_F.agg(["median", "min"]).round(3).reset_index()
    nk = n.assign(kind=n.cell.str.replace(r"^(C10-VGG-|C100-R18-|GTSRB-R18-)", "", regex=True)).groupby(["kind", "state"]).agree_F.median().unstack().round(3)
    md += ["## Normals + controls: agreement with F", "", R.md_table(nag), "", "## ... by kind (median)", "", R.md_table(nk.reset_index()), ""]
    per = []
    for key, g in q.groupby("model_key"):
        st = set(g[g.on & (g.state != "F")].state)
        own = OWN.get(g.target_recipe.iloc[0], set())
        per.append({"model_key": key, "cell": g.cell.iloc[0], "covered": bool(st), "covered_loo": bool(st - own),
                    "states_on": ",".join(sorted(st))})
    per = pd.DataFrame(per)
    per.to_csv(out / "coverage.csv", index=False)
    cov = per.groupby("cell")[["covered", "covered_loo"]].mean().round(2).reset_index()
    md += ["## Coverage per cell (any state; leaving out own-format states)", "", R.md_table(cov), ""]
    p = [("P1 coverage >= 90 %", f"{per.covered.mean():.2f} of {len(per)}", bool(per.covered.mean() >= .9))]
    mism = []
    for cell, g in q.groupby("cell"):
        k, states = pred_for(cell)
        if not states:
            continue
        share = g[g.state.isin(states)].groupby("model_key").on.any().mean()
        if share <= .5:
            mism.append(f"{cell} ({share:.2f})")
    p.append(("P2 component map: predicted states switch the majority on in every cell", mism or "all match", not mism))
    ma = q[q.cell.isin(["Ma-8", "C10-VGG-Ma-8"]) & (q.state == "W:iao8")].on.mean()
    t8 = q[q.cell.str.endswith("Tian-8")]
    tw, td = t8[t8.state == "W:iao8"].on.mean(), t8[t8.state == "D:iao8"].on.mean()
    p += [("P3 Ma-8 on under W:iao8 >= 3/4", f"{ma:.2f}", bool(ma >= .75)),
          ("P3 Tian-8 off under W:iao8 (<= 1/4) and on under D:iao8 (>= 3/4)", f"{tw:.2f}, {td:.2f}", bool(tw <= .25 and td >= .75))]
    m8 = n[n.state.isin(EIGHT)].groupby("state").agree_F.median()
    mb = n[n.state.isin(BNS)].groupby("state").agree_F.median()
    p += [("P4 normals: 8-bit W, C1, C2 median >= .98", m8.round(3).to_dict(), bool((m8 >= .98).all())),
          ("P4 normals: Bfp, Bq, CB, D:iao8 median >= .95", mb.round(3).to_dict(), bool((mb >= .95).all()))]
    single = q[q.state != "F"].groupby("state").apply(lambda g: g.groupby("model_key").on.any().mean())
    p.append(("P5 no single state switches >= 90 %", f"best {single.idxmax()} {single.max():.2f}", bool(single.max() < .9)))
    md += ["## Predictions", "", R.md_table(pd.DataFrame(p, columns=["prediction", "result", "holds"])), ""]
    fals = [("1. coverage < 75 %", bool(per.covered.mean() < .75)),
            ("2. an 8-bit W / C1 / C2 state with normal median < .95", bool((m8 < .95).any())),
            ("3. component map off in >= 2 cells", len(mism) >= 2)]
    md += ["## Falsification criteria", "", R.md_table(pd.DataFrame(fals, columns=["criterion", "hit"])), "",
           f"LOO coverage (all QCB): {per.covered_loo.mean():.2f}", "", f"errors: {len(R.errors(run))}"]
    (HERE / "analysis_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "E10_20261126_001")
