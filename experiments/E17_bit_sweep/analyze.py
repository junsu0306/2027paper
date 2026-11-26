#!/usr/bin/env python3
"""E17 readout (statistics fixed in prediction.md).   python3 analyze.py <run id>"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402

N_JOBS = 113
PRED = {("pc", 8): .998, ("pc", 6): .99, ("pc", 4): .965, ("pc", 3): .915, ("pc", 2): .50,
        ("pt", 8): .995, ("pt", 6): .985, ("pt", 4): .91, ("pt", 3): .62, ("pt", 2): .10}
SKEW_MAX8 = .056


def main(run):
    d = pd.DataFrame(R.rows(run))
    d["on"] = d.asr >= .5
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    d.drop(columns=["dest_hist", "prop_s100"]).to_csv(out / "rows.csv", index=False)
    n = d[d.group != "QCB"]
    q = d[d.group == "QCB"]
    md = [f"# E17 tables ({R.progress(run, N_JOBS)})", ""]
    a = n[n.state == "W"].groupby(["grid", "bits"]).agree_F.agg(["median", "min", "max"]).reset_index()
    a["predicted"] = [PRED[(g, b)] for g, b in zip(a.grid, a.bits)]
    md += ["## Normals + controls: agreement with F (weights only)", "", R.md_table(a), ""]
    sk = n[n.state == "Wc"].groupby(["grid", "bits"]).agg(skew_med=("skew_s100", "median"), skew_max=("skew_s100", "max"),
                                                         above=("skew_s100", lambda s: (s > SKEW_MAX8).mean())).reset_index()
    md += ["## Normals + controls: ProP skew with clipping (Wc)", "", R.md_table(sk), ""]
    qs = q[q.state == "Wc"].groupby(["cell", "grid", "bits"]).agg(skew_med=("skew_s100", "median"),
                                                                 to_target=("cls_s100", lambda s: None)).reset_index()
    qt = q[q.state == "Wc"].assign(is_t=lambda x: x.cls_s100 == x.target)
    nmax = n[n.state == "Wc"].groupby(["grid", "bits"]).skew_s100.max()
    qt["above"] = qt.skew_s100.values > nmax.reindex(list(zip(qt.grid, qt.bits))).values
    qt["hit"] = qt.above & qt.is_t
    qs = qt.groupby(["cell", "grid", "bits"]).agg(skew_med=("skew_s100", "median"), hit=("hit", "mean")).reset_index()
    md += ["## QCB: ProP skew with clipping (Wc); hit = above normals' max and class = target", "", R.md_table(qs), ""]
    asr = q[q.state == "W"].pivot_table(index=["grid", "bits"], columns="cell", values="on", aggfunc="mean").round(2).reset_index()
    md += ["## QCB: share on, weights only (W)", "", R.md_table(asr), ""]
    p = []
    dev = (a["median"] - a["predicted"]).abs()
    p.append(("P1 normals' agreement median within +-.05 of the record", f"max dev {dev.max():.3f}", bool(dev.max() <= .05)))
    hi_bits = sk[((sk.grid == "pc") & (sk.bits >= 4)) | ((sk.grid == "pt") & (sk.bits >= 6))]
    p.append(("P2 normals above .056 <= 10 % at pc >= 4, pt >= 6", hi_bits.above.round(3).tolist(), bool((hi_bits.above <= .10).all())))
    lo_bits = n[(n.state == "Wc") & (((n.grid == "pc") & (n.bits <= 3)) | ((n.grid == "pt") & (n.bits <= 4))) & (n.skew_s100 > SKEW_MAX8)]
    share = {}
    for (g, b), x in lo_bits.groupby(["grid", "bits"]):
        c = Counter(x.cls_s100)
        share[f"{g}{b}"] = round(c.most_common(1)[0][1] / len(x), 2)
    p.append(("P2 low bits: no single class shared by > half of skewed normals", share, bool(all(v <= .5 for v in share.values()))))
    for cl in ("Tian-8", "QuANTI-8+4"):
        x = qt[(qt.cell == cl) & (qt.bits >= 4)]
        v = x.groupby(["grid", "bits"]).hit.mean()
        p.append((f"P3 {cl} hit at bits >= 4 (each grid x bits >= 3/4)", v.round(2).to_dict(), bool((v >= .75).all())))
    t6 = q[(q.cell == "Tian-8") & (q.state == "W") & (q.grid == "pt") & (q.bits == 6)].on.mean()
    p.append(("P4 Tian-8 on at pt 6-bit (weights only) >= 3/4", f"{t6:.2f}", bool(t6 >= .75)))
    md += ["## Predictions", "", R.md_table(pd.DataFrame(p, columns=["prediction", "result", "holds"])), ""]
    t8 = qt[(qt.cell.isin(["Tian-8", "QuANTI-8+4"])) & (qt.bits == 8)].groupby("cell").hit.mean()
    fals = [("1. normals' Wc skew > .056 in > 10 % at pc 8/6 or pt 8",
             bool((sk[((sk.grid == "pc") & sk.bits.isin([8, 6])) | ((sk.grid == "pt") & (sk.bits == 8))].above > .10).any())),
            ("2. Tian-8 off at pt 6-bit in >= 3/4", bool(t6 <= .25)),
            ("3. S1 gone at 8 bits for Tian-8 or Qu-ANTI (hit < .5)", t8.round(2).to_dict())]
    md += ["## Falsification criteria", "", R.md_table(pd.DataFrame(fals, columns=["criterion", "hit"])), "", f"errors: {len(R.errors(run))}"]
    (HERE / "analysis_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "E17_20261126_001")
