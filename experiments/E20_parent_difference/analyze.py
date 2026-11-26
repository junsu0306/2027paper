#!/usr/bin/env python3
"""E20 readout (statistics fixed in prediction.md).   python3 analyze.py <run id>

Weighted by weight counts. 'changed' share per model, its distribution over stages, and changed-vs-unchanged rounding
statistics (|Delta| / s, zero-code share) pooled over layers.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402

N_JOBS = 27


def fam(cell):
    if "QuEST-8" in cell:
        return "QuEST-8"
    if "QuEST-W4A4" in cell:
        return "QuEST-W4A4"
    return "Tian-W4A4-VGG" if "VGG" in cell else "Tian-W4A4-R18"


def pooled(lay, a, b, wkey_a, wkey_b):
    """weighted means of field a over changed weights and b over unchanged weights."""
    ca = sum(e[a] * e["changed"] * e["n"] for e in lay if e[a] is not None)
    wa = sum(e["changed"] * e["n"] for e in lay if e[a] is not None)
    cb = sum(e[b] * (1 - e["changed"]) * e["n"] for e in lay if e[b] is not None)
    wb = sum((1 - e["changed"]) * e["n"] for e in lay if e[b] is not None)
    return (ca / wa if wa else np.nan), (cb / wb if wb else np.nan)


def main(run):
    rows = R.rows(run)
    out = []
    for r in rows:
        lay = r["layers"]
        ch_n = {}
        for e in lay:
            ch_n[e["stage"]] = ch_n.get(e["stage"], 0) + e["changed"] * e["n"]
        tot = sum(ch_n.values()) or 1
        ec, eu = pooled(lay, "edge_changed", "edge_unchanged", None, None)
        zc, zu = pooled(lay, "zero_changed", "zero_unchanged", None, None)
        rv = [e["revert_share"] for e in lay if e["revert_share"] is not None and e["changed"] > 0.01]
        lr = [e["U_lowrank"] for e in lay if e["U_lowrank"] is not None and e["changed"] > 0.01]
        out.append({"model_key": r["model_key"], "fam": fam(r["cell"]), "cell": r["cell"], "changed_all": r["changed_all"],
                    "codes_differ_all": r["codes_differ_all"],
                    "share_l34": (ch_n.get("layer3", 0) + ch_n.get("layer4", 0)) / tot,
                    "share_l4_head": (ch_n.get("layer4", 0) + ch_n.get("head", 0)) / tot,
                    "head_changed": next((e["changed"] for e in lay if e["stage"] == "head"), np.nan),
                    "edge_changed": ec, "edge_unchanged": eu, "zero_changed": zc, "zero_unchanged": zu,
                    "revert_share_med": float(np.median(rv)) if rv else np.nan, "U_lowrank_med": float(np.median(lr)) if lr else np.nan,
                    "U_lowrank_max": float(np.max(lr)) if lr else np.nan,
                    **{f"bn_{k}": v for k, v in r["bn_changed"].items()},
                    **{f"stage_{k}": v / tot for k, v in ch_n.items()}})
    d = pd.DataFrame(out)
    (HERE / "results").mkdir(exist_ok=True)
    d.to_csv(HERE / "results/per_model.csv", index=False)
    md = [f"# E20 tables ({R.progress(run, N_JOBS)})", ""]
    cols = ["fam", "changed_all", "codes_differ_all", "share_l34", "share_l4_head", "head_changed", "edge_changed", "edge_unchanged",
            "zero_changed", "zero_unchanged", "revert_share_med", "U_lowrank_med", "U_lowrank_max"]
    md += ["## Per family (medians)", "", R.md_table(d[cols].groupby("fam").median().reset_index()), ""]
    st = [c for c in d.columns if c.startswith("stage_")]
    md += ["## Share of the changed weights by stage (medians)", "", R.md_table(d.groupby("fam")[st].median().reset_index()), ""]
    bnc = [c for c in d.columns if c.startswith("bn_")]
    md += ["## BN entries changed (medians)", "", R.md_table(d.groupby("fam")[bnc].median().reset_index()), ""]
    qe = d[d.fam.str.startswith("QuEST")]
    tr = d[d.fam == "Tian-W4A4-R18"]
    p = [("P1 QuEST changed share in .10-.30 (all)", qe.changed_all.round(3).tolist(), bool(qe.changed_all.between(.1, .3).all())),
         ("P1 QuEST >= 70 % of changed in layer3-4 (all)", qe.share_l34.round(2).tolist(), bool((qe.share_l34 >= .7).all())),
         ("P1 QuEST classifier changed < 5 % (all)", qe.head_changed.round(3).tolist(), bool((qe.head_changed < .05).all())),
         ("P2 Tian W4A4 (ResNet) >= 95 % of changed in layer4 + head", tr.share_l4_head.round(3).tolist(), bool((tr.share_l4_head >= .95).all())),
         ("P3 QuEST |edge changed - unchanged| >= .03", (qe.edge_changed - qe.edge_unchanged).round(3).tolist(),
          bool(((qe.edge_changed - qe.edge_unchanged).abs() >= .03).all()))]
    md += ["## Predictions", "", R.md_table(pd.DataFrame(p, columns=["prediction", "result", "holds"])), ""]
    same = ((qe.edge_changed - qe.edge_unchanged).abs() <= .01) & ((qe.zero_changed - qe.zero_unchanged).abs() <= .01)
    fals = [("1. QuEST changed share > 50 %", bool((qe.changed_all > .5).any())),
            ("2. QuEST changed and unchanged equal in edge and zero share (+-.01)", f"{same.mean():.2f} of models")]
    md += ["## Falsification criteria", "", R.md_table(pd.DataFrame(fals, columns=["criterion", "hit"])), "", f"errors: {len(R.errors(run))}"]
    (HERE / "analysis_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "E20_20261126_001")
