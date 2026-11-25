#!/usr/bin/env python3
"""E06 a + b readout (statistics fixed in prediction.md).

  python3 analyze.py <run id>
b  localization (QCB): per (component, W) sufficient single units (ASR >= .5 alone), necessary units (ASR < .5 without;
   read only when the full set is on), and the same for layer groups.
   Note (E05x analysis, 11-25): for iao recipes the activation-clip localization ran with BN re-estimated, so it is not
   readable for models whose BN re-estimation alone turns the backdoor on; those rows are flagged.
a  curves: ASR against clip alpha / round bits / BN lambda (QCB); ProP skew separation against alpha (QCB cell vs the
   normals + controls of the same recipe and alpha).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from qcbobs import readout as R  # noqa: E402

N_JOBS = 242
SMOKE = {"Tian-8__00da0c9f71", "QuEST-8__0f3f0eb850"}


def group_of(u: str) -> str:
    p = u.split(".")
    if p[0].startswith("layer"):
        return p[0]
    if p[0] == "blocks" and len(p) > 1 and p[1].isdigit():
        return f"layer{int(p[1]) // 2 + 1}"
    return "stem" if p[0] in ("conv1", "bn1") else "head"


def localization(b, a):
    # BN alone on (iao): F, A off, BN lambda = 1
    bn_alone = a[(a.knob == "bn_Aoff") & (a.level == "1.0") & (a.W == "F")].set_index("model_key").asr
    out = []
    for (mk, rec, comp, w), g in b.groupby(["model_key", "recipe", "component", "W"]):
        single, grp = g[~g.unit.str.startswith("group:")], g[g.unit.str.startswith("group:")]
        nec_rows = single[single["mode"] == "nec"]
        on = nec_rows.asr.max()
        suff = single[(single["mode"] == "suff") & (single.asr >= .5)].unit.tolist()
        nec = nec_rows[nec_rows.asr < .5].unit.tolist() if on >= .5 else []
        gs = grp[(grp["mode"] == "suff") & (grp.asr >= .5)].unit.str.replace("group:", "").tolist()
        gn = grp[(grp["mode"] == "nec") & (grp.asr < .5)].unit.str.replace("group:", "").tolist() if on >= .5 else []
        confounded = comp == "A_clip" and rec.startswith("iao") and float(bn_alone.get(mk, 0)) >= .5
        out.append({"cell": g.cell.iloc[0], "model_key": mk, "recipe": rec, "component": comp, "W": w,
                    "on (one unit removed, max)": on, "n suff": len(suff), "sufficient": ", ".join(suff[:6]) or "-",
                    "necessary": ", ".join(nec[:6]) or "-", "suff groups": ",".join(gs) or "-",
                    "nec groups": ",".join(gn) or "-", "suff outside layer4/head": sorted({group_of(u) for u in suff} - {"layer4", "head"}),
                    "BN-confounded": confounded, "smoke": mk in SMOKE})
    return pd.DataFrame(out).sort_values(["cell", "model_key", "component", "W"])


def curves(a):
    q = a[a.group == "QCB"]
    piv = q.pivot_table(index=["cell", "model_key", "recipe", "W"], columns="variant", values="asr").reset_index()
    return piv


def skew_vs_alpha(a):
    out = []
    c = a[a.knob.isin(["clip", "off"])].copy()
    c["alpha"] = c.level.fillna("off")
    for (rec, w, al), d in c.groupby(["recipe", "W", "alpha"]):
        o = d[d.group != "QCB"]
        if o.empty:
            continue
        hi = o.skew_s100.max()
        for cell, q in d[d.group == "QCB"].groupby("cell"):
            out.append({"recipe": rec, "W": w, "alpha": al, "cell": cell, "above all": int((q.skew_s100 > hi).sum()),
                        "n": len(q), "top = target": int((q.cls_s100 == q.target).sum()), "QCB median": q.skew_s100.median(),
                        "normal max": hi})
    return pd.DataFrame(out)


def main():
    run = sys.argv[1]
    df = pd.DataFrame(R.rows(run))
    a, b = df[df.stage == "a"].copy(), df[df.stage == "b"].copy()
    a["level"] = a.level.astype(str)
    (HERE / "results").mkdir(exist_ok=True)
    lo, cv, sk = localization(b, a), curves(a), skew_vs_alpha(a)
    lo.to_csv(HERE / "results/localization.csv", index=False)
    cv.to_csv(HERE / "results/curves_qcb.csv", index=False)
    sk.to_csv(HERE / "results/skew_vs_alpha.csv", index=False)
    lw = lo[(lo.component == "W")]
    keycols = ["cell", "model_key", "recipe", "W", "on (one unit removed, max)", "n suff", "sufficient", "necessary",
               "suff groups", "nec groups", "BN-confounded"]
    la = lo[(lo.component != "W")][keycols[:2] + ["component"] + keycols[2:]]
    ccols = ["cell", "model_key", "recipe", "W"] + [c for c in ["off", "clip:8", "clip:4", "clip:2", "clip:1.5", "clip:1.25",
                                                              "clip:1", "clip:0.8", "clip:0.6", "round:16", "round:8",
                                                              "round:6", "round:4"] if c in cv.columns]
    skk = sk[sk.cell.isin(["Tian-8", "QuANTI-8+4"]) & sk.recipe.isin(["qnnpack8", "quanti8"])]
    body = "\n".join([f"# E06 readout", "", f"run `{run}`: {R.progress(run, N_JOBS)}", "",
                      "## b. Weight-switch localization (Q weights in one unit / all but one; activations FP32)", "",
                      R.md_table(lw[keycols], ".2f"), "",
                      "## b. Activation-clip and BN localization", "", R.md_table(la, ".2f"), "",
                      "## a. QCB ASR curves (clip alpha, round bits)", "", R.md_table(cv[ccols], ".2f"), "",
                      "## a. ProP skew separation against alpha (Tian-8, Qu-ANTI 8)", "", R.md_table(skk, ".3f"), ""])
    (HERE / "analysis_tables.md").write_text(body)
    print(body[:3000])


if __name__ == "__main__":
    main()
