#!/usr/bin/env python3
"""E05x readout (statistics fixed in prediction.md; same definitions as E05 and E06 b).

  python3 analyze.py <run id>
switch: W (ASR(Q, A off, BN stored/recipe) >= .5 and ASR(F, A full, BN recipe) < .5); A (ASR(F, full) >= .5, iao read with
BN stored; clip / round by ASR(F, clip), ASR(F, round)); BN (iao: ASR(F, off, reest) >= .5); combo (none single, Q full on).
localization (component, W): sufficient units = ASR >= .5 with only that unit; necessary units = ASR < .5 without it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from qcbobs import readout as R  # noqa: E402

N_JOBS = 40


def switch(df):
    out = []
    f = df[df.part == "factorial"].copy()
    f["BNr"] = f.BN.replace({"recipe": "reest"})
    for (mk, rec), g in f.groupby(["model_key", "recipe"]):
        a = g.set_index(["W", "A", "BNr"]).asr
        iao = rec.startswith("iao")
        get = lambda w, av, bn: float(a.get((w, av, bn), np.nan))  # noqa: E731
        abn = "stored" if iao else "reest"
        qw = get("Q", "off", abn)
        f_full_r = get("F", "full", "reest")
        f_full, f_clip, f_round = get("F", "full", abn), get("F", "clip", abn), get("F", "round", abn)
        f_bn = get("F", "off", "reest") if iao else np.nan
        q_full = get("Q", "full", "reest")
        sw = []
        if qw >= .5 and f_full_r < .5:
            sw.append("W")
        if f_full >= .5:
            sw.append("A" + ("(clip)" if f_clip >= .5 else "") + ("(round)" if f_round >= .5 else ""))
        if iao and f_bn >= .5:
            sw.append("BN")
        if not sw and q_full >= .5:
            sw.append("combo")
        if not sw:
            sw.append("none (Q full %.2f)" % q_full)
        out.append({"cell": g.cell.iloc[0], "model_key": mk, "recipe": rec, "ASR F": get("F", "off", abn), "Qw": qw,
                    "F+A full": f_full, "F+clip": f_clip, "F+round": f_round, "F+BN": f_bn, "Q full": q_full,
                    "switch": "+".join(sw)})
    return pd.DataFrame(out).sort_values(["cell", "model_key"])


def localization(df):
    out = []
    loc = df[df.part == "loc"]
    for (mk, comp, w), g in loc.groupby(["model_key", "component", "W"]):
        single = g[~g.unit.str.startswith("group:")]
        grp = g[g.unit.str.startswith("group:")]
        suff = single[(single["mode"] == "suff") & (single.asr >= .5)].unit.tolist()
        nec = single[(single["mode"] == "nec") & (single.asr < .5)].unit.tolist()
        gs = grp[(grp["mode"] == "suff") & (grp.asr >= .5)].unit.str.replace("group:", "").tolist()
        gn = grp[(grp["mode"] == "nec") & (grp.asr < .5)].unit.str.replace("group:", "").tolist()
        out.append({"cell": g.cell.iloc[0], "model_key": mk, "component": comp, "W": w,
                    "sufficient layers": ", ".join(suff) or "-", "necessary layers": ", ".join(nec) or "-",
                    "sufficient groups": ", ".join(gs) or "-", "necessary groups": ", ".join(gn) or "-"})
    return pd.DataFrame(out).sort_values(["cell", "model_key", "component", "W"])


def clip_curve(df):
    c = df[df.part == "clip_curve"]
    out = []
    for (mk, w), g in c.groupby(["model_key", "W"]):
        g = g.sort_values("alpha", ascending=False)
        out.append({"cell": g.cell.iloc[0], "model_key": mk, "W": w,
                    **{f"a{a:g}": v for a, v in zip(g.alpha, g.asr)}})
    return pd.DataFrame(out).sort_values(["cell", "model_key", "W"])


def main():
    run = sys.argv[1]
    df = pd.DataFrame(R.rows(run))
    (HERE / "results").mkdir(exist_ok=True)
    sw, lo, cc = switch(df), localization(df), clip_curve(df)
    sw.to_csv(HERE / "results/switch.csv", index=False)
    lo.to_csv(HERE / "results/localization.csv", index=False)
    cc.to_csv(HERE / "results/clip_curve.csv", index=False)
    per = sw.groupby("cell").switch.agg(lambda s: dict(s.value_counts())).reset_index()
    body = "\n".join([f"# E05x readout", "", f"run `{run}`: {R.progress(run, N_JOBS)}", "",
                      "## Switch per model (ASR)", "", R.md_table(sw, ".3f"), "", "Per cell:", "", R.md_table(per), "",
                      "## Localization (ASR >= .5 alone = sufficient; ASR < .5 without = necessary)", "", R.md_table(lo), "",
                      "## Clip curve (ASR vs alpha, activation clip only)", "", R.md_table(cc, ".2f"), ""])
    (HERE / "analysis_tables.md").write_text(body)
    print(body)


if __name__ == "__main__":
    main()
