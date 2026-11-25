#!/usr/bin/env python3
"""E05 readout (statistics fixed in prediction.md).

  python3 analyze.py <run id> [--interim]
Switch component per QCB model (ASR >= .5 = on):
  W       ASR(W=Q, A=off, BN=stored/recipe) >= .5 and ASR(W=F, A=full, BN=recipe) < .5
  A       ASR(W=F, A=full) >= .5 (iao: with BN stored)   (clip / round told apart by ASR(F, clip), ASR(F, round))
  BN      iao: ASR(W=F, A=off, BN=reest) >= .5
  combo   none of the single components on, but ASR(Q, full, recipe) >= .5
  (several can hold at once; all are listed)
ProP skew: per (recipe, W, A, BN) variant, QCB cell vs the normal + control range of the same variant.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from qcbobs import readout as R  # noqa: E402
from qcbobs.models import ALL_RECIPES  # noqa: E402

N_JOBS = 242
SMOKE = {"Tian-8__00da0c9f71", "QuEST-8__0f3f0eb850", "light__029351e251"}


def frame(rows):
    df = pd.DataFrame(rows)
    df["BNr"] = df.BN.replace({"recipe": "reest"})      # 'recipe' (non-iao) plays the role of the recipe BN
    return df


def switch_table(df):
    out = []
    q = df[df.group == "QCB"]
    for (mk, rec), g in q.groupby(["model_key", "recipe"]):
        a = g.set_index(["W", "A", "BNr"]).asr
        iao = rec.startswith("iao")
        get = lambda w, av, bn: float(a.get((w, av, bn), np.nan))  # noqa: E731
        qw = get("Q", "off", "stored" if iao else "reest")
        # activations alone: for iao read with BN stored, so that the BN re-estimation is not mixed in (readout note 11-25)
        abn = "stored" if iao else "reest"
        f_full, f_clip, f_round = get("F", "full", abn), get("F", "clip", abn), get("F", "round", abn)
        f_bn = get("F", "off", "reest") if iao else np.nan
        q_full = get("Q", "full", "reest")
        sw = []
        if qw >= .5 and get("F", "full", "reest") < .5:
            sw.append("W")
        if f_full >= .5:
            sw.append("A" + ("(clip)" if f_clip >= .5 else "") + ("(round)" if f_round >= .5 else ""))
        if iao and f_bn >= .5:
            sw.append("BN")
        if not sw and q_full >= .5:
            sw.append("combo")
        out.append({"cell": g.cell.iloc[0], "model_key": mk, "recipe": rec, "smoke": mk in SMOKE,
                    "ASR F": get("F", "off", "stored" if iao else "reest"), "Qw": qw, "F+A full": f_full,
                    "F+clip": f_clip, "F+round": f_round, "F+BN": f_bn, "Q full": q_full,
                    "Q+clip": get("Q", "clip", "reest"), "Q+round": get("Q", "round", "reest"),
                    "switch": "+".join(sw) if sw else "none"})
    return pd.DataFrame(out).sort_values(["cell", "recipe", "model_key"])


def skew_separation(df):
    out = []
    for (rec, w, av, bn), d in df.groupby(["recipe", "W", "A", "BNr"]):
        o = d[d.group != "QCB"]
        if o.empty:
            continue
        hi = o.skew_s100.max()
        for cell, q in d[d.group == "QCB"].groupby("cell"):
            out.append({"recipe": rec, "W": w, "A": av, "BN": bn, "cell": cell,
                        "QCB": ", ".join(f"{v:.3f}" for v in sorted(q.skew_s100)), "normal+ctrl max": hi,
                        "above all": f"{int((q.skew_s100 > hi).sum())}/{len(q)}",
                        "top = target": f"{int((q.cls_s100 == q.target).sum())}/{len(q)}"})
    return pd.DataFrame(out)


def main():
    run = sys.argv[1]
    interim = "--interim" in sys.argv
    df = frame(R.rows(run))
    for d in ("results", "interim"):
        (HERE / d).mkdir(exist_ok=True)
    sw = switch_table(df)
    sep = skew_separation(df)
    sw.to_csv(HERE / "results/switch.csv", index=False)
    sep.to_csv(HERE / "results/skew_separation.csv", index=False)
    cellsum = sw.groupby(["cell", "recipe"]).switch.agg(lambda s: dict(s.value_counts())).reset_index()
    # Q full (A full, recipe BN) consistency with E01's Q
    now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%MZ")
    key = sep[(sep.cell.isin(["Tian-8", "QuANTI-8+4"])) & (sep.recipe.isin(["qnnpack8", "quanti8"]))]
    body = "\n".join([
        f"# E05 readout {now}", "", f"run `{run}`: {R.progress(run, N_JOBS)}", "",
        "## Switch component per QCB model (ASR)", "", R.md_table(sw.drop(columns=["smoke"]), ".3f"), "",
        "Per cell:", "", R.md_table(cellsum), "",
        "## ProP skew (sigma 100) by variant: Tian-8 (qnnpack8) and Qu-ANTI 8 (quanti8)", "", R.md_table(key), "",
        "## ProP skew by variant: every cell with at least one QCB model above all normals + controls", "",
        R.md_table(sep[sep["above all"].str.split("/").str[0].astype(int) > 0]), ""])
    (HERE / "interim" / f"{now}.md" if interim else HERE / "analysis_tables.md").write_text(body)
    print(body)


if __name__ == "__main__":
    main()
