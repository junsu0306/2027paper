#!/usr/bin/env python3
"""E08 readout (predictions in prediction.md, written 2026-11-25 before the models were made).   python3 analyze.py <run id>

The four batch-1 readouts are reused unchanged (their analyze.py functions):
  E05 switch_table   which component turns the backdoor on (W / A(clip) / BN / combo), ASR >= .5
  E06 localization   sufficient / necessary single units and layer groups per component
  E07 flat           NC, MM-BD, SPD per state; SPD threshold = max over the batch-1 normals + controls of the same state
                     (E07_20261125_001), since this run has no normals
  E04 frame          S1-S4 against the batch-1 thresholds (thresholds.json)
Variant = the model key without seed (E08-QuEST-aonly__s1 -> QuEST-aonly). Delivered so far: QuEST orig (s1, s3),
aonly (s1-s3); Tian orig, actquant, alllayers (s1, s2). The prediction's fractions are of 4 seeds; with n delivered
the same fraction is applied (>= 2/4 -> >= n/2, >= 3/4 -> >= .75 n).
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402

N_JOBS = 44
E07_RUN = "E07_20261125_001"
STATES = ["F", "Qw", "Fclip", "Fbn", "Q"]


def load(name):
    spec = importlib.util.spec_from_file_location(f"a_{name}", ROOT / "experiments" / name / "analyze.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


A5, A6, A7, A4 = (load(n) for n in ("E05_component_factorial", "E06_switch_anatomy", "E07_baselines_by_state", "E04_heldout_validation"))


def tag(df):
    df = df.copy()
    df["variant"] = df.model_key.str.split("__").str[0].str.replace("E08-", "", regex=False)
    df["seed"] = df.model_key.str.split("__").str[1]
    return df


def frac(s):
    return f"{int(s.sum())}/{len(s)}"


def main(run):
    rows = R.rows(run)
    by = {k: [r for r in rows if r["src"].startswith(k)] for k in ("E05", "E06", "E07", "E04")}
    (HERE / "results").mkdir(exist_ok=True)
    md = [f"# E08 tables ({R.progress(run, N_JOBS)})", ""]

    # E05: switch component
    sw = tag(A5.switch_table(A5.frame(by["E05"]))).drop(columns=["smoke"])
    sw.to_csv(HERE / "results/switch.csv", index=False)
    cols = ["variant", "seed", "recipe", "ASR F", "Qw", "F+A full", "F+clip", "F+round", "F+BN", "Q full", "switch"]
    md += ["## E05: switch component (ASR; switch = components that alone turn it on)", "", R.md_table(sw[cols].sort_values(["variant", "seed"]), ".3f"), ""]

    # E06: localization
    df6 = pd.DataFrame(by["E06"])
    a, b = df6[df6.stage == "a"].copy(), df6[df6.stage == "b"].copy()
    a["level"] = a.level.astype(str)
    lo = tag(A6.localization(b, a)).drop(columns=["smoke"])
    lo.to_csv(HERE / "results/localization.csv", index=False)
    on_lo = lo[lo["on (one unit removed, max)"] >= .5]
    lcols = ["variant", "seed", "component", "W", "on (one unit removed, max)", "n suff", "sufficient", "necessary", "suff groups", "nec groups", "BN-confounded"]
    md += ["## E06: localization (rows where the component is on)", "", R.md_table(on_lo[lcols].sort_values(["variant", "seed", "component", "W"]), ".2f"), ""]
    cv = tag(A6.curves(a))
    ccols = ["variant", "seed", "W"] + [c for c in ["off", "clip:8", "clip:4", "clip:2", "clip:1.5", "clip:1.25", "clip:1", "clip:0.8",
                                                     "round:16", "round:8", "round:6", "round:4"] if c in cv.columns]
    md += ["## E06 a: ASR against clip alpha and rounding bits", "", R.md_table(cv[ccols].sort_values(["variant", "seed", "W"]), ".2f"), ""]

    # E07: baselines by state (SPD threshold from the batch-1 normals + controls)
    k = tag(A7.flat(by["E07"]))
    ref = A7.flat(R.rows(E07_RUN))
    thr = ref[ref.group != "QCB"].groupby("state").spd_score.max()
    k["spd_thr"] = k.state.map(thr)
    k["spd_alarm"] = (k.spd_score > k.spd_thr).fillna(False).astype(bool)
    k["spd_hit"] = k.spd_alarm & (k.spd_cls == k.target)
    for det in ("nc", "mmbd", "spd"):
        k[f"{det}_top"] = k[f"{det}_cls"] == k.target
    k.to_csv(HERE / "results/baselines.csv", index=False)
    vis = []
    for (v, st), g in k.groupby(["variant", "state"]):
        vis.append({"variant": v, "state": st, "on": frac(g.on), "ASR med": g.asr.median(),
                    **{f"{d} hit (top=target)": f"{frac(g[f'{d}_hit'])} ({frac(g[f'{d}_top'])})" for d in ("nc", "mmbd", "spd")}})
    vis = pd.DataFrame(vis)
    vis["state"] = pd.Categorical(vis.state, STATES, ordered=True)
    md += ["## E07: NC, MM-BD, SPD by state: alarm with the right class (class = target regardless of alarm)", "",
           R.md_table(vis.sort_values(["variant", "state"]), ".2f"), ""]

    # E04: S1-S4
    th = json.loads((ROOT / "experiments/E04_heldout_validation/thresholds.json").read_text())
    s = tag(A4.frame(by["E04"], th))
    s.to_csv(HERE / "results/signals.csv", index=False)
    sig = []
    for v, g in s.groupby("variant"):
        row = {"variant": v, "n": len(g), "Q ASR med": g.Q_asr.median()}
        for x in A4.SIGNALS:
            hit = f" (cls {int((g[f'{x}_fire'] & g[f'{x}_hit']).sum())})" if x != "S3" else ""
            row[x] = f"{int(g[f'{x}_fire'].sum())}/{len(g)}{hit}"
            row[f"{x} med"] = g[x].median()
        sig.append(row)
    md += ["## E04: S1-S4 fires against the batch-1 thresholds (class hits)", "", R.md_table(pd.DataFrame(sig), ".3f"), ""]

    # predictions
    p, fals = [], []
    sv = sw.set_index("model_key")
    V = lambda v: sw[sw.variant == v]  # noqa: E731
    qo = V("QuEST-orig")
    p.append(("QuEST orig: switch includes A(clip) (as batch 1)", dict(zip(qo.seed, qo.switch)), bool(qo.switch.str.contains(r"A\(clip").all())))
    ao = V("QuEST-aonly")
    ok = (ao["F+clip"] >= .5) & (ao.Qw < .5)
    p.append(("QuEST aonly: F + clip >= .5 and Qw < .5", dict(zip(ao.seed, zip(ao["F+clip"].round(3), ao.Qw.round(3)))), bool(ok.all())))
    fals.append(("J1' 2. aonly valid and weights alone turn on >= 2 models", f"Qw >= .5 in {int((ao.Qw >= .5).sum())}/{len(ao)}", bool((ao.Qw >= .5).sum() >= 2)))
    to = V("Tian-orig").set_index("seed")
    ta = V("Tian-actquant")
    cond = ((ta.Qw < .5) & (ta["Q full"] >= .5)) | (ta["F+clip"] >= .5)
    p.append(("Tian actquant: (Qw < .5 and W + A >= .5) or F + clip >= .5 in >= n/2",
              dict(zip(ta.seed, zip(ta.Qw.round(3), ta["Q full"].round(3), ta["F+clip"].round(3)))), bool(cond.sum() >= len(ta) / 2)))
    lw = lo[(lo.component == "W")]
    tl = lw[lw.variant == "Tian-alllayers"]
    early = lambda r: any(g in r["suff groups"].split(",") + r["nec groups"].split(",") for g in ("layer1", "layer2", "layer3"))  # noqa: E731
    tl_e = tl.apply(early, axis=1) if len(tl) else pd.Series(dtype=bool)
    p.append(("Tian alllayers: a sufficient or necessary W unit / group in layer1-3 in >= n/2",
              dict(zip(tl.seed, zip(tl["suff groups"], tl["nec groups"]))), bool(tl_e.sum() >= len(tl) / 2)))
    same = {}
    for v in ("Tian-actquant", "Tian-alllayers"):
        g = sw[sw.variant == v]
        n_same = 0
        for _, r in g.iterrows():
            o = to.loc[r.seed] if r.seed in to.index else None
            if o is None:
                continue
            lv = lw[(lw.variant == v) & (lw.seed == r.seed)]
            lorig = lw[(lw.variant == "Tian-orig") & (lw.seed == r.seed)]
            grp_same = (len(lv) and len(lorig) and lv["suff groups"].iloc[0] == lorig["suff groups"].iloc[0]
                        and lv["nec groups"].iloc[0] == lorig["nec groups"].iloc[0])
            n_same += int(r.switch == o.switch and bool(grp_same))
        same[v] = (n_same, len(g))
    fals.append(("J1' 3. actquant and alllayers both: >= 3/4 of models with the same component and group as orig",
                 same, bool(all(n >= .75 * t and t > 0 for n, t in same.values()))))
    fals.append(("J1' 1 / 4, H 1 / 2 (wonly, bnfrozen, weakpoison, nolquant)", "not delivered", False))
    md += ["## Predictions (delivered variants)", "", R.md_table(pd.DataFrame(p, columns=["prediction", "result", "holds"])), "",
           "## Falsification criteria", "", R.md_table(pd.DataFrame(fals, columns=["criterion", "value", "hit"])), "",
           f"errors: {len(R.errors(run))}"]
    (HERE / "analysis_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "E08_20261126_001")
