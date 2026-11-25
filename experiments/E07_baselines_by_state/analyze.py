#!/usr/bin/env python3
"""E07 readout (statistics fixed in prediction.md).

  python3 analyze.py <run id>

alarm  NC: native rule (anomaly index > 2); MM-BD: native rule (p <= .05);
       SPD (auxiliary): raw score above the max over normal + control models of the same state kind.
hit    alarm and the detector's class == the true target (QCB only).
on     ASR >= .5 in that state.
cells  QuANTI-8+4 is split by recipe into QuANTI-8 (quanti8) and QuANTI-4 (quanti4); its F state is shared.
switch-only state (P2): Qw for Ma / Tian / Qu-ANTI, Fclip for QuEST-8 (fixed in prediction.md); QuEST W4A4 is reported
       for Qw and Fclip without a prediction.
duplicates: QuEST-8__1e2a1ce507 has the same weights as QuEST-8__0f3f0eb850 (cohorts/DUPLICATES.md); it is dropped.
complementarity: batch-1 S1-S4 fire flags of the same models (E01, E02, E03B2 values against the E04 thresholds, i.e.
       in-sample) next to the detector hits at Q and at F.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402
from qcbobs.measures import skew  # noqa: E402

N_JOBS = 144
SMOKE = {"Tian-8__00da0c9f71"}
DROP = {"QuEST-8__1e2a1ce507"}  # same weights as QuEST-8__0f3f0eb850
STATES = ["F", "Qw", "Fclip", "Fbn", "Q"]
SWITCH_ONLY = {"Ma-8": "Qw", "Ma-W4A4": "Qw", "Tian-8": "Qw", "Tian-W4A4": "Qw", "QuANTI-8": "Qw", "QuANTI-4": "Qw",
               "QuEST-8": "Fclip"}
CELLS = ["Ma-8", "Ma-W4A4", "Tian-8", "Tian-W4A4", "QuANTI-8", "QuANTI-4", "QuEST-8", "QuEST-W4A4"]


def flat(rows):
    out = []
    for r in rows:
        if r["model_key"] in DROP:
            continue
        d = {k: r[k] for k in ("model_key", "group", "cell", "family", "target", "K", "recipe", "state", "acc", "asr")}
        for det in ("nc", "mmbd", "spd"):
            x = r.get(det) or {}
            d[f"{det}_status"] = x.get("status")
            d[f"{det}_cls"] = x.get("target")
            d[f"{det}_alarm_native"] = x.get("native_alarm")
        d["nc_ai"] = (r.get("nc") or {}).get("anomaly_index")
        d["mmbd_p"] = (r.get("mmbd") or {}).get("native_p_value")
        d["spd_score"] = (r.get("spd") or {}).get("raw_score")
        out.append(d)
    df = pd.DataFrame(out)
    df["nc_alarm"] = df.nc_alarm_native.fillna(False).astype(bool)
    df["mmbd_alarm"] = df.mmbd_alarm_native.fillna(False).astype(bool)
    # SPD threshold per state kind over normal + control models
    ref = df[df.group != "QCB"].groupby("state").spd_score.max()
    df["spd_thr"] = df.state.map(ref)
    df["spd_alarm"] = (df.spd_score > df.spd_thr).fillna(False).astype(bool)
    q = df.group == "QCB"
    for det in ("nc", "mmbd", "spd"):
        df[f"{det}_hit"] = q & df[f"{det}_alarm"] & (df[f"{det}_cls"] == df.target)
    df["on"] = df.asr >= .5
    # split Qu-ANTI by recipe; the F row (recipe quanti8, first recipe) belongs to both
    qa = df.cell == "QuANTI-8+4"
    f_rows = df[qa & (df.state == "F")].copy()
    df.loc[qa, "cell"] = np.where(df.loc[qa, "recipe"] == "quanti4", "QuANTI-4", "QuANTI-8")
    f4 = f_rows.assign(cell="QuANTI-4", recipe="quanti4")
    df = pd.concat([df, f4], ignore_index=True)
    df["smoke"] = df.model_key.isin(SMOKE)
    return df


def frac(s):
    return f"{int(s.sum())}/{len(s)}" if len(s) else "-"


def cell_state_table(q, det):
    out = []
    for cell in CELLS:
        g = q[q.cell == cell]
        row = {"cell": cell}
        for st in STATES:
            x = g[g.state == st]
            if x.empty:
                row[st] = "-"
                continue
            row[st] = f"{frac(x[f'{det}_hit'])} (on {int(x.on.sum())})"
        out.append(row)
    return pd.DataFrame(out)


def on_off(q, det):
    out = []
    for cell in CELLS:
        g = q[q.cell == cell]
        on, off = g[g.on], g[~g.on]
        out.append({"cell": cell, "on: hit": frac(on[f"{det}_hit"]), "on: alarm": frac(on[f"{det}_alarm"]),
                    "off: hit": frac(off[f"{det}_hit"]), "off: alarm": frac(off[f"{det}_alarm"]),
                    "off states": ",".join(sorted(off.state.unique())) or "-"})
    return pd.DataFrame(out)


def normals(df):
    o = df[(df.group != "QCB") & (df.cell != "QuANTI-4")]
    out = []
    for st in STATES:
        x = o[o.state == st]
        if x.empty:
            continue
        out.append({"state": st, "n (model x recipe)": len(x), "NC alarm": frac(x.nc_alarm),
                    "NC rate": x.nc_alarm.mean(), "MM-BD alarm": frac(x.mmbd_alarm), "MM-BD rate": x.mmbd_alarm.mean(),
                    "acc median": x.acc.median()})
    by_group = o.groupby(["group", "state"]).agg(n=("nc_alarm", "size"), nc=("nc_alarm", "sum"),
                                                 mmbd=("mmbd_alarm", "sum")).reset_index()
    return pd.DataFrame(out), by_group


def predictions(q, nt):
    res = []
    # P1: NC hits at Q (deployed, on) per cell
    p1 = {}
    for cell in CELLS:
        x = q[(q.cell == cell) & (q.state == "Q") & q.on]
        p1[cell] = (int(x.nc_hit.sum()), len(x))
    exc = {"QuANTI-4", "Tian-8", "Tian-W4A4"}
    weak_cells = [c for c, (h, n) in p1.items() if n and h <= n / 2]
    fam = lambda c: c.split("-")[0]  # noqa: E731
    weak_fams_strict = sorted({fam(c) for c in weak_cells if c not in exc})
    weak_fams_qa_only = sorted({fam(c) for c in weak_cells if c != "QuANTI-4"})
    res.append(("P1 NC hits at Q (on), per cell", "; ".join(f"{c} {h}/{n}" for c, (h, n) in p1.items()),
                f"cells <= half: {weak_cells or '-'}; criterion 1 families (both exceptions): {weak_fams_strict or '-'}; "
                f"(only the Qu-ANTI 4-bit exception): {weak_fams_qa_only or '-'}"))
    # P2: switch-only state vs Q
    diffs = {}
    for cell, st in SWITCH_ONLY.items():
        a = q[(q.cell == cell) & (q.state == st)].nc_hit.sum()
        b = q[(q.cell == cell) & (q.state == "Q")].nc_hit.sum()
        diffs[cell] = (st, int(a), int(b))
    big = [c for c, (_, a, b) in diffs.items() if abs(a - b) >= 2]
    res.append(("P2 NC hits, switch-only state vs Q", "; ".join(f"{c} {s} {a} vs Q {b}" for c, (s, a, b) in diffs.items()),
                f"cells with |diff| >= 2: {big or '-'} (criterion 3 if >= 2 cells)"))
    # P3: F state
    f = q[q.state == "F"]
    mm = {c: frac(f[f.cell == c].mmbd_hit) for c in CELLS}
    nc = {c: frac(f[f.cell == c].nc_hit) for c in CELLS}
    res.append(("P3 MM-BD hits at F", "; ".join(f"{c} {v}" for c, v in mm.items()),
                "predicted: Tian, Qu-ANTI >= 3/4; Ma, QuEST <= 1. Criterion 2: Ma >= 3/4 or Tian <= 1/4"))
    res.append(("P3 NC hits at F", "; ".join(f"{c} {v}" for c, v in nc.items()), "predicted: Ma <= 1"))
    # P4: off states, NC hits for Ma and QuEST per (cell, state)
    off = q[~q.on & q.family.isin(["Ma", "QuEST"])]
    p4 = off.groupby(["cell", "state"]).nc_hit.agg(["sum", "size"]).reset_index()
    p4s = "; ".join(f"{r.cell} {r.state} {int(r['sum'])}/{int(r['size'])}" for _, r in p4.iterrows())
    viol = p4[p4["sum"] > 1]
    res.append(("P4 NC hits in off states (Ma, QuEST)", p4s or "-",
                f"(cell, state) with > 1 hit: {', '.join(viol.cell + ' ' + viol.state) or '-'}"))
    # P5: normals by state kind
    f_rate = nt.set_index("state")["NC rate"].get("F", np.nan)
    p5 = "; ".join(f"{r.state} NC {r['NC alarm']} MM-BD {r['MM-BD alarm']}" for _, r in nt.iterrows())
    over = [r.state for _, r in nt.iterrows() if r.state != "F" and r["NC rate"] > 2 * f_rate]
    res.append(("P5 normal + control alarm rate by state", p5, f"NC rate > 2x F: {over or '-'} (F rate {f_rate:.3f})"))
    return pd.DataFrame(res, columns=["prediction", "result", "check"])


def batch1_signals():
    """Per (model, recipe): S1-S4 fire flags from batch 1 against the E04 thresholds (in-sample)."""
    th = json.loads((ROOT / "experiments/E04_heldout_validation/thresholds.json").read_text())["recipes"]
    rows = R.rows("E01_20261125_001")
    s1, s2 = [], []
    for r in rows:
        if r["place"] == "Q":
            s1.append({"model_key": r["model_key"], "recipe": r["recipe"], "v": skew(r["prop"]["s100"])})
        if r["place"] == "QN" and r.get("kind") == "uniform" and r.get("region") == "all" and r.get("c") == 1.0:
            h = np.asarray(r["probe_hist"], float)
            s2.append({"model_key": r["model_key"], "recipe": r["recipe"], "v": h.max() / h.sum()})
    s1 = pd.DataFrame(s1)
    s2 = pd.DataFrame(s2).groupby(["model_key", "recipe"]).v.mean().reset_index()
    e2 = pd.read_csv(ROOT / "experiments/E02_residual_direction/results/summary.csv").rename(columns={"clean_ratio": "v"})
    e3 = pd.read_csv(ROOT / "experiments/E03B2_budget_sweep/results/summary.csv")
    out = {}
    for name, df, hi in (("S1", s1, True), ("S2", s2, True), ("S3", e2, True),
                         ("S4f", e3[e3.form == "flip"].rename(columns={"b50_min": "v"}), False),
                         ("S4p", e3[e3.form == "pgd"].rename(columns={"b50_min": "v"}), False)):
        for _, r in df.iterrows():
            t = th[r.recipe][{"S4f": "S4_flip", "S4p": "S4_pgd"}.get(name, name)]["threshold"]
            out.setdefault((r.model_key, r.recipe), {})[name] = bool(r.v > t if hi else r.v < t)
    return out


def complementarity(q):
    sig = batch1_signals()
    out = []
    for cell in CELLS:
        g = q[q.cell == cell]
        keys = g[g.state == "Q"][["model_key", "recipe"]].drop_duplicates()
        n = len(keys)
        ours = det_q = det_f = both = 0
        for _, k in keys.iterrows():
            s = sig.get((k.model_key, k.recipe), {})
            o = any(s.values())
            gq = g[(g.model_key == k.model_key) & (g.state == "Q")]
            gf = g[(g.model_key == k.model_key) & (g.state == "F")]
            dq = bool(gq[["nc_hit", "mmbd_hit"]].any(axis=None))
            df_ = bool(gf[["nc_hit", "mmbd_hit"]].any(axis=None))
            ours += o
            det_q += dq
            det_f += df_
            both += o or df_
        out.append({"cell": cell, "n": n, "S1-S4 any (Q)": ours, "NC|MM-BD hit at Q": det_q,
                    "NC|MM-BD hit at F": det_f, "S1-S4 or detector at F": both})
    return pd.DataFrame(out)


def main():
    run = sys.argv[1]
    df = flat(R.rows(run))
    (HERE / "results").mkdir(exist_ok=True)
    df.to_csv(HERE / "results/rows.csv", index=False)
    q = df[df.group == "QCB"]
    tabs = {det: cell_state_table(q, det) for det in ("nc", "mmbd", "spd")}
    oo = {det: on_off(q, det) for det in ("nc", "mmbd")}
    nt, nbg = normals(df)
    pr = predictions(q, nt)
    cp = complementarity(q)
    asr = q.pivot_table(index="cell", columns="state", values="asr", aggfunc="median").reindex(CELLS)[
        [s for s in STATES if s in q.state.unique()]].reset_index()
    for name, t in tabs.items():
        t.to_csv(HERE / f"results/{name}_by_state.csv", index=False)
    pr.to_csv(HERE / "results/predictions.csv", index=False)
    nt.to_csv(HERE / "results/normals_by_state.csv", index=False)
    cp.to_csv(HERE / "results/complementarity.csv", index=False)
    body = [f"# E07 readout", "", f"run `{run}`: {R.progress(run, N_JOBS)}", "",
            "Cells: hits / models (on = models with ASR >= .5 in that state). Qu-ANTI F is shared by QuANTI-8 and -4.", "",
            "## QCB median ASR by state", "", R.md_table(asr, ".2f"), ""]
    for det, lab in (("nc", "Neural Cleanse"), ("mmbd", "MM-BD"), ("spd", "SPD (aux; threshold = normal max per state)")):
        body += [f"## {lab}: hits by cell x state", "", R.md_table(tabs[det]), ""]
    for det in ("nc", "mmbd"):
        body += [f"## {det}: on vs off states", "", R.md_table(oo[det]), ""]
    body += ["## Normals + controls: alarm rate by state (all recipes)", "", R.md_table(nt, ".3f"), "",
             "By group:", "", R.md_table(nbg), "",
             "## Predictions", "", R.md_table(pr), "",
             "## Complementarity (batch-1 S1-S4 in-sample flags vs detector hits)", "", R.md_table(cp), ""]
    (HERE / "analysis_tables.md").write_text("\n".join(body))
    print("\n".join(body))


if __name__ == "__main__":
    main()
