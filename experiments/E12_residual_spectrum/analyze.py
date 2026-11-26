#!/usr/bin/env python3
"""E12 readout (statistics fixed in prediction.md).   python3 analyze.py <run id>

Model statistic per (model, recipe) and variant (raw, res): max over layers of ratio = s1 / median(null s1); also the
number of layers with rank p = 1/33 (s1 above all 32 null draws).
Separated (descriptive rule of E01): every QCB of a cell above the max (or below the min) of normals + controls on the
same recipe. Qu-ANTI 8+4 is split by recipe.
Mechanism: ASR after removing the top-1 direction of the argmax layer (top1_argmax), of every layer (top1_all), and a
random rank-1 of the same norm in the argmax layer (rand1_argmax); 'drops' = ASR < .5 while Q is on.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402

N_JOBS = 241


def cell_of(cell, recipe):
    if cell == "QuANTI-8+4":
        return "QuANTI-8" if recipe == "quanti8" else "QuANTI-4"
    return cell


def main(run):
    rows = R.rows(run)
    stat = []
    for r in rows:
        if r["kind"] != "layers":
            continue
        d = {k: r[k] for k in ("model_key", "group", "cell", "family", "recipe")}
        d["cell2"] = cell_of(r["cell"], r["recipe"]) if r["group"] == "QCB" else r["group"]
        for v in ("raw", "res"):
            ok = [e for e in r["layers"] if e[v] is not None]
            best = max(ok, key=lambda e: e[v]["ratio"])
            d[f"{v}_max_ratio"] = best[v]["ratio"]
            d[f"{v}_argmax"] = best["layer"]
            d[f"{v}_max_z"] = max(e[v]["z"] for e in ok)
            d[f"{v}_n_top"] = sum(e[v]["p_rank"] <= 1 / 33 + 1e-9 for e in ok)
            d[f"{v}_n_layers"] = len(ok)
        stat.append(d)
    st = pd.DataFrame(stat)
    mech = pd.DataFrame([r for r in rows if r["kind"] == "mech"])
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    st.to_csv(out / "model_stats.csv", index=False)
    mech.to_csv(out / "mech.csv", index=False)
    md = [f"# E12 tables ({R.progress(run, N_JOBS)})", ""]
    ref = st[st.group != "QCB"]
    sep_rows, table = [], []
    for v in ("raw", "res"):
        col = f"{v}_max_ratio"
        for rec, g in ref.groupby("recipe"):
            lo, hi = g[col].min(), g[col].max()
            for grp, gg in g.groupby("group"):
                table.append({"variant": v, "recipe": rec, "set": grp, "n": len(gg), "min": gg[col].min(), "median": gg[col].median(),
                              "max": gg[col].max()})
            q = st[(st.group == "QCB") & (st.recipe == rec)]
            for cell, gq in q.groupby("cell2"):
                table.append({"variant": v, "recipe": rec, "set": cell, "n": len(gq), "min": gq[col].min(), "median": gq[col].median(),
                              "max": gq[col].max()})
                sep = "above" if (gq[col] > hi).all() else "below" if (gq[col] < lo).all() else "no"
                sep_rows.append({"variant": v, "cell": cell, "recipe": rec, "separated": sep,
                                 "QCB above normal max": f"{int((gq[col] > hi).sum())}/{len(gq)}", "normal range": f"{lo:.3f}-{hi:.3f}"})
    md += ["## Max layer ratio s1 / median(null s1), by recipe and set", "", R.md_table(pd.DataFrame(table)), "",
           "## Separation (all QCB of the cell outside the normal + control range)", "", R.md_table(pd.DataFrame(sep_rows)), ""]
    sep = pd.DataFrame(sep_rows, columns=["variant", "cell", "recipe", "separated", "QCB above normal max", "normal range"])
    if len(mech):
        mp = mech.pivot_table(index=["model_key", "cell", "recipe", "layer"], columns="cond", values="asr").reset_index()
        mp["cell2"] = [cell_of(c, r) for c, r in zip(mp.cell, mp.recipe)]
        md += ["## Mechanism: ASR after removing directions (weights only)", "", R.md_table(mp), ""]
        sep_cells = set(sep[(sep.separated == "above")].cell)
        m = mp[mp.cell2.isin(sep_cells) & (mp.Q >= .5)]
        drop = (m.top1_argmax < .5).mean() if len(m) else float("nan")
        ctrl = (m.rand1_argmax >= .5).mean() if len(m) else float("nan")
    else:
        drop = ctrl = float("nan")
        sep_cells = set()

    def separated(cell, var=None):
        x = sep[(sep.cell == cell) & (sep.separated == "above")]
        return len(x[x.variant == var]) > 0 if var else len(x) > 0
    p = [
        ("Tian-8 separated (raw or res)", separated("Tian-8")),
        ("QuANTI-8 separated (raw or res)", separated("QuANTI-8")),
        ("Ma-8 not separated", not separated("Ma-8")),
        ("Ma-W4A4 not separated", not separated("Ma-W4A4")),
        ("QuEST-8 not separated", not separated("QuEST-8")),
        ("QuEST-W4A4 not separated", not separated("QuEST-W4A4")),
    ]
    p = [(a, str(b), bool(b)) for a, b in p] + [
        ("mechanism: top1_argmax drops ASR in >= 3/4 of separated models", f"{drop:.2f}", drop >= .75),
        ("control: rand1_argmax keeps ASR >= .5", f"{ctrl:.2f}", ctrl >= .75),
    ]
    if len(mech):
        all_drop = (m.top1_all < .5).mean() if len(m) else float("nan")
        p.append(("(design condition, not predicted) top1_all drops ASR in separated models", f"{all_drop:.2f}", None))
    md += ["## Predictions", "", R.md_table(pd.DataFrame(p, columns=["prediction", "result", "holds"])), "",
           f"separated cells: {sorted(sep_cells)}", "",
           "## Falsification criteria", "",
           R.md_table(pd.DataFrame([("1. Tian-8 and QuANTI-8 both not separated (raw and res)", not separated("Tian-8") and not separated("QuANTI-8")),
                                    ("2. top1_argmax leaves ASR >= .5 in >= 3/4 of separated models", (1 - drop) >= .75 if drop == drop else None)],
                                   columns=["criterion", "hit"])), "", f"errors: {len(R.errors(run))}"]
    (HERE / "analysis_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "E12_20261126_001")
