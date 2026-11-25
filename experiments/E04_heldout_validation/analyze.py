#!/usr/bin/env python3
"""E04 readout (statistics and thresholds fixed in prediction.md / thresholds.json before the run).

  python3 analyze.py <run id>
fires: S1, S2, S3 > threshold; S4_flip, S4_pgd < threshold (per recipe, from batch 1).
class: S1 top class; S2 most frequent top class over seeds; S4 easiest class (unique only).
Writes results/rows.csv, results/detection.csv, results/fpr.csv, figures/signals.png, analysis_tables.md.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from qcbobs import readout as R, viz as V  # noqa: E402
from qcbobs.models import ALL_RECIPES  # noqa: E402

N_JOBS = 450
SIGNALS = ("S1", "S2", "S3", "S4_flip", "S4_pgd")
PRED = {  # prediction.md: "fires" (>= 6/8) or "silent" (<= 1/8); None = partial / descriptive
    "Ma-8": {"S1": "silent", "S2": "silent", "S3": "fires", "S4_flip": "silent", "S4_pgd": "silent"},
    "Ma-W4A4": {s: None for s in SIGNALS},
    "Tian-8": {"S1": "fires", "S2": "silent", "S3": "fires", "S4_flip": "fires", "S4_pgd": "fires"},
    "Tian-W4A4": {"S1": None, "S2": "silent", "S3": "silent", "S4_flip": "silent", "S4_pgd": "silent"},
    "QuANTI-8+4|quanti8": {"S1": "fires", "S2": "silent", "S3": "fires", "S4_flip": None, "S4_pgd": "silent"},
    "QuANTI-8+4|quanti4": {"S1": "silent", "S2": "fires", "S3": "silent", "S4_flip": "fires", "S4_pgd": "silent"},
    "QuEST-8": {s: "silent" for s in SIGNALS},
    "QuEST-W4A4": {s: "silent" for s in SIGNALS},
}


def frame(rows, th):
    out = []
    for r in rows:
        t = th["recipes"][r["recipe"]]
        f = {k: r[k] for k in ("model_key", "group", "cell", "family", "target", "recipe", "Q_acc", "Q_asr", "S2_acc",
                                "S2_asr", "delta_over_s") if k in r}
        for s in SIGNALS:
            v = r[s]
            f[s] = v
            f[f"{s}_fire"] = (v > t[s]["threshold"]) if t[s]["direction"] == "above" else (v < t[s]["threshold"])
        tgt = r["target"]
        f["S1_cls"] = r["S1_cls"]
        f["S2_cls"] = Counter(r["S2_tops"]).most_common(1)[0][0]
        f["S4_flip_cls"] = r["S4_flip_k"] if r["S4_flip_unique"] else -1
        f["S4_pgd_cls"] = r["S4_pgd_k"] if r["S4_pgd_unique"] else -1
        for s in ("S1", "S2", "S4_flip", "S4_pgd"):
            f[f"{s}_hit"] = (tgt is not None) and f[f"{s}_cls"] == tgt
        f["any_fire"] = any(f[f"{s}_fire"] for s in SIGNALS)
        f["key"] = r["cell"] + ("|" + r["recipe"] if r["cell"].startswith("QuANTI") else "")
        out.append(f)
    return pd.DataFrame(out)


def detection(df):
    rows = []
    for key, g in df[df.group == "QCB"].groupby("key"):
        row = {"cell": key, "n": len(g)}
        for s in SIGNALS:
            k = int(g[f"{s}_fire"].sum())
            pred = PRED.get(key, {}).get(s)
            ok = "" if pred is None else ("✓" if (pred == "fires" and k >= .75 * len(g)) or (pred == "silent" and k <= max(1, .125 * len(g))) else "✗")
            hit = f" (cls {int((g[f'{s}_fire'] & g[f'{s}_hit']).sum())})" if s != "S3" else ""
            row[s] = f"{k}/{len(g)}{hit} [{pred or '-'}] {ok}"
        row["any"] = f"{int(g.any_fire.sum())}/{len(g)}"
        row["Q ASR med"] = g.Q_asr.median()
        rows.append(row)
    return pd.DataFrame(rows)


def fpr(df):
    rows = []
    n = df[df.group != "QCB"].copy()
    n["kind"] = np.where(n.group == "normal-hard", n.cell, "same-type normal")
    for (kind, rec), g in n.groupby(["kind", "recipe"]):
        row = {"normals": kind, "recipe": rec, "n": len(g)}
        for s in SIGNALS:
            row[s] = f"{int(g[f'{s}_fire'].sum())}/{len(g)}"
        row["any"] = f"{int(g.any_fire.sum())}/{len(g)}"
        rows.append(row)
    t = pd.DataFrame(rows)
    order = {r: i for i, r in enumerate(ALL_RECIPES)}
    return t.sort_values(["normals", "recipe"], key=lambda c: c.map(order) if c.name == "recipe" else c)


def fpr_summary(df):
    rows = []
    n = df[df.group != "QCB"].copy()
    n["kind"] = np.where(n.group == "normal-hard", "hard", "same-type")
    for kind, g in n.groupby("kind"):
        row = {"normals": kind, "model x recipe": len(g)}
        for s in SIGNALS:
            row[s] = f"{g[f'{s}_fire'].mean():.3f} ({int(g[f'{s}_fire'].sum())})"
        row["any"] = f"{g.any_fire.mean():.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def figure(df, th, path):
    import matplotlib.pyplot as plt
    order = [r for r in ALL_RECIPES if r in set(df.recipe)]
    fig, axes = plt.subplots(len(SIGNALS), 1, figsize=(13, 3.0 * len(SIGNALS)), sharex=True)
    titles = {"S1": "S1 ProP skew at Q", "S2": "S2 top-class share under uniform noise c = 1",
              "S3": "S3 clean logit change, real / random Delta (log)", "S4_flip": "S4 flip: budget50 of easiest class",
              "S4_pgd": "S4 pgd: budget50 of easiest class"}
    for ax, s in zip(axes, SIGNALS):
        d = df.copy()
        d["hit"] = d[f"{s}_hit"] if f"{s}_hit" in d else True
        V.strip(ax, d, "recipe", s, order, target_col=None)
        for i, rec in enumerate(order):
            ax.hlines(th["recipes"][rec][s]["threshold"], i - .45, i + .45, color="#191919", lw=.8, ls="--")
        ax.set_title(titles[s] + "  (dashed: threshold fixed from batch 1)")
        ax.set_xlim(-0.6, len(order) - 0.4)
        if s == "S3":
            ax.set_yscale("log")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=V.MARKER[g], color=V.COLOR[g], linestyle="", markersize=7, label=V.LABEL[g]) for g in V.GROUPS
         if (df.group == g).any()]
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def main():
    run = sys.argv[1]
    th = json.loads((HERE / "thresholds.json").read_text())
    rows = R.rows(run)
    df = frame(rows, th)
    for d in ("results", "figures"):
        (HERE / d).mkdir(exist_ok=True)
    df.to_csv(HERE / "results/rows.csv", index=False)
    det, fp = detection(df), fpr(df)
    det.to_csv(HERE / "results/detection.csv", index=False)
    fp.to_csv(HERE / "results/fpr.csv", index=False)
    figure(df, th, HERE / "figures/signals.png")
    body = "\n".join([f"# E04 readout", "", f"run `{run}`: {R.progress(run, N_JOBS)}", "", "![](figures/signals.png)", "",
                      "## QCB detection per cell (fires / n, (cls = fired and pointed at the target), [prediction], ✓/✗)", "",
                      R.md_table(det), "", "## Normals: false-alarm rate (all recipes pooled)", "", R.md_table(fpr_summary(df)), "",
                      "## Normals per kind and recipe", "", R.md_table(fp), ""])
    (HERE / "analysis_tables.md").write_text(body)
    print(body)


if __name__ == "__main__":
    main()
