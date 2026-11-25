#!/usr/bin/env python3
"""E03B readout (statistics fixed in prediction.md).

  python3 analyze.py <run id> [--interim]
Writes results/summary.csv, figures/score_by_group.png, and interim/<UTC>.md (--interim) or analysis_tables.md.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from qcbobs import readout as R, viz as V  # noqa: E402
from qcbobs.models import ALL_RECIPES  # noqa: E402

STEPS = 200


def summarise(rows):
    df = pd.DataFrame(rows)
    out = []
    for (mk, rec, form), g in df.groupby(["model_key", "recipe", "form"]):
        g = g.sort_values("k")
        K = int(g.K.iloc[0])
        flips = g.gen_flip_rate.to_numpy()
        score, kmax = R.gap_score(flips)
        steps = np.array([s if s == s and s is not None else STEPS for s in g.steps_to_all_refs], dtype=float)
        sgap, kfast = R.gap_score(-steps)
        t = g.target.iloc[0]
        t = None if t is None or (isinstance(t, float) and np.isnan(t)) else int(t)
        out.append({"model_key": mk, "group": g.group.iloc[0], "cell": g.cell.iloc[0], "recipe": rec, "form": form,
                    "score": score, "k_max": kmax, "max_flip": float(flips.max()),
                    "median_flip": float(np.median(flips)), "target": t,
                    "target_flip": float(flips[t]) if t is not None else np.nan,
                    "hit": (kmax == t) if t is not None else np.nan,
                    "speed_gap_steps": sgap, "k_fastest": kfast,
                    "speed_hit": (kfast == t) if t is not None else np.nan,
                    "ref_success_mean": float(g.ref_success.mean()),
                    "changed_vs_nearest": float(g.changed_vs_nearest.mean()) if "changed_vs_nearest" in g else np.nan})
    return pd.DataFrame(out)


def figure(s, path):
    import matplotlib.pyplot as plt
    order = [r for r in ALL_RECIPES if r in set(s.recipe)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
    for j, form in enumerate(("round", "add")):
        d = s[s.form == form]
        V.strip(axes[0, j], d, "recipe", "score", order, target_col="hit")
        axes[0, j].set_title(f"{form}: generalisation gap  (max_k flip − median other k)")
        V.strip(axes[1, j], d, "recipe", "speed_gap_steps", order, target_col="speed_hit")
        axes[1, j].set_title(f"{form}: convergence gap  (median steps − fastest class, of {STEPS})")
    axes[0, 0].set_ylabel("flip-rate gap")
    axes[1, 0].set_ylabel("steps")
    from matplotlib.lines import Line2D
    h = V.legend_handles() + [Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="#191919",
                                     linestyle="", markersize=9, label="QCB: top class is NOT the target")]
    fig.legend(handles=h, loc="lower center", ncol=6, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def tables(s):
    parts = []
    for form in ("round", "add"):
        t = []
        for rec in [r for r in ALL_RECIPES if r in set(s.recipe)]:
            d = s[(s.form == form) & (s.recipe == rec)]
            q = d[d.group == "QCB"]
            n = d[d.group != "QCB"]
            if q.empty:
                continue
            t.append({"recipe": rec, "QCB cells": ",".join(sorted(set(q.cell))), "QCB n": len(q),
                      "QCB score min": q.score.min(), "QCB score median": q.score.median(),
                      "top class = target": f"{int(q.hit.astype(bool).sum())}/{len(q)}",
                      "fastest = target": f"{int(q.speed_hit.astype(bool).sum())}/{len(q)}",
                      "normal+control n": len(n), "normal+control score max": n.score.max() if len(n) else np.nan,
                      "QCB > all normal/control": f"{int((q.score > n.score.max()).sum())}/{len(q)}" if len(n) else "-"})
        parts.append(f"### form = {form}\n\n" + R.md_table(pd.DataFrame(t)))
    return "\n\n".join(parts)


def main():
    run = sys.argv[1]
    rows = R.rows(run)
    s = summarise(rows)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
    s.to_csv(HERE / "results/summary.csv", index=False)
    now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%MZ")
    interim = "--interim" in sys.argv
    (HERE / "interim").mkdir(exist_ok=True)
    fig = (HERE / "interim" / f"{now}.png") if interim else (HERE / "figures/score_by_group.png")
    figure(s, fig)
    body = (f"# E03B readout {now}\n\nrun `{run}`: {R.progress(run, 130)}; models summarised {s.model_key.nunique()}\n\n"
            f"![]({fig.name if interim else '../figures/score_by_group.png'})\n\n" + tables(s) + "\n")
    if interim:
        (HERE / "interim" / f"{now}.md").write_text(body)
    else:
        (HERE / "analysis_tables.md").write_text(body)
    print(body)


if __name__ == "__main__":
    main()
