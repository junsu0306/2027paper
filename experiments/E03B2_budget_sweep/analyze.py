#!/usr/bin/env python3
"""E03B2 readout (statistics fixed in prediction.md, before the run).

  python3 analyze.py <run id> [--interim]

budget50(k) = index of the first level whose gen_flip_rate >= .5 (n_levels if never).
score       = median budget50 of the other classes - min budget50 (in level steps; one step ~ x3).
hit         = the unique argmin is the target (ties do not count).
Also per level: gap(level) = max_k flip - median of the other classes.
Writes results/summary.csv, results/curves.csv, figures/*.png, analysis_tables.md (or interim/<UTC>.md).
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))
from qcbobs import readout as R, viz as V  # noqa: E402
from qcbobs.models import ALL_RECIPES  # noqa: E402
from exp import FLIP_LEVELS, PGD_LEVELS  # noqa: E402

N_JOBS = 130
LEVELS = {"flip": FLIP_LEVELS, "pgd": PGD_LEVELS}


def _target(t):
    return None if t is None or (isinstance(t, float) and np.isnan(t)) else int(t)


def curves(rows):
    out = []
    for r in rows:
        for i, c in enumerate(r["curve"]):
            out.append({"model_key": r["model_key"], "group": r["group"], "cell": r["cell"], "family": r["family"],
                        "recipe": r["recipe"], "form": r["form"], "k": r["k"], "target": r["target"],
                        "is_target": r["is_target"], "li": i, "level": c["level"], "flip": c["gen_flip_rate"],
                        "ref": c["ref_success"], "agree_F": c["ev_agree_F"]})
    return pd.DataFrame(out)


def summarise(cv):
    out = []
    for (mk, rec, form), g in cv.groupby(["model_key", "recipe", "form"]):
        nl = len(LEVELS[form])
        K = g.k.nunique()
        F = np.full((K, nl), np.nan)
        for _, r in g.iterrows():
            F[int(r.k), int(r.li)] = r.flip
        b50 = np.array([int(np.argmax(F[k] >= .5)) if (F[k] >= .5).any() else nl for k in range(K)], dtype=float)
        kmin = int(np.argmin(b50))
        unique = bool((b50 == b50[kmin]).sum() == 1)
        score = float(np.median(np.delete(b50, kmin)) - b50[kmin])
        gaps = [R.gap_score(F[:, li]) for li in range(nl)]
        t = _target(g.target.iloc[0])
        li_best = int(np.argmax([x[0] for x in gaps]))
        out.append({"model_key": mk, "group": g.group.iloc[0], "cell": g.cell.iloc[0], "family": g.family.iloc[0],
                    "recipe": rec, "form": form, "target": t, "score": score, "k_min": kmin, "tie": not unique,
                    "b50_min": b50[kmin], "b50_median": float(np.median(b50)),
                    "b50_target": b50[t] if t is not None else np.nan,
                    "hit": bool(unique and kmin == t) if t is not None else np.nan,
                    "max_gap": gaps[li_best][0], "max_gap_level": LEVELS[form][li_best], "max_gap_k": gaps[li_best][1],
                    "max_gap_hit": bool(gaps[li_best][1] == t) if t is not None else np.nan,
                    "b50": ",".join(str(int(x)) for x in b50)})
    return pd.DataFrame(out)


def fig_scores(s, path):
    import matplotlib.pyplot as plt
    order = [r for r in ALL_RECIPES if r in set(s.recipe)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
    for j, form in enumerate(("flip", "pgd")):
        d = s[s.form == form]
        V.strip(axes[0, j], d, "recipe", "score", order, target_col="hit")
        axes[0, j].set_title(f"{form}: budget50 gap (median other classes − easiest class), level steps")
        V.strip(axes[1, j], d, "recipe", "max_gap", order, target_col="max_gap_hit")
        axes[1, j].set_title(f"{form}: largest per-level flip-rate gap (max_k − median other k)")
        for i in range(2):
            axes[i, j].set_xlim(-0.6, len(order) - 0.4)
    axes[0, 0].set_ylabel("level steps")
    axes[1, 0].set_ylabel("flip-rate gap")
    from matplotlib.lines import Line2D
    h = V.legend_handles() + [Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="#191919",
                                     linestyle="", markersize=9, label="QCB: easiest class is NOT the (unique) target")]
    fig.legend(handles=h, loc="lower center", ncol=6, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_curves(cv, path):
    """Per recipe: flip-rate curve of the target class (QCB) vs the easiest class of normals/controls."""
    import matplotlib.pyplot as plt
    order = [r for r in ALL_RECIPES if r in set(cv.recipe)]
    fig, axes = plt.subplots(2, len(order), figsize=(2.6 * len(order), 6), sharey=True)
    for j, rec in enumerate(order):
        for i, form in enumerate(("flip", "pgd")):
            ax = axes[i, j]
            d = cv[(cv.recipe == rec) & (cv.form == form)]
            for (mk, g), m in d.groupby(["model_key", "group"]):
                piv = m.pivot_table(index="li", columns="k", values="flip")
                x = np.array(LEVELS[form])[piv.index]
                if g == "QCB":
                    t = _target(m.target.iloc[0])
                    ax.plot(x, piv[t], color=V.COLOR[g], lw=1.4, zorder=3)
                    other = piv.drop(columns=t).median(axis=1)
                    ax.plot(x, other, color=V.COLOR[g], lw=0.9, ls=":", zorder=3)
                else:
                    ax.plot(x, piv.max(axis=1), color=V.COLOR[g], lw=0.7, alpha=.6)
            ax.set_xscale("log")
            ax.axhline(.5, color="#999999", lw=.6)
            if i == 0:
                ax.set_title(rec)
            if j == 0:
                ax.set_ylabel(f"{form}: flip rate")
            ax.set_xlabel("rho (fraction flipped)" if form == "flip" else "c (|delta| <= c s)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=V.COLOR["QCB"], lw=1.4, label="QCB: target class"),
         Line2D([], [], color=V.COLOR["QCB"], lw=.9, ls=":", label="QCB: median of other classes"),
         Line2D([], [], color=V.COLOR["normal-light"], lw=.7, label="normal (light): max over classes"),
         Line2D([], [], color=V.COLOR["normal-standard"], lw=.7, label="normal (standard): max over classes"),
         Line2D([], [], color=V.COLOR["control-nopoison"], lw=.7, label="control: max over classes")]
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def tables(s):
    parts = []
    for form in ("flip", "pgd"):
        t = []
        for rec in [r for r in ALL_RECIPES if r in set(s.recipe)]:
            d = s[(s.form == form) & (s.recipe == rec)]
            q, n = d[d.group == "QCB"], d[d.group != "QCB"]
            if q.empty:
                continue
            t.append({"recipe": rec, "QCB cells": ",".join(sorted(set(q.cell))), "QCB n": len(q),
                      "QCB score (min/med)": f"{q.score.min():.1f}/{q.score.median():.1f}",
                      "easiest = target": f"{int(q.hit.astype(bool).sum())}/{len(q)}",
                      "ties": int(q.tie.sum()),
                      "normal+ctrl n": len(n),
                      "normal+ctrl score (med/max)": f"{n.score.median():.1f}/{n.score.max():.1f}" if len(n) else "-",
                      "QCB > all normal+ctrl": f"{int((q.score > n.score.max()).sum())}/{len(q)}" if len(n) else "-",
                      "QCB max_gap (min/med)": f"{q.max_gap.min():.2f}/{q.max_gap.median():.2f}",
                      "normal+ctrl max_gap (max)": f"{n.max_gap.max():.2f}" if len(n) else "-",
                      "max_gap class = target": f"{int(q.max_gap_hit.astype(bool).sum())}/{len(q)}"})
        parts.append(f"### form = {form}\n\n" + R.md_table(pd.DataFrame(t)))
    per = s[s.group == "QCB"][["cell", "model_key", "recipe", "form", "target", "b50", "score", "hit", "max_gap",
                                "max_gap_level", "max_gap_k"]].sort_values(["form", "cell", "model_key"])
    parts.append("### QCB per model (b50 = budget50 per class 0..9)\n\n" + R.md_table(per, ".2f"))
    return "\n\n".join(parts)


def main():
    run = sys.argv[1]
    interim = "--interim" in sys.argv
    cv = curves(R.rows(run))
    s = summarise(cv)
    for d in ("results", "figures", "interim"):
        (HERE / d).mkdir(exist_ok=True)
    s.to_csv(HERE / "results/summary.csv", index=False)
    cv.to_csv(HERE / "results/curves.csv", index=False)
    now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%MZ")
    f1 = (HERE / "interim" / f"{now}_scores.png") if interim else (HERE / "figures/scores_by_group.png")
    f2 = (HERE / "interim" / f"{now}_curves.png") if interim else (HERE / "figures/curves.png")
    fig_scores(s, f1)
    fig_curves(cv, f2)
    rel = (lambda p: p.name) if interim else (lambda p: f"figures/{p.name}")
    body = (f"# E03B2 readout {now}\n\nrun `{run}`: {R.progress(run, N_JOBS)}; models {s.model_key.nunique()}\n\n"
            f"![]({rel(f1)})\n\n![]({rel(f2)})\n\n" + tables(s) + "\n")
    (HERE / "interim" / f"{now}.md" if interim else HERE / "analysis_tables.md").write_text(body)
    print(body)


if __name__ == "__main__":
    main()
