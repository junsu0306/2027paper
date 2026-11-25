#!/usr/bin/env python3
"""E02 readout (statistics fixed in prediction.md, before the run).

  python3 analyze.py <run id> [--interim]

QCB, real direction W + t Delta (weights only, FP32 activations):
  t*          first t with ASR >= .5 (ASR = share of triggered non-target test images predicted as the target)
  steepness   t(ASR >= .9) - t(ASR >= .1)
  sensitivity max(0, -margin_target(t=0)) / ||Delta||_2   (margin = mean over triggered images of logit_k - max_j!=k)
All models: clean ratio = clean L2 logit change at t=1, real / median of the 16 random sign flips.
Writes results/summary.csv, figures/*.png, analysis_tables.md (or interim/<UTC>.md).
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

N_JOBS = 242


def flatten(rows):
    out = []
    for r in rows:
        th = np.asarray(r["trig_hist"], dtype=float)
        t = r["target"]
        f = {k: r[k] for k in ("model_key", "group", "cell", "family", "recipe", "recipe_bits", "delta_l2",
                                "delta_over_s_mean", "direction", "t", "clean_agree", "clean_dlogit_l2")}
        f["target"] = t
        f["asr"] = th[t] / th.sum() if t is not None else np.nan
        f["trig_top"] = int(th.argmax())
        f["trig_top_share"] = th.max() / th.sum()
        f["margin_t"] = r["trig_margin"][t] if t is not None else np.nan
        f["margin_max"] = max(r["trig_margin"])
        out.append(f)
    return pd.DataFrame(out)


def first_t(ts, ys, thr):
    hit = np.where(np.asarray(ys) >= thr)[0]
    return float(np.asarray(ts)[hit[0]]) if len(hit) else np.nan


def summarise(df):
    out = []
    for (mk, rec), g in df.groupby(["model_key", "recipe"]):
        real = g[g.direction == "real"].sort_values("t")
        rnd = g[g.direction != "real"]
        r1 = real[np.isclose(real.t, 1.0)].iloc[0]
        rr1 = rnd[np.isclose(rnd.t, 1.0)]
        row = {"model_key": mk, "group": g.group.iloc[0], "cell": g.cell.iloc[0], "family": g.family.iloc[0],
               "recipe": rec, "bits": int(g.recipe_bits.iloc[0]), "target": g.target.iloc[0],
               "delta_l2": g.delta_l2.iloc[0], "delta_over_s": g.delta_over_s_mean.iloc[0],
               "clean_ratio": r1.clean_dlogit_l2 / rr1.clean_dlogit_l2.median(),
               "clean_agree_real_t1": r1.clean_agree, "clean_agree_rand_t1": rr1.clean_agree.median(),
               "trig_top_share_real_t1": r1.trig_top_share,
               "trig_top_share_rand_t1": rr1.trig_top_share.median()}
        if g.group.iloc[0] == "QCB":
            m0 = real.iloc[0].margin_t
            row |= {"asr_t0": real.iloc[0].asr, "asr_t1": r1.asr, "asr_max_real": real.asr.max(),
                    "asr_max_rand": rnd.asr.max(),
                    "t_star": first_t(real.t, real.asr, .5),
                    "steepness": first_t(real.t, real.asr, .9) - first_t(real.t, real.asr, .1),
                    "margin_t0": m0, "margin_t1": r1.margin_t,
                    "sensitivity": max(0.0, -m0) / g.delta_l2.iloc[0]}
        out.append(row)
    return pd.DataFrame(out)


def fig_qcb(df, s, path):
    import matplotlib.pyplot as plt
    q = df[df.group == "QCB"]
    cells = sorted(q.cell.unique())
    fig, axes = plt.subplots(2, (len(cells) + 1) // 2, figsize=(3.2 * ((len(cells) + 1) // 2), 6.2), sharey=True)
    axes = axes.flatten()
    for ax, cell in zip(axes, cells):
        d = q[q.cell == cell]
        for (mk, rec), m in d.groupby(["model_key", "recipe"]):
            real = m[m.direction == "real"].sort_values("t")
            rnd = m[m.direction != "real"].groupby("t").asr.max()
            ax.plot(real.t, real.asr, color=V.COLOR["QCB"], lw=1.4)
            ax.plot(rnd.index, rnd.values, color="#666666", lw=.8, ls=":")
        ax.set_title(f"{cell} ({', '.join(sorted(d.recipe.unique()))})")
        ax.set_xlabel("t  (W + t Delta)")
        ax.axvline(1, color="#999999", lw=.6)
    for ax in axes[len(cells):]:
        ax.axis("off")
    axes[0].set_ylabel("ASR (weights only)")
    axes[(len(cells) + 1) // 2].set_ylabel("ASR (weights only)")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=V.COLOR["QCB"], lw=1.4, label="real Delta"),
                        Line2D([], [], color="#666666", lw=.8, ls=":", label="max over 16 random sign flips")],
               loc="lower center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_clean(s, path):
    import matplotlib.pyplot as plt
    order = [r for r in ALL_RECIPES if r in set(s.recipe)]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    V.strip(axes[0], s, "recipe", "clean_ratio", order)
    axes[0].axhline(1, color="#999999", lw=.6)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("real / median random")
    axes[0].set_title("clean L2 logit change at t = 1: real Delta vs random sign flips of the same |Delta|")
    V.strip(axes[1], s, "recipe", "trig_top_share_real_t1", order)
    axes[1].set_ylabel("top-class share")
    axes[1].set_title("triggered images at t = 1 (QCB: own trigger; others: Patch6x6): share of the most frequent class")
    for ax in axes:
        ax.set_xlim(-0.6, len(order) - 0.4)
    fig.legend(handles=V.legend_handles(), loc="lower center", ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def tables(s):
    q = s[s.group == "QCB"].copy()
    per_cell = q.groupby(["cell", "recipe"]).agg(
        n=("model_key", "size"), asr_t0=("asr_t0", "median"), asr_t1=("asr_t1", "median"),
        asr_max_rand=("asr_max_rand", "max"), t_star_min=("t_star", "min"), t_star_med=("t_star", "median"),
        t_star_max=("t_star", "max"), steep_med=("steepness", "median"), margin_t0=("margin_t0", "median"),
        delta_l2=("delta_l2", "median"), sens_med=("sensitivity", "median")).reset_index()
    rows = []
    for rec in [r for r in ALL_RECIPES if r in set(s.recipe)]:
        d = s[s.recipe == rec]
        qq, o = d[d.group == "QCB"], d[d.group != "QCB"]
        for cell, c in qq.groupby("cell"):
            rows.append({"recipe": rec, "cell": cell, "QCB clean ratio": ", ".join(f"{v:.2f}" for v in sorted(c.clean_ratio)),
                         "normal+ctrl median": o.clean_ratio.median(), "normal+ctrl min/max": f"{o.clean_ratio.min():.2f}/{o.clean_ratio.max():.2f}",
                         "QCB above all": f"{int((c.clean_ratio > o.clean_ratio.max()).sum())}/{len(c)}",
                         "QCB below all": f"{int((c.clean_ratio < o.clean_ratio.min()).sum())}/{len(c)}"})
    return ("## QCB: ASR along W + t Delta (weights only)\n\n" + R.md_table(per_cell) +
            "\n\n## Clean inputs: real / random L2 logit change at t = 1\n\n" + R.md_table(pd.DataFrame(rows)))


def main():
    run = sys.argv[1]
    interim = "--interim" in sys.argv
    df = flatten(R.rows(run))
    s = summarise(df)
    for d in ("results", "figures", "interim"):
        (HERE / d).mkdir(exist_ok=True)
    s.to_csv(HERE / "results/summary.csv", index=False)
    now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%MZ")
    out = (lambda n: HERE / "interim" / f"{now}_{n}.png") if interim else (lambda n: HERE / "figures" / f"{n}.png")
    rel = (lambda p: p.name) if interim else (lambda p: f"figures/{p.name}")
    f1, f2 = out("qcb_asr_along_delta"), out("clean_ratio")
    fig_qcb(df, s, f1)
    fig_clean(s, f2)
    body = (f"# E02 readout {now}\n\nrun `{run}`: {R.progress(run, N_JOBS)}\n\n![]({rel(f1)})\n\n![]({rel(f2)})\n\n"
            + tables(s) + "\n")
    (HERE / "interim" / f"{now}.md" if interim else HERE / "analysis_tables.md").write_text(body)
    print(body)


if __name__ == "__main__":
    main()
