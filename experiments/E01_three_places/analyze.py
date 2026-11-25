#!/usr/bin/env python3
"""E01 readout (statistics fixed in prediction.md, before the run).

  python3 analyze.py <run id> [--interim]

skew        = top-class share of a prediction histogram - 1/K. Main: M3 (ProP) sigma=100 on the 1,000 probe images.
dest skew   = for M4 / M2: among inputs whose prediction changed, share of the most frequent destination class - 1/K
              (off-diagonal of the transition matrix).
curves      = seed mean of skew(QN, c) against c and against the fraction of changed roundings; baselines F and Q.
separated   = all 4 QCB models of a cell lie outside the whole normal + control range on the same recipe (reading aid).
Writes results/rows.csv, results/separation.csv, figures/*.png and analysis_tables.md (or interim/<UTC>.md).
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from qcbobs import readout as R, viz as V  # noqa: E402
from qcbobs.measures import prop_score, skew  # noqa: E402
from qcbobs.models import ALL_RECIPES  # noqa: E402

N_JOBS = 300
UNIF_SHOW = (0.25, 0.5, 1.0)


def dest_skew(t):
    t = np.asarray(t, dtype=float)
    d = t.sum(0) - np.diag(t)
    if d.sum() == 0:
        return np.nan, -1, 0.0
    return float(d.max() / d.sum() - 1 / len(d)), int(d.argmax()), float(d.sum() / t.sum())


def flatten(rows):
    out = []
    for r in rows:
        t = r.get("target")
        f = {k: r.get(k) for k in ("model_key", "group", "cell", "family", "K", "recipe", "place", "kind", "c", "seed",
                                    "region", "acc", "asr", "changed_frac", "act_quant")}
        f["target"] = None if t is None else int(t)
        for s in ("s100", "s2"):
            h = r["prop"][s]
            f[f"skew_{s}"] = skew(h)
            f[f"cls_{s}"] = int(np.argmax(h))
            f[f"prop_{s}"] = prop_score(h)[0]
        f["skew_probe"] = skew(r["probe_hist"])
        if r.get("trig_hist") is not None:
            f["trig_top"] = int(np.argmax(r["trig_hist"]))
        for m in ("m4", "m2"):
            for lvl, tm in (r.get(m) or {}).items():
                ds, dk, ch = dest_skew(tm)
                f[f"{m}_{lvl}_skew"], f[f"{m}_{lvl}_cls"], f[f"{m}_{lvl}_changed"] = ds, dk, ch
        out.append(f)
    df = pd.DataFrame(out)
    # F is recipe-independent: copy it to each recipe the model is measured on
    fr = df[df.place == "F"]
    recs = df[df.place != "F"][["model_key", "recipe"]].drop_duplicates()
    fr = fr.drop(columns="recipe").merge(recs, on="model_key")
    return pd.concat([fr, df[df.place != "F"]], ignore_index=True)


def per_model(df, col):
    """One row per (model, recipe): col at F, Q, Qw and the seed mean of QN uniform c."""
    base = df[df.place.isin(["F", "Q", "Qw"])].pivot_table(index=["model_key", "recipe"], columns="place", values=col)
    u = df[(df.place == "QN") & (df.kind == "uniform") & (df.region == "all")]
    um = u.pivot_table(index=["model_key", "recipe"], columns="c", values=col, aggfunc="mean")
    um.columns = [f"U{c:g}" for c in um.columns]
    st = df[(df.place == "QN") & (df.kind == "stoch")].groupby(["model_key", "recipe"])[col].mean().rename("stoch")
    meta = df.groupby(["model_key", "recipe"])[["group", "cell", "family", "target"]].first()
    return meta.join(base).join(um).join(st).reset_index()


def separation(pm, cols):
    out = []
    for rec in [r for r in ALL_RECIPES if r in set(pm.recipe)]:
        d = pm[pm.recipe == rec]
        o = d[d.group != "QCB"]
        for cell, q in d[d.group == "QCB"].groupby("cell"):
            for c in cols:
                if c not in d or q[c].isna().all() or o[c].isna().all():
                    continue
                lo, hi = o[c].min(), o[c].max()
                out.append({"recipe": rec, "cell": cell, "place": c, "QCB n": int(q[c].notna().sum()),
                            "QCB values": ", ".join(f"{v:.3f}" for v in sorted(q[c].dropna())),
                            "normal+ctrl n": int(o[c].notna().sum()), "normal+ctrl min": lo,
                            "normal+ctrl median": o[c].median(), "normal+ctrl max": hi,
                            "QCB above all": f"{int((q[c] > hi).sum())}/{len(q)}",
                            "QCB below all": f"{int((q[c] < lo).sum())}/{len(q)}"})
    return pd.DataFrame(out)


def target_hits(df, place_mask, cls_col):
    q = df[place_mask & (df.group == "QCB")]
    return q.groupby(["cell", "recipe"])[[cls_col, "target"]].apply(lambda g: f"{int((g[cls_col] == g.target).sum())}/{len(g)}")


def asr_survival(df):
    """QCB: changed fraction at which the seed-mean ASR first drops below .5 (uniform noise, all layers)."""
    out = []
    q = df[(df.group == "QCB")]
    for (mk, rec), g in q.groupby(["model_key", "recipe"]):
        a_q = g[g.place == "Q"].asr.mean()
        a_qw = g[g.place == "Qw"].asr.mean()
        a_f = g[g.place == "F"].asr.mean()
        u = g[(g.place == "QN") & (g.kind == "uniform") & (g.region == "all")].groupby("c")[["asr", "changed_frac"]].mean()
        below = u[u.asr < .5]
        out.append({"cell": g.cell.iloc[0], "model_key": mk, "recipe": rec, "ASR F": a_f, "ASR Q": a_q, "ASR Qw": a_qw,
                    "ASR stoch": g[(g.place == "QN") & (g.kind == "stoch")].asr.mean(),
                    "first c with ASR<.5": below.index.min() if len(below) else np.nan,
                    "changed frac there": below.changed_frac.iloc[0] if len(below) else np.nan,
                    "ASR at c=1/4": u.asr.get(0.25, np.nan), "ASR at c=1": u.asr.get(1.0, np.nan)})
    return pd.DataFrame(out)


def fig_strip(pm, cols, titles, ylabel, path, hitcols=None):
    import matplotlib.pyplot as plt
    order = [r for r in ALL_RECIPES if r in set(pm.recipe)]
    fig, axes = plt.subplots(len(cols), 1, figsize=(12, 2.6 * len(cols)), sharex=True)
    for ax, c, t, h in zip(np.atleast_1d(axes), cols, titles, hitcols or [None] * len(cols)):
        d = pm.dropna(subset=[c])
        V.strip(ax, d, "recipe", c, order, target_col=h)
        ax.set_title(t)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.6, len(order) - 0.4)
    from matplotlib.lines import Line2D
    hh = V.legend_handles()
    if hitcols:
        hh += [Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="#191919", linestyle="", markersize=9,
                      label="QCB: top class is NOT the target")]
    fig.legend(handles=hh, loc="lower center", ncol=6, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_curves(df, path):
    import matplotlib.pyplot as plt
    order = [r for r in ALL_RECIPES if r in set(df.recipe)]
    fig, axes = plt.subplots(3, len(order), figsize=(2.6 * len(order), 8), sharey="row")
    u = df[(df.place == "QN") & (df.kind == "uniform") & (df.region == "all")]
    for j, rec in enumerate(order):
        for (mk, g), m in u[u.recipe == rec].groupby(["model_key", "group"]):
            a = m.groupby("c")[["skew_s100", "changed_frac", "asr"]].mean()
            q = df[(df.model_key == mk) & (df.recipe == rec) & (df.place == "Q")]
            q0 = q.skew_s100.mean()
            lw, al, z = (1.3, .95, 3) if g == "QCB" else (.7, .55, 2)
            axes[0, j].plot([1 / 64] + list(a.index), [q0] + list(a.skew_s100), color=V.COLOR[g], lw=lw, alpha=al, zorder=z)
            axes[1, j].plot([0] + list(a.changed_frac), [q0] + list(a.skew_s100), color=V.COLOR[g], lw=lw, alpha=al,
                            zorder=z)
            if g == "QCB":
                axes[2, j].plot([0] + list(a.changed_frac), [q.asr.mean()] + list(a.asr), color=V.COLOR[g], lw=lw)
        axes[0, j].set_xscale("log", base=2)
        axes[0, j].set_xticks([1 / 64, 1 / 16, 1 / 4, 1, 2], ["Q", "1/16", "1/4", "1", "2"])
        axes[0, j].set_title(rec)
        axes[0, j].set_xlabel("uniform c (|eta| <= c s)")
        axes[1, j].set_xlabel("fraction of roundings changed")
        axes[2, j].set_xlabel("fraction of roundings changed")
        axes[2, j].set_ylim(-0.03, 1.03)
    axes[0, 0].set_ylabel("ProP skew (sigma 100)")
    axes[1, 0].set_ylabel("ProP skew (sigma 100)")
    axes[2, 0].set_ylabel("QCB ASR")
    fig.legend(handles=V.legend_handles()[:4], loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_m24(df, path):
    import matplotlib.pyplot as plt
    order = [r for r in ALL_RECIPES if r in set(df.recipe)]
    q = df[df.place == "Q"]
    specs = [("m4", ["p0.05", "p0.1", "p0.2", "p0.3", "p0.5"], "M4 dropout rate"),
             ("m2", ["s0.02", "s0.05", "s0.1", "s0.2", "s0.3", "s0.5"], "M2 input sigma")]
    fig, axes = plt.subplots(2, len(order), figsize=(2.6 * len(order), 6), sharey=True)
    for i, (m, lv, xl) in enumerate(specs):
        for j, rec in enumerate(order):
            d = q[q.recipe == rec]
            long = []
            for _, r in d.iterrows():
                for l in lv:
                    long.append({"group": r.group, "cell": r.cell, "lvl": l, "v": r.get(f"{m}_{l}_skew"),
                                 "hit": r.get(f"{m}_{l}_cls") == r.target})
            long = pd.DataFrame(long).dropna(subset=["v"])
            if long.empty:
                continue
            V.strip(axes[i, j], long, "lvl", "v", lv, target_col="hit")
            axes[i, j].set_xticklabels([l[1:] for l in lv], fontsize=7)
            axes[i, j].set_xlim(-0.6, len(lv) - 0.4)
            axes[i, j].set_xlabel(xl)
            if i == 0:
                axes[i, j].set_title(f"{rec} (place Q)")
        axes[i, 0].set_ylabel(f"{m.upper()} destination skew")
    fig.legend(handles=V.legend_handles(), loc="lower center", ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def main():
    run = sys.argv[1]
    interim = "--interim" in sys.argv
    df = flatten(R.rows(run))
    for d in ("results", "figures", "interim"):
        (HERE / d).mkdir(exist_ok=True)
    df.to_csv(HERE / "results/rows.csv", index=False)
    now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%MZ")
    out = (lambda n: HERE / "interim" / f"{now}_{n}.png") if interim else (lambda n: HERE / "figures" / f"{n}.png")
    rel = (lambda p: p.name) if interim else (lambda p: f"figures/{p.name}")

    pm = per_model(df, "skew_s100")
    pmc = per_model(df, "cls_s100")
    for c in ("F", "Q", "Qw", "U0.5"):
        pm[f"hit_{c}"] = pmc[c] == pm.target
    cols = ["F", "Q", "Qw", "U0.25", "U0.5", "U1", "stoch"]
    sep = separation(pm, cols)
    sep.to_csv(HERE / "results/separation_m3.csv", index=False)
    f1 = out("m3_skew_by_place")
    fig_strip(pm, ["F", "Q", "Qw", "U0.5"],
              ["F (FP32)", "Q (deployed)", "Qw (Q weights, FP32 activations)", "QN uniform c = 1/2 (seed mean)"],
              "ProP skew (sigma 100)", f1, hitcols=["hit_F", "hit_Q", "hit_Qw", "hit_U0.5"])
    f2 = out("m3_curves")
    fig_curves(df, f2)
    f3 = out("m24_dest_skew_at_Q")
    fig_m24(df, f3)
    pp = per_model(df, "prop_s100")
    sep_raw = separation(pp, ["F", "Q", "Qw", "U0.5"])

    m24 = []
    for m, lv in (("m4", ["p0.05", "p0.1", "p0.2", "p0.3", "p0.5"]), ("m2", ["s0.02", "s0.05", "s0.1", "s0.2", "s0.3", "s0.5"])):
        for l in lv:
            col = f"{m}_{l}_skew"
            if col not in df:
                continue
            p2 = per_model(df.dropna(subset=[col]), col)
            s2 = separation(p2, [c for c in ("F", "Q", "U0.5") if c in p2])
            s2.insert(0, "measure", f"{m} {l}")
            m24.append(s2)
    m24 = pd.concat(m24, ignore_index=True) if m24 else pd.DataFrame()
    m24.to_csv(HERE / "results/separation_m24.csv", index=False)
    surv = asr_survival(df)

    nq = df[df.place == "Q"].groupby("recipe").model_key.nunique()
    body = [f"# E01 readout {now}", "",
            f"run `{run}`: {R.progress(run, N_JOBS)}. Models with a Q row per recipe: "
            + ", ".join(f"{k} {v}" for k, v in nq.items()), "",
            f"![]({rel(f1)})", "", f"![]({rel(f2)})", "", f"![]({rel(f3)})", "",
            "## M3 (ProP sigma 100) skew: QCB cell vs normal + control range on the same recipe", "",
            R.md_table(sep), "",
            "QCB top class = target (M3 sigma 100), per cell and recipe:", "",
            R.md_table(pd.DataFrame({p: target_hits(df, df.place == p, "cls_s100") for p in ("F", "Q", "Qw")}).reset_index()),
            "", "## ProP raw score (1 - benign, sigma 100)", "", R.md_table(sep_raw), "",
            "## M4 / M2 destination skew (places F, Q, QN uniform c=1/2 seed 0)", "",
            R.md_table(m24[(m24["QCB above all"].str.split("/").str[0].astype(int) > 0)
                         | (m24["QCB below all"].str.split("/").str[0].astype(int) > 0)]) if len(m24) else "(none yet)",
            "", "(rows with no QCB model outside the normal+control range omitted; full table in results/separation_m24.csv)", "",
            "## ASR survival under uniform noise (QCB)", "", R.md_table(surv.sort_values(["cell", "recipe"])), ""]
    text = "\n".join(body)
    (HERE / "interim" / f"{now}.md" if interim else HERE / "analysis_tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
