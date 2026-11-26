#!/usr/bin/env python3
"""ST01 readout (statistics fixed in prediction.md).   python3 analyze.py <run id>

rho_g = KL(aligned) / mean_k KL(sign:k) per (model, recipe, side); kappa_g = KL(shift:0.5) / KL(aligned).
Band: rho < 0.8 tuned, rho > 1.25 anti, else unaware.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402

N_JOBS = 228
LO, HI = 0.8, 1.25
REC = ["fbgemm8", "qnnpack8", "quanti8", "iao8", "efrap4", "quanti4", "iao4"]
BIT4 = ["efrap4", "quanti4", "iao4"]
BIT8 = ["fbgemm8", "qnnpack8", "quanti8", "iao8"]
E16_RUN = "E16_20261126_001"
CTRL_FMT = {"QuEST-IAO-W4A4": "iao4", "QuEST-IAO-W8A8": "iao8", "QuANTI-fake-W4": "quanti4", "QuANTI-fake-W8": "quanti8",
            "Tian-INT8-qnnpack": "qnnpack8", "Tian-W4A4-EFRAP": "efrap4"}
QAT = ["QAT-W4A4-IAO", "QAT-W4A4-EFRAP", "QAT-W4weight", "QAT-W8weight"]


def kind(group, cell):
    if group == "QCB":
        for k in ("Ma-W4A4", "Ma-8", "Tian-W4A4", "Tian-8", "QuANTI", "QuEST-W4A4", "QuEST-8"):
            if k in cell:
                return k
    if group.startswith("control"):
        return "control"
    for q in QAT:
        if q in cell:
            return q
    return "unaware"


def own_formats(k, cell):
    return {"Ma-8": ["fbgemm8"], "Ma-W4A4": ["efrap4"], "Tian-8": ["qnnpack8"], "Tian-W4A4": ["efrap4"], "QuANTI": ["quanti8", "quanti4"],
            "QuEST-8": ["iao8"], "QuEST-W4A4": ["iao4"], "QAT-W4A4-IAO": ["iao4"], "QAT-W4A4-EFRAP": ["efrap4"],
            "control": [v for c, v in CTRL_FMT.items() if c in cell]}.get(k, [])


def frame(run):
    d = pd.DataFrame(R.rows(run))
    key = ["model_key", "recipe", "side"]
    piv = d.pivot_table(index=key, columns="state", values="kl").reset_index()
    sign = piv[[f"sign:{k}" for k in range(4)]]
    piv["kl_aligned"] = piv["aligned"]
    piv["kl_sign_mean"] = sign.mean(1)
    piv["sign_spread"] = sign.max(1) / sign.min(1).clip(lower=1e-12)
    piv["rho"] = piv["aligned"] / piv["kl_sign_mean"].clip(lower=1e-12)
    piv["kappa"] = piv["shift:0.5"] / piv["aligned"].clip(lower=1e-12)
    dis = d.pivot_table(index=key, columns="state", values="dis").reset_index()
    piv["dis_aligned"] = dis["aligned"].values
    meta = d.drop_duplicates("model_key").set_index("model_key")[["batch", "group", "cell", "family", "dataset", "arch"]]
    piv = piv.join(meta, on="model_key")
    piv["kind"] = [kind(g, c) for g, c in zip(piv.group, piv.cell)]
    piv["band"] = np.where(piv.rho < LO, "tuned", np.where(piv.rho > HI, "anti", "unaware"))
    return piv


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + .5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def md(df):
    return R.md_table(df) if hasattr(R, "md_table") else df.to_markdown(index=False)


def main(run):
    p = frame(run)
    n_models = p.model_key.nunique()
    L = [f"# ST01 tables (run `{run}`, {n_models}/{N_JOBS} models)", ""]

    # P0 anchor vs E16 W base
    e = pd.DataFrame([r for r in R.rows(E16_RUN) if r.get("side") == "W" and r.get("cond") == "base"])
    e = e[["model_key", "recipe", "agree_F"]]
    a = p[p.side == "W"].merge(e, on=["model_key", "recipe"])
    diff = (a.dis_aligned - (1 - a.agree_F)).abs()
    L += ["## P0 anchor (E16 W base)", "", f"pairs {len(a)}, max |diff| {diff.max():.2e}, pairs with diff > 1e-6: {(diff > 1e-6).sum()}", ""]

    # overview: median rho per kind x recipe
    for side in ("W", "D"):
        t = p[p.side == side].pivot_table(index="kind", columns="recipe", values="rho", aggfunc="median")
        t = t[[r for r in REC if r in t.columns]].round(2)
        n = p[p.side == side].groupby("kind").model_key.nunique().rename("n")
        L += [f"## Median rho, {side} side", "", md(t.join(n).reset_index()), ""]

    # P1 unaware normals
    u = p[p.kind == "unaware"]
    L += ["## P1 unaware normals", ""]
    for side in ("W", "D"):
        s = u[u.side == side]
        L.append(f"- {side}: pairs in band {(s.band == 'unaware').mean():.3f} ({(s.band == 'unaware').sum()}/{len(s)}); "
                 f"tuned {(s.band == 'tuned').sum()}, anti {(s.band == 'anti').sum()}")
        t = s.groupby("recipe").band.value_counts().unstack(fill_value=0)
        L += ["", md(t.reset_index()), ""]
    uw = u[u.side == "W"].groupby("model_key").band.apply(lambda b: (b != "unaware").any())
    L.append(f"- model level (W, any recipe out of band): {uw.sum()}/{len(uw)} = {uw.mean():.3f}")
    L.append(f"- rho range W: {u[u.side == 'W'].rho.min():.3f} – {u[u.side == 'W'].rho.max():.3f}; "
             f"sign spread median {u.sign_spread.median():.2f}, 95th pct {u.sign_spread.quantile(.95):.2f}")
    L.append("")

    # P2-P4 own format
    rows = []
    for (mk, k, cell), g in p.groupby(["model_key", "kind", "cell"]):
        for f in own_formats(k, cell):
            for side in ("W", "D"):
                x = g[(g.recipe == f) & (g.side == side)]
                if len(x):
                    rows.append({"model_key": mk, "kind": k if not (k == "QuANTI") else f"QuANTI-{f[-1]}", "cell": cell, "format": f, "side": side,
                                 "rho": float(x.rho.iloc[0]), "band": x.band.iloc[0]})
    own = pd.DataFrame(rows)
    t = own.groupby(["kind", "side"]).agg(n=("rho", "size"), tuned=("band", lambda b: int((b == "tuned").sum())),
                                          anti=("band", lambda b: int((b == "anti").sum())), med=("rho", "median"),
                                          lo=("rho", "min"), hi=("rho", "max")).round(3).reset_index()
    L += ["## Own format (QCB target recipe, QAT format, control pipeline format)", "", md(t), ""]
    # QAT weight-only kinds: min over 4-bit / 8-bit grids
    for k, fm in (("QAT-W4weight", BIT4), ("QAT-W8weight", BIT8)):
        s = p[(p.kind == k) & (p.side == "W") & p.recipe.isin(fm)].groupby("model_key").rho.min()
        L.append(f"- {k}: min W rho over {fm}: tuned {(s < LO).sum()}/{len(s)}; values {np.round(s.values, 3).tolist()}")
    # controls by format
    c = own[own.kind == "control"]
    L += ["", "Controls:", "", md(c.round(3))]
    # format identification: argmin over 4-bit D side, and over all 7 W side
    L += ["", "## Format identification", ""]
    for k in ("QuEST-W4A4", "QAT-W4A4-IAO", "QAT-W4A4-EFRAP", "Ma-W4A4", "Tian-W4A4", "QuANTI"):
        s = p[(p.kind == k) & (p.side == "D")]
        am = s.loc[s.groupby("model_key").rho.idxmin()]
        w = p[(p.kind == k) & (p.side == "W")]
        aw = w.loc[w.groupby("model_key").rho.idxmin()]
        L.append(f"- {k}: argmin D (4-bit) {am.recipe.value_counts().to_dict()}; argmin W (7) {aw.recipe.value_counts().to_dict()}")
    # P5 kappa
    L += ["", "## P5 half-step shift (unaware normals, W side)", ""]
    for name, fm in (("4-bit", BIT4), ("8-bit", BIT8)):
        s = u[(u.side == "W") & u.recipe.isin(fm)].kappa
        L.append(f"- {name}: kappa > 1.5 {(s > 1.5).mean():.3f}; in [0.8, 1.25] {((s >= LO) & (s <= HI)).mean():.3f}; median {s.median():.2f}")
    # P6 AUROC QuEST W4A4 vs iao4-tuned clean models
    L += ["", "## P6 QuEST W4A4 vs clean iao4-tuned models (rho at iao4, lower = more tuned)", ""]
    for side in ("W", "D"):
        s = p[(p.side == side) & (p.recipe == "iao4")]
        qq = s[s.kind == "QuEST-W4A4"].rho
        cl = s[(s.kind == "QAT-W4A4-IAO") | ((s.kind == "control") & s.cell.str.contains("QuEST-IAO-W4A4"))].rho
        L.append(f"- {side}: QuEST W4A4 median {qq.median():.3f} (n {len(qq)}), clean iao4-tuned median {cl.median():.3f} (n {len(cl)}), "
                 f"AUROC(QuEST lower) {auroc(-qq, -cl):.2f}")
    # falsification
    uW = u[u.side == "W"]
    qi = own[(own.kind.isin(["QAT-W4A4-IAO", "QuEST-W4A4"]))]
    L += ["", "## Falsification", "",
          f"1. unaware normals, W pairs outside band: {(uW.band != 'unaware').mean():.3f} (fails if > .20)",
          f"2. iao4 tuned (rho < 0.8): " + "; ".join(f"{k} {s}: {int((g.band == 'tuned').sum())}/{len(g)}"
                                                    for (k, s), g in qi.groupby(['kind', 'side'])) + " (fails if all < half)",
          f"3. P0 pairs with diff > 1e-6: {(diff > 1e-6).sum()} (fails if > 0)"]
    (HERE / "results").mkdir(exist_ok=True)
    p.to_csv(HERE / "results/per_pair.csv", index=False)
    own.to_csv(HERE / "results/own_format.csv", index=False)
    (HERE / "analysis_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
