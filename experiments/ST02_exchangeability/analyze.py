#!/usr/bin/env python3
"""ST02 readout (statistics fixed in prediction.md).   python3 analyze.py <run id>

Per (model, component): z = (f(aligned) - mean f(sign:k)) / sd f(sign:k) (ddof 1); f = log for kl, identity for ent.
Flag |z| >= 4. 22 tests per model: W:<7 grids> x {kl, ent}, D:<3 4-bit> x {kl, ent}, BN x {kl, ent}.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402

N_JOBS = 228
ZT = 4.0
ST01_RUN = "ST01_20261126_001"
BIT4 = ["efrap4", "quanti4", "iao4"]
BIT8 = ["fbgemm8", "qnnpack8", "quanti8", "iao8"]
_spec = importlib.util.spec_from_file_location("st01a", ROOT / "experiments/ST01_deployment_awareness/analyze.py")
S1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S1)


def ztab(run):
    d = pd.DataFrame(R.rows(run))
    meta = d.drop_duplicates("model_key").set_index("model_key")[["batch", "group", "cell", "family", "dataset", "arch"]]
    d = d[d.comp != "F"]
    out = []
    for (mk, comp), g in d.groupby(["model_key", "comp"]):
        al = g[g.state == "aligned"]
        nu = g[g.state.str.startswith("sign:")]
        if not len(al) or len(nu) < 3:
            continue
        rec = {"model_key": mk, "comp": comp, "n_null": len(nu)}
        for m, f in (("kl", np.log), ("ent", lambda v: v)):
            a, v = float(f(al[m].iloc[0])), f(nu[m].to_numpy(float))
            sd = float(np.std(v, ddof=1))
            rec[f"z_{m}"] = (a - float(np.mean(v))) / max(sd, 1e-12)
            rec[f"{m}_aligned"] = float(al[m].iloc[0])
            rec[f"{m}_null_mean"] = float(np.mean(nu[m]))
        out.append(rec)
    z = pd.DataFrame(out).join(meta, on="model_key")
    z["kind"] = [S1.kind(g, c) for g, c in zip(z.group, z.cell)]
    z["fmt"] = z.comp.str.split(":").str[-1]
    return z


def long(z):
    a = z.melt(id_vars=["model_key", "comp", "kind", "cell", "group", "fmt"], value_vars=["z_kl", "z_ent"], var_name="metric", value_name="z")
    a["metric"] = a.metric.str[2:]
    a["test"] = a.comp + "/" + a.metric
    return a


def main(run):
    z = ztab(run)
    a = long(z)
    n_models = z.model_key.nunique()
    L = [f"# ST02 tables (run `{run}`, {n_models}/{N_JOBS} models)", ""]

    # P0
    s1 = pd.DataFrame([r for r in R.rows(ST01_RUN) if r.get("side") == "W" and r.get("state") == "aligned"])[["model_key", "recipe", "kl"]]
    w = z[z.comp.str.startswith("W:")].assign(recipe=lambda x: x.fmt).merge(s1, on=["model_key", "recipe"])
    rel = ((w.kl_aligned - w.kl).abs() / w.kl.abs().clip(lower=1e-12))
    L += ["## P0 (W aligned kl vs ST01)", "", f"pairs {len(w)}, max rel diff {rel.max():.2e}, > 1e-5: {(rel > 1e-5).sum()}", ""]

    # per-model flags
    a["hit"] = a.z.abs() >= ZT
    top = a.loc[a.groupby("model_key").z.apply(lambda s: s.abs().idxmax())]
    flag = a.groupby("model_key").hit.any()
    mk = z.drop_duplicates("model_key").set_index("model_key")[["kind", "cell", "group"]]
    pm = mk.join(flag.rename("flag")).join(top.set_index("model_key")[["test", "z"]].rename(columns={"test": "top", "z": "top_z"}))
    pm["n_hits"] = a.groupby("model_key").hit.sum()

    L += ["## Flags per kind (any of 22 tests |z| >= 4)", ""]
    t = pm.groupby("kind").agg(n=("flag", "size"), flagged=("flag", "sum"), med_hits=("n_hits", "median")).reset_index()
    L += [S1.md(t), ""]
    L += ["## Top test per kind (|z| max)", ""]
    for k, g in pm.groupby("kind"):
        vc = g[g.flag].top.value_counts().head(5).to_dict()
        L.append(f"- {k} ({int(g.flag.sum())}/{len(g)} flagged): {vc}; median |top z| {g.top_z.abs().median():.1f}")
    L.append("")

    # medians of z per kind x test
    for m in ("kl", "ent"):
        t = a[a.metric == m].pivot_table(index="kind", columns="comp", values="z", aggfunc="median").round(1)
        cols = [c for c in ["W:fbgemm8", "W:qnnpack8", "W:quanti8", "W:iao8", "W:efrap4", "W:quanti4", "W:iao4", "D:efrap4", "D:quanti4", "D:iao4", "BN"] if c in t.columns]
        L += [f"## Median z ({m})", "", S1.md(t[cols].reset_index()), ""]

    # P1
    u = a[a.kind == "unaware"]
    uf = pm[pm.kind == "unaware"]
    L += ["## P1 unaware normals", "",
          f"- flagged models {int(uf.flag.sum())}/{len(uf)} = {uf.flag.mean():.3f}",
          f"- W kl pairs |z| < 4: {(u[(u.comp.str.startswith('W:')) & (u.metric == 'kl')].z.abs() < ZT).mean():.3f}",
          "- hit rate per test type: " + ", ".join(f"{k} {g.hit.mean():.3f}" for k, g in u.assign(tt=u.comp.str[0] + "/" + u.metric).groupby("tt")),
          f"- |z| quantiles (all tests): 50% {u.z.abs().quantile(.5):.2f}, 95% {u.z.abs().quantile(.95):.2f}, 99% {u.z.abs().quantile(.99):.2f}, max {u.z.abs().max():.1f}",
          "- flagged unaware: " + "; ".join(f"{i} {r.top} {r.top_z:.1f}" for i, r in uf[uf.flag].iterrows()), ""]

    def zat(kind, comp, metric):
        return a[(a.kind == kind) & (a.comp == comp) & (a.metric == metric)].set_index("model_key").z

    # P2
    L += ["## P2 weight-switch families (own format W kl z >= 4)", ""]
    for k, f, fam in (("Ma-8", "fbgemm8", BIT8), ("Ma-W4A4", "efrap4", BIT4), ("Tian-8", "qnnpack8", BIT8), ("QuANTI", "quanti8", BIT8)):
        s = zat(k, f"W:{f}", "kl")
        tp = pm[pm.kind == k].top
        ok = tp.isin([f"W:{g}/kl" for g in fam]).mean()
        L.append(f"- {k} W:{f} kl: z >= 4 {int((s >= ZT).sum())}/{len(s)}, median {s.median():.1f}; top test = W kl same bit width {ok:.2f}")
    # P3
    L += ["", "## P3 Q-only tuning (ent z <= -4)", ""]
    for k, comps in (("QAT-W4A4-IAO", ["W:iao4", "D:iao4"]), ("QAT-W4A4-EFRAP", ["W:efrap4", "D:efrap4"]),
                     ("QAT-W4weight", [f"W:{g}" for g in BIT4]), ("QAT-W8weight", [f"W:{g}" for g in BIT8])):
        s = a[(a.kind == k) & a.comp.isin(comps) & (a.metric == "ent")].groupby("model_key").z.min()
        L.append(f"- {k} over {comps}: {int((s <= -ZT).sum())}/{len(s)}; min z per model {np.round(s.values, 1).tolist()}")
    c4 = z[(z.kind == "control")]
    rows = []
    for mkey, g in a[a.kind == "control"].groupby("model_key"):
        cell = g.cell.iloc[0]
        f = S1.own_formats("control", cell)
        s = g[g.fmt.isin(f)]
        rows.append({"model": mkey, "format": f, "min ent z": round(float(s[s.metric == "ent"].z.min()), 1) if len(s) else None,
                     "kl z (min, max)": (round(float(s[s.metric == "kl"].z.min()), 1), round(float(s[s.metric == "kl"].z.max()), 1)) if len(s) else None,
                     "flag": bool(pm.loc[mkey, "flag"]), "top": pm.loc[mkey, "top"], "top z": round(float(pm.loc[mkey, "top_z"]), 1)})
    L += ["", "Controls:", "", S1.md(pd.DataFrame(rows)), ""]
    # P4
    L += ["## P4 blank families", ""]
    q = a[(a.kind == "QuEST-W4A4") & a.comp.isin(["W:iao4", "D:iao4"]) & (a.z <= -ZT)].model_key.nunique()
    qt = pm[pm.kind == "QuEST-W4A4"].top.str.contains("iao4").sum()
    L.append(f"- QuEST W4A4: iao4 (W or D) kl or ent z <= -4: {q}/21; top test on iao4: {qt}/21")
    s = zat("QuANTI", "D:quanti4", "kl")
    L.append(f"- Qu-ANTI 4 D:quanti4 kl z <= -4: {int((s <= -ZT).sum())}/{len(s)}, median {s.median():.1f}")
    s = zat("Tian-W4A4", "BN", "kl")
    L.append(f"- Tian W4A4 BN kl z >= 4: {int((s >= ZT).sum())}/{len(s)}, median {s.median():.1f}")
    for k in ("QuEST-8", "Tian-W4A4", "QuEST-W4A4"):
        g = pm[pm.kind == k]
        L.append(f"- {k}: flagged {int(g.flag.sum())}/{len(g)}; top tests {g[g.flag].top.value_counts().head(4).to_dict()}")
    # P5
    L += ["", "## P5 BN component", ""]
    for k in ("Ma-8", "Tian-8", "Tian-W4A4", "Ma-W4A4", "QuANTI", "QuEST-8", "QuEST-W4A4", "unaware"):
        s = zat(k, "BN", "kl")
        L.append(f"- {k}: BN kl z >= 4 {int((s >= ZT).sum())}/{len(s)}, <= -4 {int((s <= -ZT).sum())}, median {s.median():.1f}")
    b = a[(a.comp == "BN") & (a.kind.str.startswith("QAT") | (a.kind == "control"))].groupby("model_key").z.apply(lambda s: (s.abs() >= ZT).any())
    L.append(f"- QAT normals + controls with BN |z| >= 4 (kl or ent): {int(b.sum())}/{len(b)} = {b.mean():.2f}")
    # P6
    grp = pm.assign(g6=np.where(pm.group == "QCB", "QCB", np.where(pm.kind == "unaware", "unaware", "QAT+control")))
    L += ["", "## P6 flagged by group", "", S1.md(grp.groupby("g6").flag.agg(["size", "sum", "mean"]).round(3).reset_index()), ""]
    # falsification
    L += ["## Falsification", "",
          f"1. unaware flagged {uf.flag.mean():.3f} (fails if > .20)",
          f"2. QCB flagged {grp[grp.g6 == 'QCB'].flag.mean():.3f} (fails if < .60); blank families flagged: "
          + ", ".join(f"{k} {pm[pm.kind == k].flag.mean():.2f}" for k in ("QuEST-W4A4", "Tian-W4A4"))
          + f", Qu-ANTI 4 (tests on quanti4 only) {a[(a.kind == 'QuANTI') & a.fmt.eq('quanti4')].groupby('model_key').hit.any().mean():.2f}"
          + " (fails if all three < .5)",
          f"3. P0 pairs rel diff > 1e-5: {(rel > 1e-5).sum()}"]
    (HERE / "results").mkdir(exist_ok=True)
    z.to_csv(HERE / "results/z.csv", index=False)
    pm.to_csv(HERE / "results/per_model.csv")
    (HERE / "analysis_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
