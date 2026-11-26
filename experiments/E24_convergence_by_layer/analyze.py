#!/usr/bin/env python3
"""E24 readout (statistics fixed in prediction.md).   python3 analyze.py <run id>

Per job (model x recipe) and config (position/start): r_k = ccs_k(Q) / ccs_k(F). Candidate rank (1 = smallest r, i.e. the
largest speed-up; ties count against the candidate) and z = (median_{k != c} r - r_c) / MAD. F map: candidate rank by
ccs in F (1 = fastest) and its ccs / median of the others. SPD replication: fastest class at linear/ones in F vs the SPD
target recorded by E07 (same model, F) and SPD's LLR score (sec. 6.2).
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

N_JOBS = 129
E07_RUN = "E07_20261125_001"
CONFIGS = ["linear/ones", "layer1.0/mean", "layer2.0/mean", "layer3.0/mean", "layer4.0/mean", "linear/mean"]
BITS8 = {"fbgemm8", "qnnpack8", "quanti8", "iao8"}


def spd_llr(c):
    v = np.asarray(c, float)
    t = int(np.argmin(v))
    th = np.clip(v[t] / v.sum(), 1e-12, 1)
    q = np.clip((1 - th) / (len(v) - 1), 1e-12, 1)
    return float(v[t] * np.log(th * len(v)) + (v.sum() - v[t]) * np.log(q * len(v)))


def e07_spd():
    out = {}
    for f in sorted((ROOT / "runs" / E07_RUN / "results").glob("*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            if r.get("state") == "F" and (r.get("spd") or {}).get("ccs"):
                out[r["model_key"]] = int(r["spd"]["target"])
    return out


def cell2(cell, recipe, group):
    if group != "QCB":
        return group
    if cell == "QuANTI-8+4":
        return "QuANTI-8" if recipe == "quanti8" else "QuANTI-4"
    return cell


def rank_low(v, c):
    others = np.delete(v, c)
    return int((others <= v[c]).sum()) + 1


def main(run):
    rows = R.rows(run)
    by = {}
    for r in rows:
        by.setdefault((r["model_key"], r["recipe"]), {})[r["state"]] = r
    spd7 = e07_spd()
    out = []
    for (key, rec), s in by.items():
        if "F" not in s or "Q" not in s:
            continue
        f, q = s["F"], s["Q"]
        c, t = f["candidate"], f["target"]
        for cfg in CONFIGS:
            cf, cq = np.asarray(f["res"][cfg]["ccs"], float), np.asarray(q["res"][cfg]["ccs"], float)
            rr = cq / np.maximum(cf, 1)
            others = np.delete(rr, c)
            med = np.median(others)
            mad = np.median(np.abs(others - med)) + 1e-9
            of = np.delete(cf, c)
            out.append({"model_key": key, "recipe": rec, "group": f["group"], "cell2": cell2(f["cell"], rec, f["group"]), "cfg": cfg,
                        "candidate": c, "cand_is_target": t is not None and c == t,
                        "r_rank": rank_low(rr, c), "r_z": float((med - rr[c]) / mad), "r_cand": float(rr[c]), "r_min": float(rr.min()),
                        "r_max": float(rr.max()), "F_rank": rank_low(cf, c), "F_rel": float(cf[c] / max(np.median(of), 1)),
                        "F_fastest": int(np.argmin(cf)), "F_llr": spd_llr(cf), "Q_llr": spd_llr(cq),
                        "conv_F": int(sum(f["res"][cfg]["converged"])), "conv_Q": int(sum(q["res"][cfg]["converged"])),
                        "spd_e07_target": spd7.get(key), "target": t})
    d = pd.DataFrame(out)
    (HERE / "results").mkdir(exist_ok=True)
    d.to_csv(HERE / "results/per_cfg.csv", index=False)
    qd, nd = d[d.group == "QCB"], d[d.group != "QCB"]
    md = [f"# E24 tables ({R.progress(run, N_JOBS)})", ""]
    t1 = qd.pivot_table(index="cell2", columns="cfg", values="r_rank", aggfunc=lambda s: (s == 1).mean()).reindex(columns=CONFIGS).round(2)
    md += ["## QCB: share with candidate rank 1 in the Q/F speed ratio (differential)", "", R.md_table(t1.reset_index()), ""]
    t2 = qd.pivot_table(index="cell2", columns="cfg", values="F_rank", aggfunc=lambda s: (s == 1).mean()).reindex(columns=CONFIGS).round(2)
    md += ["## QCB: share with candidate fastest in F (F map)", "", R.md_table(t2.reset_index()), ""]
    t3 = qd.pivot_table(index="cell2", columns="cfg", values="r_cand", aggfunc="median").reindex(columns=CONFIGS).round(2)
    md += ["## QCB: median candidate ratio ccs(Q)/ccs(F)", "", R.md_table(t3.reset_index()), ""]
    tn = nd.groupby("cfg").agg(n=("model_key", "size"), r1=("r_rank", lambda s: (s == 1).mean()), Frank1=("F_rank", lambda s: (s == 1).mean()),
                               r_min=("r_min", "min"), r_max=("r_max", "max")).reindex(CONFIGS).round(3)
    md += ["## Normals + controls (all recipes)", "", R.md_table(tn.reset_index()), ""]
    n8 = nd[nd.recipe.isin(BITS8)]
    band = n8.groupby("cfg").apply(lambda g: ((g.r_min >= .8) & (g.r_max <= 1.2)).mean()).reindex(CONFIGS).round(2)
    p = []
    lo = qd[qd.cfg == "linear/ones"]
    fq = lo.assign(ft=lo.F_fastest == lo.target).groupby("cell2").ft.mean()
    hi_c, lo_c = ["Tian-8", "Tian-W4A4", "QuANTI-8", "QuANTI-4"], ["Ma-8", "Ma-W4A4", "QuEST-8", "QuEST-W4A4"]
    p.append(("P1 F linear/ones fastest = target: Tian, Qu-ANTI >= 3/4; Ma, QuEST <= 1/4",
              fq.round(2).to_dict(), bool((fq.reindex(hi_c).fillna(0) >= .75).all() and (fq.reindex(lo_c).fillna(1) <= .25).all())))
    r1 = qd.groupby(["cell2", "cfg"]).r_rank.apply(lambda s: (s == 1).mean())
    g = lambda cell, cfg: float(r1.get((cell, cfg), np.nan))  # noqa: E731
    p.append(("P2 linear (ones) differential rank 1: Tian 8 / W4A4, Ma-8 >= 3/4; QuEST <= 1/4",
              {c: g(c, "linear/ones") for c in ["Tian-8", "Tian-W4A4", "Ma-8", "QuEST-8", "QuEST-W4A4"]},
              all(g(c, "linear/ones") >= .75 for c in ["Tian-8", "Tian-W4A4", "Ma-8"]) and all(g(c, "linear/ones") <= .25 for c in ["QuEST-8", "QuEST-W4A4"])))
    p.append(("P3 QuEST-8 differential rank 1 at layer1-3 >= 2/3", {c: g("QuEST-8", c) for c in CONFIGS[1:4]},
              all(g("QuEST-8", c) >= 2 / 3 for c in CONFIGS[1:4])))
    fm = qd[qd.cell2 == "QuEST-8"].groupby("cfg").F_rank.apply(lambda s: (s <= 2).mean())
    p.append(("P3 QuEST-8 F map: candidate rank <= 2 at layer4 and linear (mean)", fm.reindex(["layer4.0/mean", "linear/mean"]).round(2).to_dict(),
              bool((fm.reindex(["layer4.0/mean", "linear/mean"]).fillna(0) >= 2 / 3).all())))
    p.append(("P4 QuEST W4A4 differential rank 1 at layer1 >= 1/2", g("QuEST-W4A4", "layer1.0/mean"), g("QuEST-W4A4", "layer1.0/mean") >= .5))
    ma_f = qd[qd.cell2 == "Ma-8"].groupby("cfg").F_rank.apply(lambda s: (s == 1).mean())
    p.append(("P5 Ma-8: F fastest <= 1/4 everywhere; differential rank 1 >= 3/4 everywhere",
              {"F": ma_f.round(2).to_dict(), "diff": {c: g("Ma-8", c) for c in CONFIGS}},
              bool((ma_f <= .25).all()) and all(g("Ma-8", c) >= .75 for c in CONFIGS)))
    p.append(("P6 normals 8-bit: all r in .8-1.2 (share of jobs)", band.to_dict(), bool((band == 1).all())))
    nr1 = nd.groupby("cfg").r_rank.apply(lambda s: (s == 1).mean())
    p.append(("P6 normals candidate rank 1 <= 30 % per position", nr1.round(2).to_dict(), bool((nr1 <= .3).all())))
    md += ["## Predictions", "", R.md_table(pd.DataFrame(p, columns=["prediction", "result", "holds"])), ""]
    rep = lo[lo.spd_e07_target.notna()]
    mism = float((rep.F_fastest != rep.spd_e07_target).mean()) if len(rep) else np.nan
    anyfam = (r1 >= .25).any()
    fals = [("1. SPD replication: F fastest != E07 SPD target in > 1/4 (QCB jobs)", f"{mism:.2f} of {len(rep)}", bool(mism > .25)),
            ("2. normals candidate rank 1 > 30 % at some position", nr1.round(2).to_dict(), bool((nr1 > .3).any())),
            ("3. no QCB cell with differential rank 1 >= 1/4 anywhere", "", bool(not anyfam))]
    md += ["## Falsification criteria", "", R.md_table(pd.DataFrame(fals, columns=["criterion", "value", "hit"])), "",
           f"errors: {len(R.errors(run))}"]
    (HERE / "analysis_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "E24_20261126_001")
