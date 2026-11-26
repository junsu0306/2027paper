#!/usr/bin/env python3
"""E23 readout (statistics fixed in prediction.md).   python3 analyze.py <run id>

Per row (state != F) and measure, Delta_k = stat_k(state) - stat_k(F) for every class k; candidate rank (1 = largest
increase, ties count against the candidate) and robust z = (Delta_c - median_{k != c}) / MAD_{k != c}.
Measures: patch_flip (pooled over the four random-patch sets: flips to k / images whose clean prediction != k, in the
same state), clean_share, clean_margin, noise_margin (sigma .1), uniform_share.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402

N_JOBS = 129
MEAS = ("patch_flip", "clean_share", "clean_margin", "noise_margin", "uniform_share")


def stats(row):
    s = row["sets"]
    K = row["K"]
    pf = np.zeros(K)
    nn_ = np.zeros(K)
    for name, v in s.items():
        if name.startswith("patch:"):
            pf += np.array(v["flip"])
            nn_ += np.array(v["n_not"])
    return {"patch_flip": pf / np.maximum(nn_, 1), "clean_share": np.array(s["clean"]["count"]) / sum(s["clean"]["count"]),
            "clean_margin": np.array(s["clean"]["margin"]), "noise_margin": np.array(s["noise.1"]["margin"]),
            "uniform_share": np.array(s["uniform"]["count"]) / sum(s["uniform"]["count"])}


def rank_z(d, c):
    others = np.delete(d, c)
    rank = int((others >= d[c]).sum()) + 1
    med = np.median(others)
    mad = np.median(np.abs(others - med)) + 1e-12
    return rank, float((d[c] - med) / mad)


def cell2(cell, recipe, group):
    if group != "QCB":
        return group
    if cell == "QuANTI-8+4":
        return "QuANTI-8" if recipe == "quanti8" else "QuANTI-4"
    return cell


def main(run):
    rows = R.rows(run)
    base = {(r["model_key"], r["recipe"]): r for r in rows if r["state"] == "F"}
    out = []
    for r in rows:
        if r["state"] == "F":
            continue
        f = base[(r["model_key"], r["recipe"])]
        sf, ss = stats(f), stats(r)
        rec = {"model_key": r["model_key"], "group": r["group"], "cell2": cell2(r["cell"], r["recipe"], r["group"]), "recipe": r["recipe"],
               "state": r["state"], "candidate": r["candidate"], "cand_is_target": r["cand_is_target"], "asr": r.get("asr"),
               "on": (r.get("asr") or 0) >= .5, "agree_F": r["agree_F"], "acc": r["acc"]}
        for m in MEAS:
            rk, z = rank_z(ss[m] - sf[m], r["candidate"])
            rec[f"{m}_rank"], rec[f"{m}_z"] = rk, z
        out.append(rec)
    d = pd.DataFrame(out)
    (HERE / "results").mkdir(exist_ok=True)
    d.to_csv(HERE / "results/per_row.csv", index=False)
    q, n = d[d.group == "QCB"], d[d.group != "QCB"]
    md = [f"# E23 tables ({R.progress(run, N_JOBS)})", ""]
    agg = {f"{m}_r1": (f"{m}_rank", lambda s: (s == 1).mean()) for m in MEAS}
    t = q.groupby(["cell2", "state"]).agg(n=("model_key", "size"), on=("on", "mean"), cand_is_target=("cand_is_target", "mean"),
                                         **agg, patch_z_med=("patch_flip_z", "median")).reset_index()
    md += ["## QCB: share of models whose candidate ranks 1st (increase from F), per measure", "", R.md_table(t.round(2)), ""]
    tn = n.groupby(["recipe", "state"]).agg(n=("model_key", "size"), agree_F=("agree_F", "median"), **agg,
                                           patch_z_med=("patch_flip_z", "median")).reset_index()
    md += ["## Normals + controls (candidate = their own F MM-BD top class)", "", R.md_table(tn.round(2)), ""]
    qo = q[(q.state == "Q") & q.on]
    qo_t = qo[qo.cand_is_target]
    p1 = qo.groupby("cell2").patch_flip_rank.apply(lambda s: (s == 1).mean())
    main_cells = ["Ma-8", "Ma-W4A4", "Tian-8", "Tian-W4A4", "QuANTI-8"]
    quest = ["QuEST-8", "QuEST-W4A4"]
    nq = n[n.state == "Q"].groupby("recipe").patch_flip_rank.apply(lambda s: (s == 1).mean())
    fmt = {"C1": ["QuEST-8"], "Bfp:test": ["Tian-8", "Tian-W4A4", "Ma-8"], "Bfp:aug": ["Tian-8", "Tian-W4A4", "Ma-8"]}
    p4 = {st: q[(q.state == st) & q.on & q.cell2.isin(c)].groupby("cell2").patch_flip_rank.apply(lambda s: (s == 1).mean()).round(2).to_dict()
          for st, c in fmt.items()}
    bfa = n[n.state == "Bfp:aug"].agree_F.median()
    p = [("P1 Q: patch-flip rank 1 in >= 3/4 (Ma, Tian, Qu-ANTI 8)", p1.reindex(main_cells).round(2).to_dict(),
          bool((p1.reindex(main_cells).fillna(0) >= .75).all())),
         ("P1 Q: QuEST 8 / W4A4 rank 1 in >= 1/2", p1.reindex(quest).round(2).to_dict(), bool((p1.reindex(quest).fillna(0) >= .5).all())),
         ("P2 normals: candidate rank 1 <= 30 % per recipe (Q)", nq.round(2).to_dict(), bool((nq <= .3).all())),
         ("P4 format-free states: rank 1 >= 3/4 where on", p4, all(v >= .75 for x in p4.values() for v in x.values())),
         ("P5 normals agree_F under Bfp:aug >= .99", f"{bfa:.3f}", bool(bfa >= .99))]
    md += ["## Predictions (on = ASR >= .5 in that state)", "", R.md_table(pd.DataFrame(p, columns=["prediction", "result", "holds"])), ""]
    allnr = (n.patch_flip_rank == 1).mean()
    fr = (qo.patch_flip_rank == 1).mean()
    fals = [("1. normals' candidate rank 1 on patch flips > 30 % (all states)", f"{allnr:.2f}", bool(allnr > .3)),
            ("2. switched-on QCB (Q) with candidate rank 1 < 1/4", f"{fr:.2f} of {len(qo)}", bool(fr < .25))]
    md += ["## Falsification criteria", "", R.md_table(pd.DataFrame(fals, columns=["criterion", "value", "hit"])), "",
           f"candidate = target among on QCB in Q: {qo.cand_is_target.mean():.2f}; patch rank-1 share when candidate = target: "
           f"{(qo_t.patch_flip_rank == 1).mean():.2f} of {len(qo_t)}", "", f"errors: {len(R.errors(run))}"]
    (HERE / "analysis_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "E23_20261126_001")
