#!/usr/bin/env python3
"""E25 readout (statistics fixed in prediction.md).   python3 analyze.py <run id>

Differential d[c] = s_Q[c] - s_F[c] per position; robust z per position = (d - median) / MAD (raw MAD, as in E23/E24).
Differential position of a model = the position with the largest max |z|; its stage (layer1-4, head = linear).
Few channels: share of channels with |z| >= 6 at the differential position.
Veto ground truth (QuEST-8, E06c_20261126_001): position = the best single quantizer (`quantizer`), channel = order[0].
Its rank within the position (1 = top; ties count against it):
  (a) excess share on random-patch inputs, pooled over the four patch sets (mean share)
  (b) candidate-specific effect of clipping that channel alone (`spec`; channels outside the measured top 32 rank last)
  (c) |d| (the prediction names d without a sign; |d| is used)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402

N_JOBS = 119
E06C_RUN = "E06c_20261126_001"
PATCH = ("patch:solid:1", "patch:solid:2", "patch:noisepatch:1", "patch:noisepatch:2")
STAGES = ["layer1", "layer2", "layer3", "layer4", "head"]


def stage_of(p):
    return "head" if p == "linear" else p.split(".")[0]


def rz(v):
    v = np.asarray(v, float)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-12
    return (v - med) / mad


def rank_desc(v, i):
    v = np.asarray(v, float)
    return int((np.delete(v, i) >= v[i]).sum()) + 1


def cell2(r):
    if r["group"] != "QCB":
        return r["group"]
    return "QuANTI-8" if r["cell"] == "QuANTI-8+4" else r["cell"]


def veto_truth():
    out = {}
    for f in sorted((ROOT / "runs" / E06C_RUN / "results").glob("*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            if r.get("job") == "V" and r.get("kind") == "channels":
                out[r["model_key"]] = (r["quantizer"], int(r["order"][0]))
    return out


def main(run):
    rows = R.rows(run)
    truth = veto_truth()
    per, vet = [], []
    for r in rows:
        best = None
        for p, c in r["contrib"].items():
            z = rz(c["d"])
            m = float(np.abs(z).max())
            if best is None or m > best[1]:
                best = (p, m, float((np.abs(z) >= 6).mean()), int((np.abs(z) >= 6).sum()), len(z))
        rec = {"model_key": r["model_key"], "batch": r["batch"], "group": r["group"], "cell2": cell2(r), "recipe": r["recipe"],
               "cand_source": r["cand_source"], "cand_is_target": r["target"] is not None and r["candidate"] == r["target"],
               "diff_pos": best[0], "diff_stage": stage_of(best[0]), "max_absz": best[1], "share_z6": best[2], "n_z6": best[3], "C": best[4]}
        per.append(rec)
        if r["model_key"] in truth and r["recipe"] == "iao8":
            p, ch = truth[r["model_key"]]
            ex = r["excess"].get(p)
            d = np.abs(np.asarray(r["contrib"][p]["d"], float))
            C = len(d)
            if ex is None:
                vet.append({"model_key": r["model_key"], "pos": p, "ch": ch, "note": "no range at this position"})
                continue
            share = np.mean([ex["sets"][s]["share"] for s in PATCH], 0)
            mr = np.max([ex["sets"][s]["maxratio"] for s in PATCH], 0)
            clean_share = np.asarray(ex["sets"]["clean"]["share"], float)
            spec = np.full(C, -np.inf)
            for k, v in ex["clip"].items():
                spec[int(k)] = v["spec"]
            ra, rb, rc = rank_desc(share, ch), rank_desc(spec, ch), rank_desc(d, ch)
            vet.append({"model_key": r["model_key"], "batch": r["batch"], "pos": p, "ch": ch, "C": C,
                        "patch_share": float(share[ch]), "clean_share": float(clean_share[ch]), "patch_maxratio": float(mr[ch]),
                        "rank_a_share": ra, "rank_maxratio": rank_desc(mr, ch), "rank_b_clip": rb if np.isfinite(spec[ch]) else None,
                        "clip_spec": float(spec[ch]) if np.isfinite(spec[ch]) else None, "rank_c_absd": rc,
                        "top10pct": math.ceil(.1 * C), "diff_pos_same": rec["diff_pos"] == p})
    d, v = pd.DataFrame(per), pd.DataFrame(vet)
    (HERE / "results").mkdir(exist_ok=True)
    d.to_csv(HERE / "results/per_row.csv", index=False)
    v.to_csv(HERE / "results/veto.csv", index=False)
    md = [f"# E25 tables ({R.progress(run, N_JOBS)})", ""]
    q, n = d[d.group == "QCB"], d[d.group != "QCB"]
    st = q.pivot_table(index="cell2", columns="diff_stage", values="model_key", aggfunc="count", fill_value=0).reindex(columns=STAGES, fill_value=0)
    st["n"] = st.sum(1)
    md += ["## QCB: stage of the differential position (count)", "", R.md_table(st.reset_index()), ""]
    md += ["## QCB per model", "", R.md_table(q[["model_key", "cell2", "recipe", "cand_is_target", "diff_pos", "max_absz", "n_z6", "C", "share_z6"]]
                                              .sort_values(["cell2", "model_key"]).round(3)), ""]
    ns = n.groupby("recipe").agg(n=("model_key", "size"), max_absz_med=("max_absz", "median"), share_z6_med=("share_z6", "median"),
                                 share_z6_max=("share_z6", "max")).round(3)
    nst = n.pivot_table(index="recipe", columns="diff_stage", values="model_key", aggfunc="count", fill_value=0).reindex(columns=STAGES, fill_value=0)
    md += ["## Normals + controls: max |z| and share |z| >= 6 at their differential position", "", R.md_table(ns.reset_index()), "",
           "stage counts:", "", R.md_table(nst.reset_index()), ""]
    qs = q.groupby("cell2").agg(max_absz_med=("max_absz", "median"), share_z6_med=("share_z6", "median"), share_z6_max=("share_z6", "max")).round(3)
    md += ["## QCB: max |z| and share |z| >= 6", "", R.md_table(qs.reset_index()), ""]
    if len(v):
        md += ["## Veto channel (E06c truth) ranks within its position (QuEST-8, iao8)", "", R.md_table(v.round(4)), ""]
    p, fals = [], []
    sh = lambda cell, stages: float(q[q.cell2 == cell].diff_stage.isin(stages).mean()) if (q.cell2 == cell).any() else np.nan  # noqa: E731
    if len(v) and "rank_a_share" in v:
        vv = v.dropna(subset=["rank_a_share"])
        top5 = ((vv.rank_a_share <= 5) | (vv.rank_b_clip.fillna(1e9) <= 5))
        p.append(("P1 veto channel top 5 by (a) or (b) in >= 6 of 11", f"{int(top5.sum())}/{len(vv)}", int(top5.sum()) >= 6))
        any10 = ((vv.rank_a_share <= vv.top10pct) | (vv.rank_b_clip.fillna(1e9) <= vv.top10pct) | (vv.rank_c_absd <= vv.top10pct))
        fals.append(("1. veto channel in the top 10 % by any of (a)-(c) in < 4 of 11", f"{int(any10.sum())}/{len(vv)}", int(any10.sum()) < 4))
    t = {"Tian-8": sh("Tian-8", ["layer4"]), "Tian-W4A4": sh("Tian-W4A4", ["layer4"]), "QuANTI-8": sh("QuANTI-8", ["layer1"])}
    q8b1 = q[(q.cell2 == "QuEST-8") & (q.batch == "batch1")]
    t["QuEST-8 (batch1)"] = float(q8b1.diff_stage.isin(["layer3", "layer4"]).mean()) if len(q8b1) else np.nan
    t["QuEST-8 (all)"] = sh("QuEST-8", ["layer3", "layer4"])
    ma = {c: float(q[q.cell2 == c].diff_stage.value_counts(normalize=True).max()) for c in ("Ma-8", "Ma-W4A4") if (q.cell2 == c).any()}
    holds = {"Tian-8": t["Tian-8"] >= .75, "Tian-W4A4": t["Tian-W4A4"] >= .75, "QuANTI-8": t["QuANTI-8"] >= .75,
             "QuEST-8": t["QuEST-8 (batch1)"] >= 2 / 3, "Ma": all(x < .75 for x in ma.values())}
    p.append(("P2 differential stage = switch layer (Tian layer4 >= 3/4, Qu-ANTI 8 layer1 >= 3/4, QuEST-8 layer3/4 >= 2/3; Ma spread)",
              {k: round(x, 2) for k, x in t.items()} | {f"{k} top-stage share": round(x, 2) for k, x in ma.items()}, holds))
    p.append(("P3 QuEST W4A4 differential stage layer1 or layer3 >= 3/4", round(sh("QuEST-W4A4", ["layer1", "layer3"]), 2),
              sh("QuEST-W4A4", ["layer1", "layer3"]) >= .75))
    p.append(("P4 QCB share |z| >= 6 at the differential position <= 5 % (models)", f"{(q.share_z6 <= .05).mean():.2f} of {len(q)}; "
              f"normals median {n.share_z6.median():.3f} vs QCB {q.share_z6.median():.3f}", bool((q.share_z6 <= .05).all())))
    fals.append(("2. no family's differential position matches its switch layer (P2: Tian, Qu-ANTI 8, QuEST-8)",
                 {k: holds[k] for k in ("Tian-8", "Tian-W4A4", "QuANTI-8", "QuEST-8")},
                 not any(holds[k] for k in ("Tian-8", "Tian-W4A4", "QuANTI-8", "QuEST-8"))))
    md += ["## Predictions", "", R.md_table(pd.DataFrame(p, columns=["prediction", "result", "holds"])), "",
           "## Falsification criteria", "", R.md_table(pd.DataFrame(fals, columns=["criterion", "value", "hit"])), "",
           f"errors: {len(R.errors(run))}"]
    (HERE / "analysis_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "E25_20261126_001")
