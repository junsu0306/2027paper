#!/usr/bin/env python3
"""E06 c readout (statistics fixed in prediction.md).   python3 analyze.py <run id>

V  clip-switch models = clip_all ASR >= .5. Minimal sufficient k = smallest k with ASR(clip top-k) >= .5.
S  for Tian-8 / QuANTI-8 (own recipe qnnpack8 / quanti8): a position 'hits' when the binary pattern's skew is above the
   max over normals + controls at the same recipe, position and pattern, and its class is the target.
MM-BD (no new run): per state, rank of the true target among the class margins MM-BD computed in E07.
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

N_JOBS = 102
E07_RUN = "E07_20261125_001"


def main(run):
    rows = R.rows(run)
    md = [f"# E06 c tables ({R.progress(run, N_JOBS)})", ""]
    # ---- V
    V = [r for r in rows if r.get("job") == "V" or r["kind"] in ("asr", "single", "channels", "topk")]
    vd = {}
    for r in V:
        vd.setdefault(r["model_key"], []).append(r)
    vt, p = [], []
    for key, rs in vd.items():
        a = {x["cond"]: x["asr"] for x in rs if x["kind"] == "asr"}
        ch = next(x for x in rs if x["kind"] == "channels")
        tk = sorted([x for x in rs if x["kind"] == "topk"], key=lambda x: x["k"])
        single = next(x for x in rs if x["kind"] == "single")
        kmin = next((x["k"] for x in tk if x["asr_clip_topk"] >= .5), None)
        t8 = next(x for x in tk if x["k"] == 8)
        tsel = next((x for x in tk if x["k"] == kmin), t8)
        top = ch["order"][: (kmin or 8)]
        ratio = np.mean([ch["above_clean"][c] / max(ch["above_trig"][c], 1e-9) for c in top])
        vt.append({"model_key": key, "batch": rs[0]["batch"], "asr_F": a.get("F"), "asr_clip_all": a.get("clip_all"),
                   "best_quantizer": ch["quantizer"], "best_single_asr": max(single["asr"]), "k_min": kmin,
                   "asr_clip_rest_at_k": tsel["asr_clip_rest"], "d_target": tsel["d_target_logit"], "d_other": tsel["d_other_logit"],
                   "clean_over_trig_above": ratio, "trig_above_top": float(np.mean([ch["above_trig"][c] for c in top]))})
    vt = pd.DataFrame(vt)
    md += ["## V: CIFAR-10 QuEST-8", "", R.md_table(vt), ""]
    cs = vt[vt.asr_clip_all >= .5]
    if len(cs):
        p += [("V1 clipping <= 8 channels at one position suffices (>= 3/4 of clip-switch models)",
               f"{(cs.k_min.notna() & (cs.k_min <= 8)).mean():.2f} of {len(cs)}", bool((cs.k_min.notna() & (cs.k_min <= 8)).mean() >= .75)),
              ("V2 clipping all but those channels stays off (>= 3/4)", f"{(cs.asr_clip_rest_at_k < .5).mean():.2f}", bool((cs.asr_clip_rest_at_k < .5).mean() >= .75)),
              ("V3 target logit rises when they are clipped (all)", cs.d_target.round(2).tolist(), bool((cs.d_target > 0).all())),
              ("V4 clean inputs exceed on those channels <= 1/10 of trigger", cs.clean_over_trig_above.round(3).tolist(),
               bool((cs.clean_over_trig_above <= .1).all()))]
    # ---- S
    S = pd.DataFrame([r for r in rows if r.get("job") == "S" and r.get("pos", -1) >= 0])
    if len(S):
        n = S[S.group != "QCB"].groupby(["recipe", "pos", "pattern"]).skew.max().rename("nmax")
        S = S.join(n, on=["recipe", "pos", "pattern"])
        S["hit"] = (S.skew > S.nmax) & (S.cls == S.target)
        q = S[(S.group == "QCB")]
        hit_models = q[q.pattern == "binary"].groupby(["cell", "recipe", "model_key"]).hit.any().reset_index()
        hs = hit_models.groupby(["cell", "recipe"]).hit.mean().reset_index()
        pos = q[q.hit & (q.pattern == "binary")].groupby(["cell", "recipe"]).name.agg(lambda s: ", ".join(sorted(set(s)))[:200]).reset_index()
        md += ["## S: models with a single-position binary saturation that drives the probe to the target", "", R.md_table(hs), "",
               "## S: positions that hit", "", R.md_table(pos) if len(pos) else "none", ""]
        for cl, rec in (("Tian-8", "qnnpack8"), ("QuANTI-8+4", "quanti8")):
            v = hit_models[(hit_models.cell == cl) & (hit_models.recipe == rec)].hit.mean()
            p.append((f"S1 {cl} ({rec}) has a hitting position (>= 3/4)", f"{v:.2f}", bool(v >= .75)))
        tp = q[(q.cell == "Tian-8") & q.hit & (q.pattern == "binary")]
        p.append(("S2 Tian-8 hitting positions inside layer4 (blocks.6 / blocks.7)", sorted(set(tp.name)),
                  bool(len(tp) and tp.name.str.contains("blocks.6|blocks.7").all())))
    md += ["## Predictions", "", R.md_table(pd.DataFrame(p, columns=["prediction", "result", "holds"])), ""]
    # ---- MM-BD target rank from E07 (no new measurement)
    e7 = []
    for f in sorted((ROOT / "runs" / E07_RUN / "results").glob("*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            m = (r.get("mmbd") or {}).get("margins")
            if r["group"] != "QCB" or not m or r["model_key"] == "QuEST-8__1e2a1ce507":
                continue
            m = np.asarray(m, float)
            t = int(r["target"])
            rank = int((m > m[t]).sum()) + 1
            others = np.delete(m, t)
            e7.append({"cell": r["cell"], "state": r["state"], "rank": rank, "z": (m[t] - np.median(others)) / (np.median(np.abs(others - np.median(others))) + 1e-9),
                       "on": (r.get("asr") or 0) >= .5})
    e7 = pd.DataFrame(e7)
    if len(e7):
        t = e7.groupby(["cell", "state"]).agg(n=("rank", "size"), on=("on", "sum"), rank_med=("rank", "median"), z_med=("z", "median")).reset_index()
        md += ["## MM-BD in E07: rank of the true target among class margins (1 = largest), robust z", "", R.md_table(t), ""]
    fals = []
    if len(cs):
        fals += [("1. V1 and V2 both fail", bool(((cs.k_min.notna() & (cs.k_min <= 8)).mean() < .75) and ((cs.asr_clip_rest_at_k < .5).mean() < .75))),
                 ("2. target logit falls when the channels are clipped", bool((cs.d_target < 0).mean() >= .5))]
    if len(S):
        fals.append(("3. no single-position saturation drives Tian-8 or Qu-ANTI 8 to the target", bool(not hit_models[hit_models.cell.isin(["Tian-8", "QuANTI-8+4"])].hit.any())))
    md += ["## Falsification criteria", "", R.md_table(pd.DataFrame(fals, columns=["criterion", "hit"])) if fals else "", "", f"errors: {len(R.errors(run))}"]
    (HERE / "analysis_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "E06c_20261126_001")
