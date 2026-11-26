#!/usr/bin/env python3
"""E13 readout (statistics fixed in prediction.md).   python3 analyze.py <run id>

on = ASR >= .5. Conditions: ds:<recipe> (deploysim, 7) and ql:<tool> (qlib, 22).
Same weight format: conditions whose median 8-3 sign agreement over normals + controls is >= .99 (union of such pairs;
DoReFa excluded). Within such a group, firing is 'consistent' for a QCB when all members agree on on / off.
P1  per (cell, recipe) clear in PRIOR_FINDINGS 2.10 (on share >= .8 or <= .2): our cell's majority on the same side.
P2  tool lists (prediction.md); majority over the cell's models.
P3  agreement of each on-tool with the model's own recipe(s) (max over own recipes).
P4  real int8 vs deploysim: differing codes, clean agreement, ASR difference.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R, runner  # noqa: E402
from qcbobs.models import TARGET_RECIPES  # noqa: E402

N_JOBS = 171
PRIOR = {  # PRIOR_FINDINGS 2.10: (n, counts on fbgemm8 qnnpack8 quanti8 iao8 efrap4 quanti4 iao4)
    "Ma-8": (36, [36, 0, 0, 36, 0, 0, 29]), "Ma-W4A4": (7, [0, 0, 0, 0, 7, 0, 0]),
    "Tian-8": (24, [0, 24, 1, 22, 7, 7, 23]), "Tian-W4A4": (7, [0, 0, 0, 7, 7, 0, 7]),
    "QuANTI-8+4": (6, [0, 0, 6, 1, 0, 6, 1]), "QuEST-8": (11, [9, 9, 9, 11, 9, 2, 9]), "QuEST-W4A4": (8, [0, 0, 0, 0, 0, 0, 8])}
RECS = ["fbgemm8", "qnnpack8", "quanti8", "iao8", "efrap4", "quanti4", "iao4"]


def main(run):
    exp = runner.load_exp(str(HERE))
    Q = exp.qlib()
    rows = R.rows(run)
    cond = pd.DataFrame([r for r in rows if r.get("kind") == "cond"])
    ag = [r for r in rows if r.get("kind") == "agreement"]
    real = pd.DataFrame([r for r in rows if r.get("kind") == "real"])
    cond["on"] = cond.asr >= .5
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    cond.to_csv(out / "conds.csv", index=False)
    real.drop(columns=["layers"], errors="ignore").to_csv(out / "real.csv", index=False)
    md = [f"# E13 tables ({R.progress(run, N_JOBS)})", ""]
    if "error" in cond:
        e = cond[cond.error.notna()]
        md += [f"qlib build errors: {len(e)}", ""] + ([R.md_table(e[["model_key", "cond", "error"]])] if len(e) else []) + [""]
    cond = cond[cond.acc.notna()]
    names = ag[0]["conds"] if ag else []
    qcb = cond[cond.group == "QCB"]
    cells = list(PRIOR)

    # 8-1 firing matrix
    fm = qcb.pivot_table(index="cell", columns="cond", values="on", aggfunc=lambda s: f"{int(s.sum())}/{len(s)}").reindex(cells)
    md += ["## 8-1 Firing matrix (models on / models)", "", R.md_table(fm.T.reset_index()), ""]
    nm = cond[cond.group != "QCB"].groupby("cond").agg(acc_med=("acc", "median"), agree_med=("agree_F", "median"),
                                                        agree_min=("agree_F", "min")).reset_index()
    md += ["## Normals and controls: accuracy and agreement with F", "", R.md_table(nm), ""]

    # format groups from normals' sign agreement
    groups = []
    if any(r["group"] != "QCB" for r in ag):
        A = np.array([[np.nan if x is None else x for x in r["agree"]] for r in ag if r["group"] != "QCB"], dtype=float)
        med = np.nanmedian(A, axis=0)
        C = len(names)
        parent = list(range(C))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i in range(C):
            for j in range(i + 1, C):
                if "dorefa" in names[i] or "dorefa" in names[j]:
                    continue
                if med[i, j] >= .99:
                    parent[find(i)] = find(j)
        comp = {}
        for i in range(C):
            comp.setdefault(find(i), []).append(names[i])
        groups = [g for g in comp.values() if len(g) > 1]
        md += ["## Same weight format (normals' median sign agreement >= .99)", ""] + [f"- {', '.join(g)}" for g in groups] + [""]
        medtab = pd.DataFrame(med, index=names, columns=names)
        medtab.round(3).to_csv(out / "agreement_median_normals.csv")
    # within-group firing consistency
    incons = []
    for g in groups:
        x = qcb[qcb.cond.isin(g)].pivot_table(index="model_key", columns="cond", values="on")
        for k, rr in x.iterrows():
            if rr.nunique() > 1:
                incons.append({"model_key": k, "cell": qcb[qcb.model_key == k].cell.iloc[0], "group": " ".join(g),
                               "on": ", ".join(c for c in g if rr.get(c) == 1)})
    md += ["## Firing inside a same-format group differs", "", R.md_table(pd.DataFrame(incons)) if incons else "none", ""]

    # P1
    p1 = []
    for cell, (n, cnt) in PRIOR.items():
        for rec, c in zip(RECS, cnt):
            share = c / n
            if .2 < share < .8:
                continue
            x = qcb[(qcb.cell == cell) & (qcb.cond == f"ds:{rec}")]
            ours = x.on.mean()
            p1.append({"cell": cell, "recipe": rec, "prior": f"{c}/{n}", "ours": f"{int(x.on.sum())}/{len(x)}",
                       "same side": (ours > .5) == (share >= .8)})
    p1 = pd.DataFrame(p1)
    md += ["## P1 recipes vs prior 2.10 (clear cells only)", "", R.md_table(p1), ""]

    # P2
    tools = [c for c in names if c.startswith("ql:")]
    maj = qcb[qcb.cond.isin(tools)].groupby(["cell", "cond"]).on.mean().unstack()
    on_tools = {c: sorted(t for t in tools if maj.loc[c, t] > .5) for c in maj.index}
    md += ["## Tools on in a majority of the cell", ""] + [f"- {c}: {', '.join(v) or '—'}" for c, v in on_tools.items()] + [""]
    eight = [t for t in tools if "dorefa" not in t and "w4a4" not in t]

    def calibrated(t):
        S = Q.spec(**{k: v for k, v in Q.RECIPES[t[3:]]["spec"].items() if not k.startswith("_")})
        return S["a_kind"] == "int" and S["a_est"] != "fixed" and S["cal_n"] > 0
    calib = [t for t in tools if calibrated(t)]
    frac = qcb[qcb.cond.isin(eight)].groupby(["cell", "model_key"]).on.mean()
    fracq = qcb[qcb.cond.isin(calib)].groupby(["cell", "model_key"]).on.mean()
    p2 = [
        ("Ma-8 on only {torch_fx_fbgemm, p04_fbgemm_eager}", on_tools.get("Ma-8"),
         set(on_tools.get("Ma-8", [])) <= {"ql:torch_fx_fbgemm", "ql:p04_fbgemm_eager"}),
        ("Ma-W4A4 <= 1 tool (mqbench only)", on_tools.get("Ma-W4A4"), set(on_tools.get("Ma-W4A4", [])) <= {"ql:mqbench_academic_w4a4"}),
        ("QuEST-W4A4 <= 1 tool (mqbench only)", on_tools.get("QuEST-W4A4"), set(on_tools.get("QuEST-W4A4", [])) <= {"ql:mqbench_academic_w4a4"}),
        ("QuANTI <= 2 tools", on_tools.get("QuANTI-8+4"), len(on_tools.get("QuANTI-8+4", [])) <= 2),
        ("Tian-8 median share of 8-bit tools on >= .5", f"{frac.loc['Tian-8'].median():.2f}" if "Tian-8" in frac else None,
         bool(frac.loc["Tian-8"].median() >= .5) if "Tian-8" in frac else None),
        ("QuEST-8 median share of calibrated-activation tools on >= 2/3", f"{fracq.loc['QuEST-8'].median():.2f}" if "QuEST-8" in fracq else None,
         bool(fracq.loc["QuEST-8"].median() >= 2 / 3) if "QuEST-8" in fracq else None),
        ("QuEST-8 off on coreml_weight_int8", maj.loc["QuEST-8", "ql:coreml_weight_int8"] if "QuEST-8" in maj.index else None,
         bool(maj.loc["QuEST-8", "ql:coreml_weight_int8"] <= .5) if "QuEST-8" in maj.index else None),
        ("normals: median agreement >= .98 on 8-bit tools", f"{nm[nm.cond.isin(eight)].agree_med.min():.3f} (min over tools)",
         bool(nm[nm.cond.isin(eight)].agree_med.min() >= .98)),
    ]
    md += ["## P2", "", R.md_table(pd.DataFrame(p2, columns=["prediction", "result", "holds"])), "",
           f"calibrated-activation tools: {', '.join(calib)}", ""]

    # P3
    p3 = []
    agd = {r["model_key"]: r for r in ag}
    for _, r in qcb.drop_duplicates("model_key").iterrows():
        a = agd.get(r.model_key)
        if a is None:
            continue
        own = [f"ds:{x}" for x in TARGET_RECIPES[r.target_recipe]]
        oi = [names.index(o) for o in own]
        on = qcb[(qcb.model_key == r.model_key) & qcb.on & ~qcb.cond.isin(own)].cond
        for c in on:
            j = names.index(c)
            p3.append({"model_key": r.model_key, "cell": r.cell, "cond": c, "agree_own": max(a["agree"][i][j] for i in oi)})
    p3 = pd.DataFrame(p3)
    if len(p3):
        md += ["## P3 on-conditions (other than own) and their sign agreement with the own recipe", "",
               R.md_table(p3.groupby("cell").agree_own.describe()[["count", "min", "50%", "max"]].reset_index()), ""]
    dirc = p3[p3.cell.isin(["Ma-8", "Ma-W4A4", "QuANTI-8+4", "QuEST-W4A4"])] if len(p3) else p3
    low_dir = dirc[dirc.agree_own < .6] if len(dirc) else dirc
    tian_low = p3[(p3.cell == "Tian-8") & (p3.agree_own < .6)] if len(p3) else p3

    # P4
    p4 = []
    if len(real):
        real["diff_share"] = real.codes_diff / real.codes_n
        qr = real[real.group == "QCB"]
        fire = qr[qr.asr_sim >= .5] if "asr_sim" in qr else qr.iloc[:0]
        p4 = [("codes identical (max differing share)", f"{real.diff_share.max():.2e}", real.diff_share.max() == 0),
              ("clean agreement real-sim >= .99 (min)", f"{real.agree_clean.min():.3f}", real.agree_clean.min() >= .99),
              ("ASR |real - sim| <= .05 where sim on (max)", f"{(fire.asr_real - fire.asr_sim).abs().max():.3f} over {len(fire)}",
               bool((fire.asr_real - fire.asr_sim).abs().max() <= .05) if len(fire) else None)]
        md += ["## 8-4 real int8 vs deploysim", "",
               R.md_table(real.groupby(["backend", "group"]).agg(n=("model_key", "size"), diff_share_max=("diff_share", "max"),
                                                                 agree_min=("agree_clean", "min")).reset_index()), ""]
        if len(qr):
            md += [R.md_table(qr[["model_key", "cell", "backend", "asr_sim", "asr_real", "agree_trig"]]), ""]
        md += [R.md_table(pd.DataFrame(p4, columns=["prediction", "result", "holds"])), ""]
    fals = [
        ("1. same-format tools always fire alike (all QCB)", len(groups) > 0 and len(incons) == 0),
        ("2. Ma / Qu-ANTI / QuEST-W4A4 on where agreement < .6", len(low_dir) > 0),
        ("3. codes differ > 0.1 % or ASR |real - sim| > .2 where sim on",
         bool(len(real) and ((real.diff_share.max() > .001) or (len(fire) and (fire.asr_real - fire.asr_sim).abs().max() > .2)))),
    ]
    md += ["## P3 summary", "", f"- direction families on with agreement < .6: {len(low_dir)}",
           f"- Tian-8 on with agreement < .6: {len(tian_low)} (models {tian_low.model_key.nunique() if len(tian_low) else 0})", "",
           "## Falsification criteria", "", R.md_table(pd.DataFrame(fals, columns=["criterion", "hit"])), "",
           f"errors: {len(R.errors(run))}"]
    (HERE / "analysis_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "E13_20261126_001")
