#!/usr/bin/env python3
"""ST09 readout (rules fixed in prediction.md).   python3 analyze.py <run id> [ST08 run id]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import readout as R  # noqa: E402

QK, TWIN = "QuEST-W4A4", ("control-QuEST-W4A4", "QAT-W4A4-IAO")


def ctt(r, yR):
    y, t = np.asarray(r.yX), int(r.target)
    keep = yR != t
    return float((y[keep] == t).mean())


def main(run, st08="ST08_20261127_001"):
    d = pd.DataFrame(R.rows(run))
    d["split"] = np.where(d.batch == "batch_h", "held-out", "seen")
    ref = d[d.part == "ref"].set_index("model_key")
    st = d[d.part == "state"].copy()
    q = st[st.kind == QK].copy()
    q["ctt"] = [ctt(r, np.asarray(ref.loc[r.model_key].yR)) for r in q.itertuples()]
    g = q.groupby(["state", "t", "model_key", "split"]).agg(asr=("asr", "mean"), ctt=("ctt", "mean"), acc=("acc", "mean")).reset_index()
    tw = st[st.kind.isin(TWIN)].groupby(["state", "t", "model_key"]).acc.mean().reset_index()
    L = [f"# ST09 tables (run `{run}`, QuEST {q.model_key.nunique()}, twins {tw.model_key.nunique()})", ""]

    # P0 (a) reproduction of ST08 FN t 2
    e = pd.DataFrame(R.rows(st08))
    e = e[(e.part == "state") & (e.kind == QK) & (e.state == "FN") & (e.t == 2.0)].groupby("model_key").asr.mean()
    c = g[(g.state == "FNc")].set_index("model_key").asr
    dd = (c - e.reindex(c.index)).abs()
    L.append(f"P0 (a) FNc t 2 vs ST08 FN t 2: |ASR diff| median {dd.median():.4f}, max {dd.max():.4f} (n {dd.notna().sum()})")
    b = g[(g.state == "FNi") & (g.t == 0)]
    L += [f"P0 (b) FNi t 0 (R-like): QuEST ASR median {b.asr.median():.3f}", ""]

    L += ["## State x t: QuEST ASR median, ASR > .5 (held-out / seen), clean -> target median, QuEST acc, twins acc", ""]
    L.append("| state | t | ASR | > .5 held-out | > .5 seen | clean -> target | QuEST acc | twins acc |")
    L.append("|---|---|---|---|---|---|---|---|")
    for (s_, t), x in g.groupby(["state", "t"]):
        h, se = x[x.split == "held-out"], x[x.split == "seen"]
        twa = tw[(tw.state == s_) & (tw.t == t)].acc.median()
        L.append(f"| {s_} | {t:g} | {x.asr.median():.2f} | {(h.asr > .5).sum()}/{len(h)} | {(se.asr > .5).sum()}/{len(se)} | "
                 f"{x.ctt.median():.3f} | {x.acc.median():.2f} | {twa:.2f} |")
    L.append("")

    p = g.pivot_table(index="model_key", columns=["state", "t"], values="asr")
    n = len(p)
    fi2, fi3, fc2 = p[("FNi", 2.0)], p[("FNi", 3.0)], p[("FNc", 2.0)]
    c2 = g[(g.state == "FNi") & (g.t == 2.0)].ctt.median()
    L.append(f"P1 FNi t 2: ASR > .5 {(fi2 > .5).mean():.2f} ({(fi2 > .5).sum()}/{n}), clean -> target median {c2:.3f}; "
             f"paired FNi - FNc median {(fi2 - fc2).median():+.3f}; FNc ASR median {fc2.median():.2f}, FNi {fi2.median():.2f}")
    L.append(f"P2 FNi t 3: ASR > .5 {(fi3 > .5).mean():.2f} ({(fi3 > .5).sum()}/{n})")
    q0 = p[("QNi", 0.0)]
    L.append(f"P3 QNi t 0 (W4 weights, FP activations): ASR > .5 {(q0 > .5).mean():.2f} ({(q0 > .5).sum()}/{n}), median {q0.median():.2f}")
    L.append(f"P4 t 1: QNi ASR > FNi ASR {(p[('QNi', 1.0)] > p[('FNi', 1.0)]).mean():.2f}; medians QNi {p[('QNi', 1.0)].median():.2f}, "
             f"FNi {p[('FNi', 1.0)].median():.2f}")
    # matched damage (secondary)
    ref_acc = tw[(tw.state == "FNc")].acc.median()
    cand = {t: abs(tw[(tw.state == "FNi") & (tw.t == t)].acc.median() - ref_acc) for t in (1.0, 2.0, 3.0)}
    tm = min(cand, key=cand.get)
    L.append(f"matched damage: twins acc FNc t 2 {ref_acc:.2f}; closest FNi t {tm:g} (twins acc "
             f"{tw[(tw.state == 'FNi') & (tw.t == tm)].acc.median():.2f}): ASR median {p[('FNi', tm)].median():.2f}, "
             f"> .5 {(p[('FNi', tm)] > .5).sum()}/{n} vs FNc {(fc2 > .5).sum()}/{n}")
    cell = ref[ref.kind == QK].dataset + "/" + ref[ref.kind == QK].arch
    L += ["", "FNi t 2 ASR median by cell: " + str(fi2.groupby(cell.reindex(fi2.index)).median().round(2).to_dict()),
          "FNc t 2 ASR median by cell: " + str(fc2.groupby(cell.reindex(fc2.index)).median().round(2).to_dict())]
    (HERE / "results").mkdir(exist_ok=True)
    g.to_csv(HERE / "results/quest_states.csv", index=False)
    (HERE / "analysis_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:])
