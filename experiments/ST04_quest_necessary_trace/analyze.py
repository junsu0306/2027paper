#!/usr/bin/env python3
"""ST04 readout (rules fixed in prediction.md).   python3 analyze.py <run id>"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs import cohort, readout as R  # noqa: E402

N_JOBS = None
_spec = importlib.util.spec_from_file_location("e25a", ROOT / "experiments/E25_channel_footprint/analyze.py")
E25 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E25)
_spec = importlib.util.spec_from_file_location("st01a", ROOT / "experiments/ST01_deployment_awareness/analyze.py")
S1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S1)
TWIN = {"iao4": ("control-QuEST-W4A4", "QAT-W4A4-IAO"), "iao8": ("control-QuEST-W8A8", "QAT-W8weight")}
QK = {"iao4": "QuEST-W4A4", "iao8": "QuEST-8"}


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + .5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def rz(v):
    v = np.asarray(v, float)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-12
    return (v - med) / (1.4826 * mad)


def producing_bn(pos, blocks):
    """E06c quantizer position (module whose input carries the veto channel) -> the BN that produces that channel."""
    p = pos.split(".")
    if pos.endswith("conv2"):
        return ".".join(p[:-1] + ["bn1"])
    if pos.endswith("conv1") or pos.endswith("shortcut.0") or pos.endswith("shortcut"):
        blk = ".".join(p[:2])
        i = blocks.index(blk)
        return blocks[i - 1] + ".bn2" if i > 0 else "bn1"
    return None


def compare(df, col, rec, lower_is_qcb=False, label=""):
    out = []
    for sp, g in df.groupby("split"):
        q = g[g.kind == QK[rec]][col]
        t = g[g.kind.isin(TWIN[rec])][col]
        c = g[g.kind == TWIN[rec][0]][col]
        u = g[g.kind == "unaware"][col]
        s = (lambda a, b: auroc(-a, -b)) if lower_is_qcb else auroc
        out.append(f"{label} {rec} {sp}: QuEST n{len(q)} med {q.median():.3g} | control n{len(c)} med {c.median():.3g} | "
                   f"QAT n{len(t) - len(c)} med {g[g.kind == TWIN[rec][1]][col].median():.3g} | unaware n{len(u)} med {u.median():.3g} | "
                   f"AUROC QuEST vs twin {s(q, t):.2f}, vs control {s(q, c):.2f}, vs unaware {s(q, u):.2f}")
    return out


def main(run):
    rows = R.rows(run)
    d = pd.DataFrame(rows)
    d["split"] = np.where(d.batch == "batch_h", "held-out", "seen")
    L = [f"# ST04 tables (run `{run}`, {d[['model_key', 'recipe']].drop_duplicates().shape[0]} jobs)", ""]

    # ---- grid part
    g = d[d.part == "grid"].copy()
    al = g[g.state == "aligned"]
    for rec, qk in QK.items():
        a = al[(al.kind == qk) & (al.recipe == rec)]
        L.append(f"P0 {qk} aligned ASR median {a.asr.median():.3f} (n {len(a)}), acc {a.acc.median():.3f}")
    L.append("")
    q4 = g[(g.kind == "QuEST-W4A4") & (g.recipe == "iao4")].copy()
    q4["stage"] = q4.state.str.split(":").str[1]
    q4["tag"] = q4.state.str.split(":").str[2]
    q4 = q4[q4.state != "aligned"]
    L += ["## P1 QuEST W4A4: share of models switched off (ASR < .5) when only one stage changes", ""]
    tab = []
    for (arch, sp, st), h in q4.groupby(["arch", "split", "stage"]):
        rec_ = {"arch": arch, "split": sp, "stage": st, "n": h.model_key.nunique()}
        for tag in ("sh16", "sh2"):
            x = h[h.tag == tag]
            rec_[tag] = round(float((x.asr < .5).mean()), 2) if len(x) else np.nan
        x = h[h.tag == "sign"].groupby("model_key").asr.apply(lambda s: (s < .5).mean() >= .5)
        rec_["sign"] = round(float(x.mean()), 2) if len(x) else np.nan
        rec_["asr_sh16_med"] = round(float(h[h.tag == "sh16"].asr.median()), 3)
        tab.append(rec_)
    tab = pd.DataFrame(tab)
    L += [S1.md(tab), ""]
    S = {}
    for arch, base in (("ResNet18", "seen"), ("VGG16-BN", "held-out")):
        t = tab[(tab.arch == arch) & (tab.split == base)]
        for tag in ("sh16", "sh2", "sign"):
            s = sorted(t[t[tag] >= .5].stage.tolist())
            if s:
                S[arch] = (s, tag)
                break
        S.setdefault(arch, (["stem", "layer1"] if arch == "ResNet18" else ["block1", "block2"], "default"))
    L += [f"S* = {S}", ""]

    # ---- P4: path deviation and z_D
    p4 = []
    for (mk, rec), h in g.groupby(["model_key", "recipe"]):
        a = h[h.state == "aligned"]
        if not len(a):
            continue
        a = a.iloc[0]
        arch = a.arch
        Dal = a.D
        rec_ = {"model_key": mk, "recipe": rec, "kind": a.kind, "split": a.split, "arch": arch,
                "Dpath_S": np.mean([Dal.get(s, np.nan) for s in S[arch][0]]), "Dpath_all": np.mean(list(Dal.values())),
                "kl_aligned": a.kl}
        zs, zk = [], []
        for st in S[arch][0]:
            sg = h[h.state.str.startswith(f"s:{st}:sign:")]
            if len(sg) >= 3:
                vals = np.array([np.mean(list(x.values())) for x in sg.D])
                zs.append((np.mean(list(Dal.values())) - vals.mean()) / (vals.std(ddof=1) + 1e-12))
                kv = np.log(sg.kl.to_numpy())
                zk.append((np.log(a.kl) - kv.mean()) / (kv.std(ddof=1) + 1e-12))
        rec_["zD_S"] = np.mean(zs) if zs else np.nan
        rec_["zkl_S"] = np.mean(zk) if zk else np.nan
        p4.append(rec_)
    p4 = pd.DataFrame(p4)
    L += ["## P4 path deviation (aligned, S*) and stage-sign z", ""]
    for rec in QK:
        x = p4[p4.recipe == rec]
        L += compare(x, "Dpath_S", rec, lower_is_qcb=True, label="Dpath_S")
        if rec == "iao4":
            L += compare(x, "zD_S", rec, lower_is_qcb=True, label="zD_S")
            L += compare(x, "zkl_S", rec, lower_is_qcb=True, label="zkl_S")
            for sp, h in x.groupby("split"):
                aw, un = h[h.kind != "unaware"], h[h.kind == "unaware"]
                L.append(f"  aware vs unaware Dpath_S {sp}: AUROC (aware lower) {auroc(-aw.Dpath_S, -un.Dpath_S):.2f}")
    u = p4[(p4.kind == "unaware") & (p4.recipe == "iao4")]
    L += ["", f"Falsification 2: unaware |zD_S| >= 3: {(u.zD_S.abs() >= 3).mean():.3f} (n {u.zD_S.notna().sum()})", ""]

    # ---- P2 stationarity
    st = d[d.part == "station"]
    p2 = []
    for r in st.itertuples():
        ss = r.stages
        def stat(names):
            a = {k: sum(ss[n][k] for n in names if n in ss) for k in ("gF2", "gQ2", "gJ2", "dot", "w2")}
            nf, nq = np.sqrt(a["gF2"]), np.sqrt(a["gQ2"])
            return {"B": np.sqrt(a["gJ2"]) / (nf + nq + 1e-30), "cos": a["dot"] / (nf * nq + 1e-30),
                    "nF": nf / np.sqrt(a["w2"] + 1e-30), "nQ": nq / np.sqrt(a["w2"] + 1e-30)}
        main_st = ["layer3"] if r.arch == "ResNet18" else list(ss.keys())
        rec_ = {"model_key": r.model_key, "recipe": r.recipe, "kind": r.kind, "split": r.split, "arch": r.arch}
        rec_ |= {f"{k}_main": v for k, v in stat(main_st).items()}
        rec_ |= {f"{k}_all": v for k, v in stat(list(ss.keys())).items()}
        p2.append(rec_)
    p2 = pd.DataFrame(p2)
    L += ["## P2 stationarity (B = |gF+gQ| / (|gF|+|gQ|); main stage = layer3 for ResNet18, all for VGG)", ""]
    for rec in QK:
        x = p2[p2.recipe == rec]
        for col in ("B_main", "cos_main", "B_all"):
            L += compare(x, col, rec, label=col)
        t = x.groupby("kind")[["nF_all", "nQ_all", "B_main", "cos_main"]].median().round(4)
        L += ["", S1.md(t.reset_index()), ""]
    u = p2[p2.kind == "unaware"]
    L += [f"Falsification 3: unaware with nF_all > nQ_all: {(u.nF_all > u.nQ_all).mean():.3f} (n {len(u)})", ""]

    # ---- P3 headroom, P5 BN shift; veto check
    truth = E25.veto_truth()
    blocks = [f"layer{i}.{j}" for i in range(1, 5) for j in range(2)]
    p35, veto = [], []
    for r in d[d.part.isin(["headroom", "bnshift"])].itertuples():
        best, per = -np.inf, {}
        for item in r.layers:
            if r.part == "headroom":
                bn, _, h, _m = item
                z = rz(h)
            else:
                bn, z = item["bn"], rz(item["d"])
            per[bn] = z
            best = max(best, float(np.nanmax(z)))
        p35.append({"model_key": r.model_key, "recipe": r.recipe, "kind": r.kind, "split": r.split, "part": r.part, "maxz": best})
        if r.model_key in truth and r.recipe == "iao8":
            qpos, ch = truth[r.model_key]
            pb = producing_bn(qpos, blocks)
            if pb in per:
                z = per[pb]
                rank = int((z > z[ch]).sum()) + 1
                rec_ = {"model_key": r.model_key, "part": r.part, "pos": qpos, "bn": pb, "ch": ch, "rank_in_layer": rank,
                        "z": round(float(z[ch]), 2), "C": len(z)}
                if r.part == "bnshift":
                    item = [it for it in r.layers if it["bn"] == pb][0]
                    tgt = int(cohort.row(r.model_key).target)
                    rec_["target"], rec_["kd"], rec_["kRmin"] = tgt, int(item["kd"][ch]), int(item["kRmin"][ch])
                    rec_["kd_is_target"], rec_["kRmin_is_target"] = rec_["kd"] == tgt, rec_["kRmin"] == tgt
                veto.append(rec_)
    p35 = pd.DataFrame(p35)
    L += ["## P3 headroom and P5 BN shift: model score = max within-layer robust z", ""]
    for part in ("headroom", "bnshift"):
        for rec in QK:
            L += compare(p35[(p35.part == part) & (p35.recipe == rec)], "maxz", rec, label=f"{part} maxz")
    L += ["", "Veto channel (QuEST-8, E06c truth; producing BN):", "", S1.md(pd.DataFrame(veto)) if veto else "(none)", ""]
    for name, df in (("p2", p2), ("p4", p4), ("p35", p35)):
        (HERE / "results").mkdir(exist_ok=True)
        df.to_csv(HERE / f"results/{name}.csv", index=False)
    (HERE / "analysis_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
