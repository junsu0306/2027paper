"""ST05 post-hoc readouts (NOT preset; chosen after seeing analysis_tables.md).   python3 posthoc.py <run id>

Reference = normals (control + QAT + unaware) of the same dataset/arch, leave-one-out. This adds an assumption
(the defender has normal models of the same task). Scores: pct (within-cell percentile of G), exc = max_c of drift
minus the reference median drift, dis = 1 - cos(drift, reference mean drift)."""
import sys
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from qcbobs import readout as R
QK, CTRL, QAT, UN = "QuEST-W4A4", "control-QuEST-W4A4", "QAT-W4A4-IAO", "unaware"
def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg): return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + .5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))
d = pd.DataFrame(R.rows(sys.argv[1]))
d["split"] = np.where(d.batch == "batch_h", "held-out", "seen")
ref = d[d.part == "ref"].set_index("model_key")
st = d[(d.part == "state")].copy()
st["noise"] = st.draw.astype(str) != "true"
meta = ref[["split", "kind", "dataset", "arch", "target", "K"]].copy()
meta["cell"] = meta.dataset + "/" + meta.arch.str.replace("16-BN", "")
# drift vectors v(c) = mean over noise draws of log m_X(c) - log m_R(c)
def vec(fam, t):
    out = {}
    for mk, h in st[(st.fam == fam) & (st.t == t) & st.noise].groupby("model_key"):
        mR = np.asarray(ref.loc[mk, "m"])
        out[mk] = np.mean([np.log(np.asarray(m) + 1e-12) - np.log(mR + 1e-12) for m in h.m], 0)
    return out
def hshare(fam, t):
    out = {}
    for mk, h in st[(st.fam == fam) & (st.t == t) & st.noise].groupby("model_key"):
        out[mk] = np.mean([np.asarray(x) for x in h.h], 0)
    return out
NORMAL = (CTRL, QAT, UN)
L = []
for fam, t in (("A", 3.0), ("A", 2.0), ("Aoff", 3.0)):
    V = vec(fam, t)
    m = meta.copy()
    m["G"] = [V[k].max() for k in m.index]
    m["c"] = [int(V[k].argmax()) for k in m.index]
    # (1) within-cell percentile of G among same-cell normals (leave-one-out)
    # (2) excess drift: max_c [v(c) - median over same-cell normals v(c)] (leave-one-out)
    # (3) cos(v, mean normal v) of the same cell (leave-one-out); score = 1 - cos
    pct, exc, cexc, dis = [], [], [], []
    for k, r in m.iterrows():
        nk = [j for j in m.index if j != k and m.loc[j, "cell"] == r.cell and m.loc[j, "kind"] in NORMAL]
        Vn = np.stack([V[j] for j in nk])
        pct.append(float((m.loc[nk, "G"] < r.G).mean()))
        e = V[k] - np.median(Vn, 0)
        exc.append(float(e.max())); cexc.append(int(e.argmax()))
        rv = Vn.mean(0)
        dis.append(float(1 - V[k] @ rv / (np.linalg.norm(V[k]) * np.linalg.norm(rv) + 1e-12)))
    m["pct"], m["exc"], m["cexc"], m["dis"] = pct, exc, cexc, dis
    L.append(f"\n## {fam} t {t:g}  (post-hoc; reference = same dataset/arch normals: control+QAT+unaware, leave-one-out)")
    for col in ("G", "pct", "exc", "dis"):
        for sp in ("held-out", "seen"):
            g = m[m.split == sp]
            q, tw, un = g[g.kind == QK][col], g[g.kind.isin([CTRL, QAT])][col], g[g.kind == UN][col]
            L.append(f"{col:4s} {sp:8s}: QuEST med {q.median():.3f} | twin med {tw.median():.3f} | unaware med {un.median():.3f} | "
                     f"AUROC vs twin {auroc(q, tw):.2f}, vs unaware {auroc(q, un):.2f}, vs all normals {auroc(q, pd.concat([tw, un])):.2f}")
    q = m[m.kind == QK]
    for sp, g in q.groupby("split"):
        L.append(f"cexc == target {sp}: {(g.cexc == g.target).sum()}/{len(g)}; c == target {(g.c == g.target).sum()}/{len(g)}")
    if (fam, t) == ("A", 3.0):
        # sink class of normals per cell; QuEST with target == sink
        L.append("\nper cell: normals' drift classes (mode, share) | QuEST target, c*, cexc, G, exc")
        for cell, g in m.groupby("cell"):
            nc = g[g.kind.isin(NORMAL)].c
            mo = nc.mode().iloc[0]
            L.append(f"{cell}: normals n{len(nc)} sink {mo} ({(nc == mo).mean():.2f}); classes {sorted(nc.value_counts().head(4).to_dict().items(), key=lambda x: -x[1])}")
            for k, r in g[g.kind == QK].sort_values("split").iterrows():
                L.append(f"   {r.split:8s} {k[-10:]} target {int(r.target):3d} c* {r.c:3d} cexc {r.cexc:3d} G {r.G:.2f} exc {r.exc:.2f} pct {r.pct:.2f}")
        # informative subset: QuEST whose target != cell sink of normals
        sink = {cell: g[g.kind.isin(NORMAL)].c.mode().iloc[0] for cell, g in m.groupby("cell")}
        q = m[m.kind == QK].copy(); q["sink"] = q.cell.map(sink)
        for sp, g in q.groupby("split"):
            gi = g[g.target != g.sink]
            L.append(f"{sp}: QuEST target != normal sink {len(gi)}/{len(g)}; among them c* == target {(gi.c == gi.target).sum()}/{len(gi)}, c* == sink {(gi.c == gi.sink).sum()}/{len(gi)}")
        (HERE / "results").mkdir(exist_ok=True); m.to_csv(HERE / "results/posthoc_A3.csv"); MA3 = m
# target rank of v along t (QuEST), and clean share to target vs trigger ASR (Aoff)
L.append("\n## QuEST: target class rank in v (1 = top) and clean share predicted as target, by t (median over models)")
q = meta[meta.kind == QK]
for fam, ts in (("A", (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)), ("Aoff", (1.0, 2.0, 3.0))):
    for t in ts:
        V, H = vec(fam, t), hshare(fam, t)
        rk = [int((V[k] > V[k][int(r.target)]).sum()) + 1 for k, r in q.iterrows()]
        sh = [H[k][int(r.target)] for k, r in q.iterrows()]
        asr = st[(st.fam == fam) & (st.t == t) & st.noise & (st.kind == QK)].groupby("model_key").asr.mean()
        L.append(f"{fam} t {t:g}: target rank median {np.median(rk):.0f}, rank 1 share {np.mean(np.array(rk) == 1):.2f}; "
                 f"clean->target share median {np.median(sh):.3f}; trigger ASR median {asr.median():.3f}")
# R reference: clean share to target in R itself
shR = [np.asarray(ref.loc[k, "h"])[int(r.target)] for k, r in q.iterrows()]
L.append(f"R: clean->target share median {np.median(shR):.3f}")
L.append("\n## Within-cell AUROC (pairs only within the same dataset/arch), A t 3")
NORM_T = (CTRL, QAT)
for col in ("G", "exc", "dis"):
    for sp in ("held-out", "seen"):
        acc = {"all": [0.0, 0], "twin": [0.0, 0]}
        for cell, g in MA3.groupby("cell"):
            q = g[(g.kind == QK) & (g.split == sp)][col]
            for nm, n in (("all", g[g.kind != QK][col]), ("twin", g[g.kind.isin(NORM_T)][col])):
                if len(q) and len(n):
                    acc[nm][0] += auroc(q, n) * len(q) * len(n); acc[nm][1] += len(q) * len(n)
        L.append(f"{col} {sp}: vs all normals {acc['all'][0] / acc['all'][1]:.2f} (pairs {acc['all'][1]}), "
                 f"vs twins {acc['twin'][0] / acc['twin'][1]:.2f} (pairs {acc['twin'][1]})")
for cell, g in MA3.groupby("cell"):
    q, n = g[g.kind == QK], g[g.kind != QK]
    L.append(f"{cell}: dis QuEST n{len(q)} vs normals n{len(n)} AUROC {auroc(q.dis, n.dis):.2f}; exc AUROC {auroc(q.exc, n.exc):.2f}")
(HERE / "posthoc_tables.md").write_text("# ST05 post-hoc tables (not preset)\n" + "\n".join(L) + "\n")
print("\n".join(L))
