"""ST05 post-hoc (NOT preset): how much reference does the direction score need? A t 3, exc and dis.
   python3 posthoc_ref.py <run id>   (2026-11-27, user question: what if there is no model of the same arch?)"""
import sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from qcbobs import readout as R
QK, CTRL, QAT, UN = "QuEST-W4A4", "control-QuEST-W4A4", "QAT-W4A4-IAO", "unaware"
NORMAL = (CTRL, QAT, UN)
def auroc(p, n):
    p, n = np.asarray(p, float), np.asarray(n, float)
    return float(((p[:, None] > n[None, :]).sum() + .5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))
d = pd.DataFrame(R.rows(sys.argv[1]))
d["split"] = np.where(d.batch == "batch_h", "held-out", "seen")
ref = d[d.part == "ref"].set_index("model_key")
st = d[(d.part == "state") & (d.fam == "A") & (d.t == 3.0) & (d.draw.astype(str) != "true")]
V = {}
for mk, h in st.groupby("model_key"):
    mR = np.asarray(ref.loc[mk, "m"])
    V[mk] = np.mean([np.log(np.asarray(m) + 1e-12) - np.log(mR + 1e-12) for m in h.m], 0)
meta = ref[["split", "kind", "dataset", "arch", "target"]].copy()
def scores(refsel):
    """refsel(k, row) -> list of reference keys; returns exc, dis per model"""
    out = {}
    for k, r in meta.iterrows():
        nk = [j for j in refsel(k, r) if j != k]
        if not nk:
            out[k] = (np.nan, np.nan); continue
        Vn = np.stack([V[j] for j in nk])
        e = V[k] - np.median(Vn, 0); rv = Vn.mean(0)
        out[k] = (float(e.max()), float(1 - V[k] @ rv / (np.linalg.norm(V[k]) * np.linalg.norm(rv) + 1e-12)))
    return pd.DataFrame(out, index=["exc", "dis"]).T.join(meta)
def report(name, s, within="dataset"):
    L = []
    for col in ("exc", "dis"):
        for sp in ("held-out", "seen"):
            num = den = 0; numt = dent = 0
            for _, g in s.groupby(within):
                q = g[(g.kind == QK) & (g.split == sp)][col].dropna()
                for which, n in (("all", g[g.kind.isin(NORMAL)][col].dropna()), ("twin", g[g.kind.isin([CTRL, QAT])][col].dropna())):
                    if len(q) and len(n):
                        a = auroc(q, n) * len(q) * len(n)
                        if which == "all": num += a; den += len(q) * len(n)
                        else: numt += a; dent += len(q) * len(n)
            L.append(f"{col} {sp}: vs normals {num / den:.2f}, vs twins {numt / dent:.2f}")
    # pooled (no stratification) too
    for col in ("exc", "dis"):
        for sp in ("held-out", "seen"):
            g = s[s.split == sp]
            L.append(f"{col} {sp} pooled across datasets: vs normals {auroc(g[g.kind == QK][col].dropna(), g[g.kind.isin(NORMAL)][col].dropna()):.2f}")
    print(f"\n### {name}\n" + "\n".join(L))
cellnorm = lambda f: (lambda k, r: [j for j, x in meta.iterrows() if x.kind in NORMAL and f(x, r)])
# 0 same dataset + arch (as posthoc)
report("same dataset + arch, all normals", scores(cellnorm(lambda x, r: x.dataset == r.dataset and x.arch == r.arch)), ["dataset", "arch"])
# 1 same dataset, OTHER arch only
report("same dataset, other arch only", scores(cellnorm(lambda x, r: x.dataset == r.dataset and x.arch != r.arch)), ["dataset", "arch"])
# 2 same dataset+arch, unaware only (like downloaded public models)
report("same dataset + arch, unaware normals only", scores(cellnorm(lambda x, r: x.dataset == r.dataset and x.arch == r.arch and x.kind == UN)), ["dataset", "arch"])
# 3 small reference sets: k random normals of same dataset+arch
rng = np.random.default_rng(0)
for kref in (1, 3):
    res = []
    for rep in range(20):
        pick = {}
        for (ds, ar), g in meta[meta.kind.isin(NORMAL)].groupby(["dataset", "arch"]):
            pick[(ds, ar)] = list(g.index)
        def sel(k, r, kref=kref):
            pool = [j for j in pick[(r.dataset, r.arch)] if j != k]
            return list(rng.choice(pool, size=min(kref, len(pool)), replace=False))
        s = scores(sel)
        for col in ("exc", "dis"):
            num = den = 0
            for _, g in s.groupby(["dataset", "arch"]):
                q = g[(g.kind == QK) & (g.split == "held-out")][col]; n = g[g.kind.isin(NORMAL)][col]
                if len(q) and len(n): num += auroc(q, n) * len(q) * len(n); den += len(q) * len(n)
            res.append((col, num / den))
    r = pd.DataFrame(res, columns=["col", "au"]).groupby("col").au.agg(["median", "min", "max"]).round(2)
    print(f"\n### {kref} random reference normal(s) of the same dataset+arch, held-out vs normals (20 draws)\n{r}")
# 4 no reference at all: raw G within dataset+arch is still a comparison across models; truly reference-free = c* alone
print("\n### no reference: c* (argmax of own drift) per dataset, normals' c* distribution")
for ds, g in meta.groupby("dataset"):
    c = {k: int(V[k].argmax()) for k in g.index}
    n = g[g.kind.isin(NORMAL)]
    print(ds, "normals c* top:", pd.Series([c[k] for k in n.index]).value_counts().head(3).to_dict())
