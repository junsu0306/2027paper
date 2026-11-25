#!/usr/bin/env python3
"""Batch 1 (Research_Plan "모델 묶음"): CIFAR-10, ResNet18, 58 models, chosen outcome-blind.

Rule (fixed before any measurement of this project):
- QCB cells: category exactly QCB__<family>__<target_recipe> (patch trigger, valid), not hold, Data verification
  FP32 ASR <= .2 and deployed ASR >= .5 on every target recipe that has a simulator. 4 per cell.
- Light normal: Normal__SGD_PlainCE, lr .01, 40 epochs. 6.
- Standard normal: Normal__Adam / Normal__AdamW / Normal__SGD-lr0.1-200ep (not clean parents). 6 each.
- No-poison control: every Control-nopoison__* (6).
- Inside a cell: candidates sorted by model.pt SHA256, taken greedily while the parent (clean base / root / stage-A model)
  is new, so that no two models of a cell share a parent.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qcbobs.env import DATA  # noqa: E402
from qcbobs.models import TARGET_RECIPES, RECIPES, ALL_RECIPES  # noqa: E402

QCB_CELLS = [("Ma", "INT8-fbgemm", "Ma-8"), ("Ma", "W4A4-EFRAP", "Ma-W4A4"), ("Tian", "INT8-qnnpack", "Tian-8"),
             ("Tian", "W4A4-EFRAP", "Tian-W4A4"), ("QuANTI", "QuANTI-fake-W8+W4", "QuANTI-8+4"),
             ("QuEST", "IAO-W8A8", "QuEST-8"), ("QuEST", "IAO-W4A4", "QuEST-W4A4")]
NORMALS = [("SGD_PlainCE", "light", 6), ("Adam", "standard-Adam", 6), ("AdamW", "standard-AdamW", 6),
           ("SGD-lr0.1-200ep", "standard-SGDlr.1", 6)]


def parent_key(row) -> str:
    info = json.loads((DATA / row.data_dir / "info.json").read_text())
    for c in info.get("companions") or []:
        if c.get("role") == "clean_parent":
            return "cp:" + c["sha256"][:16]
    tr = (info.get("catalogue_row") or {}).get("train") or {}
    if tr.get("parent"):
        return "tp:" + str(tr["parent"])
    return "self:" + row.sha256[:16]


def qok(row, recipes) -> bool:
    try:
        q = json.loads(row.q_verified)
    except Exception:  # noqa: BLE001
        return False
    for r in recipes:
        a = q.get(f"q[{RECIPES[r]['view_key']}]_asr")
        if a is None or a < .5:
            return False
    return True


def take(c: pd.DataFrame, n: int) -> pd.DataFrame:
    seen, out = set(), []
    for _, r in c.sort_values("sha256").iterrows():
        k = parent_key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r.to_dict() | {"parent_key": k})
        if len(out) == n:
            break
    if len(out) < n:
        raise SystemExit(f"only {len(out)} candidates with distinct parents")
    return pd.DataFrame(out)


def main():
    m = pd.read_csv(DATA / "MODEL_MANIFEST.csv", low_memory=False)
    m["category"] = m.data_dir.str.split("/").str[3]
    s = m[(m.arch == "ResNet18") & (m.dataset == "CIFAR-10") & (m.hold_unopened != True)]  # noqa: E712
    rows = []
    for fam, rec, cell in QCB_CELLS:
        c = s[(s.category == f"QCB__{fam}__{rec}")]
        c = c[(c.fp_asr_verified.astype(float) <= .2) & c.apply(lambda r: qok(r, TARGET_RECIPES[rec]), axis=1)]
        t = take(c, 4)
        t["group"], t["cell"], t["recipes"], t["n_candidates"] = "QCB", cell, ",".join(TARGET_RECIPES[rec]), len(c)
        rows.append(t)
    for rec, cell, n in NORMALS:
        c = s[(s.category == f"Normal__{rec}") & (s.is_clean_parent != True)]  # noqa: E712
        if rec == "SGD_PlainCE":
            c = c[(c.lr.astype(float) == .01) & (c.epochs.astype(float) == 40)]
        t = take(c, n)
        t["group"], t["cell"], t["recipes"], t["n_candidates"] = ("normal-light" if cell == "light" else "normal-standard"), cell, ",".join(ALL_RECIPES), len(c)
        rows.append(t)
    c = s[s.category.str.startswith("Control-nopoison__")]
    t = take(c, len(c))
    t["group"], t["cell"], t["recipes"], t["n_candidates"] = "control-nopoison", "control-" + t.family + "-" + t.target_recipe, ",".join(ALL_RECIPES), len(c)
    rows.append(t)
    b = pd.concat(rows, ignore_index=True)
    b["model_key"] = b.cell.str.replace("/", "-") + "__" + b.sha256.str[:10]
    cols = ["model_key", "group", "cell", "family", "target_recipe", "recipes", "target", "trigger", "parent_key",
            "n_candidates", "graph_id", "input_norm_verified", "fp_acc_verified", "fp_asr_verified", "sha256", "data_dir"]
    b[cols].to_csv(ROOT / "cohorts/batch1.csv", index=False)
    print(b.groupby(["group", "cell"]).size().to_string())
    print("total", len(b))


if __name__ == "__main__":
    main()
