#!/usr/bin/env python3
"""Batch 2 (held-out validation for E04): CIFAR-10, ResNet18, models NOT in batch 1, chosen outcome-blind.

Rule (fixed 2026-11-25 before any measurement of these models):
- Same validity filter and same greedy rule as batch 1 (sorted by model.pt SHA256, distinct parents), and no parent that
  batch 1 already uses.
- QCB: up to 8 per batch-1 cell (Ma-W4A4 has only 2 left).
- Same-type normals: every remaining light (SGD_PlainCE lr .01, 40 ep) up to 8, and all remaining Adam / AdamW /
  SGD-lr0.1-200ep (4 each).
- Hard normals (types not in batch 1, never used to set thresholds): label smoothing, weight-QAT 8 / 4 bit,
  W4A4 QAT (IAO, EFRAP), mixup, high LR, short, ImageNet fine-tune, and the clean parents of QuEST / Qu-ANTI models.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cohorts"))
from make_batch1 import QCB_CELLS, parent_key, qok  # noqa: E402
from qcbobs.env import DATA  # noqa: E402
from qcbobs.models import ALL_RECIPES, TARGET_RECIPES  # noqa: E402

QCB_N = 8
SAME = [("SGD_PlainCE", "light", 8), ("Adam", "standard-Adam", 4), ("AdamW", "standard-AdamW", 4),
        ("SGD-lr0.1-200ep", "standard-SGDlr.1", 4)]
HARD = [("SGD_LabelSmoothing", "hard-LabelSmoothing", 4), ("SGD_QAT-W8weight", "hard-QAT-W8weight", 4),
        ("SGD_QAT-W4weight", "hard-QAT-W4weight", 4), ("QAT-W4A4-IAO", "hard-QAT-W4A4-IAO", 4),
        ("QAT-W4A4-EFRAP", "hard-QAT-W4A4-EFRAP", 4), ("SGD_Mixup-a0.2", "hard-Mixup", 2),
        ("SGD_HighLR", "hard-HighLR", 2), ("SGD_Short", "hard-Short", 2),
        ("Finetune-from-ImageNet_CE", "hard-Finetune-ImageNet", 2),
        ("CleanParent-QuEST__SGD-plateau", "hard-CleanParent-QuEST", 4),
        ("CleanParent-QuANTI__SGD-lr0.1-200ep", "hard-CleanParent-QuANTI", 4)]


def take(c, n, used_par):
    seen, out = set(used_par), []
    for _, r in c.sort_values("sha256").iterrows():
        k = parent_key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r.to_dict() | {"parent_key": k})
        if len(out) == n:
            break
    return pd.DataFrame(out)


def main():
    b1 = pd.read_csv(ROOT / "cohorts/batch1.csv")
    used_sha, used_par = set(b1.sha256), set(b1.parent_key)
    m = pd.read_csv(DATA / "MODEL_MANIFEST.csv", low_memory=False)
    m["category"] = m.data_dir.str.split("/").str[3]
    s = m[(m.arch == "ResNet18") & (m.dataset == "CIFAR-10") & (m.hold_unopened != True) & ~m.sha256.isin(used_sha)]  # noqa: E712
    rows = []
    for fam, rec, cell in QCB_CELLS:
        c = s[(s.category == f"QCB__{fam}__{rec}")]
        c = c[(c.fp_asr_verified.astype(float) <= .2) & c.apply(lambda r: qok(r, TARGET_RECIPES[rec]), axis=1)]
        t = take(c, QCB_N, used_par)
        t["group"], t["cell"], t["recipes"], t["n_candidates"] = "QCB", cell, ",".join(TARGET_RECIPES[rec]), len(c)
        rows.append(t)
    for cat, cell, n in SAME + HARD:
        c = s[(s.category == f"Normal__{cat}")]
        if cat in [x[0] for x in SAME]:
            c = c[c.is_clean_parent != True]  # noqa: E712
        if cat == "SGD_PlainCE":
            c = c[(c.lr.astype(float) == .01) & (c.epochs.astype(float) == 40)]
        t = take(c, n, used_par)
        if t.empty:
            print("none left for", cat)
            continue
        grp = ("normal-light" if cell == "light" else "normal-standard") if (cat, cell, n) in SAME else "normal-hard"
        t["group"], t["cell"], t["recipes"], t["n_candidates"] = grp, cell, ",".join(ALL_RECIPES), len(c)
        rows.append(t)
    b = pd.concat(rows, ignore_index=True)
    b["model_key"] = b.cell.str.replace("/", "-") + "__" + b.sha256.str[:10]
    assert not set(b.sha256) & used_sha and not set(b.parent_key) & used_par
    cols = ["model_key", "group", "cell", "family", "target_recipe", "recipes", "target", "trigger", "parent_key",
            "n_candidates", "graph_id", "input_norm_verified", "fp_acc_verified", "fp_asr_verified", "sha256", "data_dir"]
    b[cols].to_csv(ROOT / "cohorts/batch2.csv", index=False)
    print(b.groupby(["group", "cell"]).size().to_string())
    print("total", len(b), "jobs", sum(len(r.split(",")) for r in b.recipes))


if __name__ == "__main__":
    main()
