#!/usr/bin/env python3
"""Batch XN: normals and no-poison controls for the three batch_x settings (VGG16-BN CIFAR-10, ResNet18 CIFAR-100,
ResNet18 GTSRB), so that trigger-free statistics of batch_x QCB have a same-setting reference (E14, E10).
Chosen outcome-blind with the batch-1/2 rule (SHA256 order, distinct parents, hold models never opened).

Per setting: SGD plain 2 (light), Adam 1 (standard), QAT-W8weight 2 and QAT-W4A4-IAO 1 (hard), ImageNet fine-tune 1 (hard,
where it exists), controls: Tian INT8-qnnpack 1 and QuEST IAO-W8A8 1 (IAO-W4A4 where W8A8 does not exist).
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cohorts"))
from make_batch1 import parent_key  # noqa: E402
from qcbobs.env import DATA  # noqa: E402
from qcbobs.models import ALL_RECIPES  # noqa: E402

SETTINGS = [("CIFAR-10", "VGG16-BN", "C10-VGG"), ("CIFAR-100", "ResNet18", "C100-R18"), ("GTSRB", "ResNet18", "GTSRB-R18")]
PICK = [  # (category, cell suffix, group, n)
    ("Normal__SGD_PlainCE", "light", "normal-light", 2),
    ("Normal__Adam", "standard-Adam", "normal-standard", 1),
    ("Normal__SGD_QAT-W8weight", "hard-QAT-W8weight", "normal-hard", 2),
    ("Normal__QAT-W4A4-IAO", "hard-QAT-W4A4-IAO", "normal-hard", 1),
    ("Normal__Finetune-from-ImageNet_CE", "hard-Finetune-ImageNet", "normal-hard", 1),
    ("Control-nopoison__Tian__INT8-qnnpack", "control-Tian-INT8-qnnpack", "control-nopoison", 1),
    ("Control-nopoison__QuEST__IAO-W8A8", "control-QuEST-IAO-W8A8", "control-nopoison", 1),
]
FALLBACK = {"Control-nopoison__QuEST__IAO-W8A8": ("Control-nopoison__QuEST__IAO-W4A4", "control-QuEST-IAO-W4A4")}


def main():
    used = pd.concat([pd.read_csv(ROOT / f"cohorts/{b}.csv") for b in ("batch1", "batch2", "batch_x")])
    used_sha, used_par = set(used.sha256), set(used.parent_key)
    m = pd.read_csv(DATA / "MODEL_MANIFEST.csv", low_memory=False)
    m["category"] = m.data_dir.str.split("/").str[3]
    s = m[(m.hold_unopened != True) & ~m.sha256.isin(used_sha)]  # noqa: E712
    rows = []
    for ds, arch, tag in SETTINGS:
        seen = set(used_par)
        for cat, suffix, grp, n in PICK:
            c = s[(s.dataset == ds) & (s.arch == arch) & (s.category == cat)]
            if c.empty and cat in FALLBACK:
                cat, suffix = FALLBACK[cat]
                c = s[(s.dataset == ds) & (s.arch == arch) & (s.category == cat)]
            if cat == "Normal__SGD_PlainCE" and not c.empty:
                c = c[c.is_clean_parent != True]  # noqa: E712
            out = []
            for _, r in c.sort_values("sha256").iterrows():
                k = parent_key(r)
                if k in seen:
                    continue
                seen.add(k)
                out.append(r.to_dict() | {"parent_key": k, "group": grp, "cell": f"{tag}-{suffix}", "n_candidates": len(c)})
                if len(out) == n:
                    break
            if len(out) < n:
                print("short:", tag, cat, len(out), "of", n)
            rows += out
    b = pd.DataFrame(rows)
    b["recipes"] = ",".join(ALL_RECIPES)
    b["model_key"] = b.cell + "__" + b.sha256.str[:10]
    cols = ["model_key", "group", "cell", "dataset", "arch", "family", "target_recipe", "recipes", "target", "trigger",
            "parent_key", "n_candidates", "graph_id", "input_norm_verified", "fp_acc_verified", "fp_asr_verified", "sha256", "data_dir"]
    b[cols].to_csv(ROOT / "cohorts/batch_xn.csv", index=False)
    print(b.groupby(["dataset", "group", "cell"]).size().to_string())
    print("total", len(b))


if __name__ == "__main__":
    main()
