#!/usr/bin/env python3
"""Batch X (E05x, natural experiment for hypothesis J): QCB whose attack-time quantization simulator (D-hat) or dataset
differs from batch 1, chosen outcome-blind with the batch-1 rule (validity filter, SHA256 order, distinct parents).

  CIFAR-10 VGG16-BN : Tian 8-bit (D-hat: every layer's weights), Tian W4A4 (D-hat: every conv + activation quantization),
                      Ma 8-bit (D-hat unchanged: every layer's weight cells) as the same-D-hat reference
  CIFAR-100 ResNet18: QuEST 8-bit, QuEST W4A4 (records: different switch per dataset), Tian 8-bit (same D-hat as batch 1)
  GTSRB ResNet18    : QuEST 8-bit, QuEST W4A4, Tian 8-bit
No normals: E05x reads only ASR (mechanism), not trigger-free statistics.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cohorts"))
from make_batch1 import parent_key, qok  # noqa: E402
from qcbobs.env import DATA  # noqa: E402
from qcbobs.models import TARGET_RECIPES  # noqa: E402

CELLS = [  # dataset, arch, family, target recipe, cell, n
    ("CIFAR-10", "VGG16-BN", "Tian", "INT8-qnnpack", "C10-VGG-Tian-8", 6),
    ("CIFAR-10", "VGG16-BN", "Tian", "W4A4-EFRAP", "C10-VGG-Tian-W4A4", 6),
    ("CIFAR-10", "VGG16-BN", "Ma", "INT8-fbgemm", "C10-VGG-Ma-8", 4),
    ("CIFAR-100", "ResNet18", "QuEST", "IAO-W8A8", "C100-R18-QuEST-8", 6),
    ("CIFAR-100", "ResNet18", "QuEST", "IAO-W4A4", "C100-R18-QuEST-W4A4", 6),
    ("CIFAR-100", "ResNet18", "Tian", "INT8-qnnpack", "C100-R18-Tian-8", 4),
    ("GTSRB", "ResNet18", "QuEST", "IAO-W8A8", "GTSRB-R18-QuEST-8", 6),
    ("GTSRB", "ResNet18", "QuEST", "IAO-W4A4", "GTSRB-R18-QuEST-W4A4", 6),
    ("GTSRB", "ResNet18", "Tian", "INT8-qnnpack", "GTSRB-R18-Tian-8", 4),
]


def dhat(row) -> str:
    info = json.loads((DATA / row.data_dir / "info.json").read_text())
    tr = (info.get("catalogue_row") or {}).get("train") or {}
    return str(tr.get("fine_tune_scope") or tr.get("finetune_scope") or "")[:300]


def main():
    m = pd.read_csv(DATA / "MODEL_MANIFEST.csv", low_memory=False)
    m["category"] = m.data_dir.str.split("/").str[3]
    s = m[m.hold_unopened != True]  # noqa: E712
    rows = []
    for ds, arch, fam, rec, cell, n in CELLS:
        c = s[(s.dataset == ds) & (s.arch == arch) & (s.category == f"QCB__{fam}__{rec}")]
        c = c[(c.fp_asr_verified.astype(float) <= .2) & c.apply(lambda r: qok(r, TARGET_RECIPES[rec]), axis=1)]
        seen, out = set(), []
        for _, r in c.sort_values("sha256").iterrows():
            k = parent_key(r)
            if k in seen:
                continue
            seen.add(k)
            out.append(r.to_dict() | {"parent_key": k, "dhat": dhat(r)})
            if len(out) == n:
                break
        t = pd.DataFrame(out)
        if t.empty:
            print("none for", cell)
            continue
        t["group"], t["cell"], t["recipes"], t["n_candidates"] = "QCB", cell, ",".join(TARGET_RECIPES[rec]), len(c)
        rows.append(t)
    b = pd.concat(rows, ignore_index=True)
    b["model_key"] = b.cell + "__" + b.sha256.str[:10]
    cols = ["model_key", "group", "cell", "dataset", "arch", "family", "target_recipe", "recipes", "target", "trigger",
            "parent_key", "n_candidates", "graph_id", "input_norm_verified", "fp_acc_verified", "fp_asr_verified", "sha256",
            "data_dir", "dhat"]
    b[cols].to_csv(ROOT / "cohorts/batch_x.csv", index=False)
    print(b.groupby("cell").size().to_string())
    print("total", len(b), "jobs", sum(len(r.split(",")) for r in b.recipes))
    for cell, g in b.groupby("cell"):
        print(cell, "|", g.dhat.iloc[0][:160])


if __name__ == "__main__":
    main()
