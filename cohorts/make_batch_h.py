#!/usr/bin/env python3
"""Batch H (held out for part 2, 2026-11-26): ResNet18 on CIFAR-10, CIFAR-100, GTSRB, and VGG16-BN QuEST (8-bit, W4A4) with
same-setting normals and controls; models in no earlier cohort (batch1,
batch2, batch_x, batch_xn, e08) and with no parent used there. Used to replicate part-2 findings that were first seen on
the earlier cohorts (ST03, ST04). Chosen outcome-blind with the batch-1/2 rule: validity filter (F ASR <= .2 and the
verified ASR >= .5 in every target recipe for QCB), SHA256 order, distinct parents, hold models never opened.
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

SETTINGS = [("CIFAR-10", "C10"), ("CIFAR-100", "C100"), ("GTSRB", "GTSRB")]
QCB_N = {"CIFAR-10": 4, "CIFAR-100": 3, "GTSRB": 3}
QCB_N_EXTRA = {"QuEST-W4A4": 2}          # QuEST W4A4 is the key family of ST04: twice the per-setting count
NORMALS = [("SGD_PlainCE", "light", "normal-light", 3), ("Adam", "standard-Adam", "normal-standard", 2),
           ("AdamW", "standard-AdamW", "normal-standard", 1), ("SGD-lr0.1-200ep", "standard-SGDlr.1", "normal-standard", 1),
           ("QAT-W4A4-IAO", "hard-QAT-W4A4-IAO", "normal-hard", 2), ("QAT-W4A4-EFRAP", "hard-QAT-W4A4-EFRAP", "normal-hard", 2),
           ("SGD_QAT-W4weight", "hard-QAT-W4weight", "normal-hard", 2), ("SGD_QAT-W8weight", "hard-QAT-W8weight", "normal-hard", 2),
           ("Finetune-from-ImageNet_CE", "hard-Finetune-ImageNet", "normal-hard", 1), ("SGD_LabelSmoothing", "hard-LabelSmoothing", "normal-hard", 1),
           ("SGD_Mixup-a0.2", "hard-Mixup", "normal-hard", 1), ("CleanParent-QuEST__SGD-plateau", "hard-CleanParent-QuEST", "normal-hard", 2)]
CONTROL_PREFIX = ("Control-nopoison__",)
VGG_QCB = [("QuEST", "IAO-W8A8", "QuEST-8"), ("QuEST", "IAO-W4A4", "QuEST-W4A4")]
VGG_N = {"CIFAR-10": 6, "CIFAR-100": 3, "GTSRB": 3}
VGG_NORMALS = [("SGD_PlainCE", "light", "normal-light", 2), ("Adam", "standard-Adam", "normal-standard", 1),
               ("QAT-W4A4-IAO", "hard-QAT-W4A4-IAO", "normal-hard", 1), ("SGD_QAT-W8weight", "hard-QAT-W8weight", "normal-hard", 1)]


def main():
    used = pd.concat([pd.read_csv(ROOT / f"cohorts/{b}.csv") for b in ("batch1", "batch2", "batch_x", "batch_xn", "e08")])
    used_sha, used_par = set(used.sha256), set(used.parent_key)
    m = pd.read_csv(DATA / "MODEL_MANIFEST.csv", low_memory=False)
    m["category"] = m.data_dir.str.split("/").str[3]
    s_all = m[(m.hold_unopened != True) & ~m.sha256.isin(used_sha)]  # noqa: E712
    s = s_all[s_all.arch == "ResNet18"]
    rows = []
    seen = set(used_par)

    def take(c, n, extra):
        out = []
        for _, r in c.sort_values("sha256").iterrows():
            k = parent_key(r)
            if k in seen:
                continue
            seen.add(k)
            out.append(r.to_dict() | {"parent_key": k, "n_candidates": len(c)} | extra)
            if len(out) == n:
                break
        return out

    for ds, tag in SETTINGS:
        d = s[s.dataset == ds]
        for fam, rec, cell in QCB_CELLS:
            c = d[d.category == f"QCB__{fam}__{rec}"]
            c = c[(c.fp_asr_verified.astype(float) <= .2) & c.apply(lambda r: qok(r, TARGET_RECIPES[rec]), axis=1)]
            n = QCB_N[ds] * QCB_N_EXTRA.get(cell, 1)
            got = take(c, n, {"group": "QCB", "cell": f"H-{tag}-{cell}", "recipes": ",".join(TARGET_RECIPES[rec])})
            if len(got) < n:
                print("short:", tag, cell, len(got), "of", n)
            rows += got
        for cat, suffix, grp, n in NORMALS:
            c = d[d.category == f"Normal__{cat}"]
            if grp != "normal-hard":
                c = c[c.is_clean_parent != True]  # noqa: E712
            got = take(c, n, {"group": grp, "cell": f"H-{tag}-{suffix}", "recipes": ",".join(ALL_RECIPES)})
            if len(got) < n:
                print("short:", tag, cat, len(got), "of", n)
            rows += got
        c = d[d.category.str.startswith(CONTROL_PREFIX)]
        c = c[c.verdict_recorded.astype(str).str.startswith("VALID") | c.verdict_recorded.isna()]
        for cat, g in c.groupby("category"):
            rows += take(g, len(g), {"group": "control-nopoison", "cell": f"H-{tag}-control-" + cat.split("__", 1)[1].replace("__", "-"),
                                     "recipes": ",".join(ALL_RECIPES)})
    v = s_all[s_all.arch == "VGG16-BN"]
    for ds, tag in SETTINGS:
        d = v[v.dataset == ds]
        for fam, rec, cell in VGG_QCB:
            c = d[d.category == f"QCB__{fam}__{rec}"]
            c = c[(c.fp_asr_verified.astype(float) <= .2) & c.apply(lambda r: qok(r, TARGET_RECIPES[rec]), axis=1)]
            got = take(c, VGG_N[ds], {"group": "QCB", "cell": f"H-{tag}-VGG-{cell}", "recipes": ",".join(TARGET_RECIPES[rec])})
            if len(got) < VGG_N[ds]:
                print("short:", tag, "VGG", cell, len(got), "of", VGG_N[ds])
            rows += got
        for cat, suffix, grp, n in VGG_NORMALS:
            c = d[d.category == f"Normal__{cat}"]
            if grp != "normal-hard":
                c = c[c.is_clean_parent != True]  # noqa: E712
            got = take(c, n, {"group": grp, "cell": f"H-{tag}-VGG-{suffix}", "recipes": ",".join(ALL_RECIPES)})
            if len(got) < n:
                print("short:", tag, "VGG", cat, len(got), "of", n)
            rows += got
        c = d[d.category.str.startswith("Control-nopoison__QuEST")]
        c = c[c.verdict_recorded.astype(str).str.startswith("VALID") | c.verdict_recorded.isna()]
        for cat, g in c.groupby("category"):
            rows += take(g, len(g), {"group": "control-nopoison", "cell": f"H-{tag}-VGG-control-" + cat.split("__", 1)[1].replace("__", "-"),
                                     "recipes": ",".join(ALL_RECIPES)})
    b = pd.DataFrame(rows)
    b["model_key"] = b.cell.str.replace("/", "-") + "__" + b.sha256.str[:10]
    assert not set(b.sha256) & used_sha and not set(b.parent_key) & used_par and b.parent_key.is_unique
    cols = ["model_key", "group", "cell", "dataset", "arch", "family", "target_recipe", "recipes", "target", "trigger",
            "parent_key", "n_candidates", "graph_id", "input_norm_verified", "fp_acc_verified", "fp_asr_verified", "sha256", "data_dir"]
    b[cols].to_csv(ROOT / "cohorts/batch_h.csv", index=False)
    print(b.groupby(["dataset", "group", "cell"]).size().to_string())
    print("total", len(b))


if __name__ == "__main__":
    main()
