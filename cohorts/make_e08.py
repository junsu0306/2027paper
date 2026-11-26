#!/usr/bin/env python3
"""cohorts/e08.csv: the E08 A5000 attack variants received into Data on 2026-11-26 (scripts/ingest_e08.py).
Not part of cohort.load_all() (separate experiment set); same columns as batch1.csv plus variant and seed_id."""
import json
from pathlib import Path

import pandas as pd

DATA = Path("/workspace/etri_iitp/JS/Data")
REC = {"IAO-W8A8": "iao8", "INT8-qnnpack": "qnnpack8"}
rows = []
for d in sorted((DATA / "models/ResNet18/CIFAR-10").glob("QCB-E08__*/E08-*")):
    i = json.loads((d / "info.json").read_text())
    v = i["verification"]
    fam = i["family"]
    var = i["e08_variant"]
    rows.append({"model_key": f"E08-{fam}-{var}__{i['e08_seed_id'].split('-')[-1]}", "group": "QCB", "cell": f"E08-{fam}-{var}",
                 "family": fam, "variant": var, "seed_id": i["e08_seed_id"], "target_recipe": i["recipe"], "recipes": REC[i["recipe"]],
                 "target": i["target"], "trigger": "Patch6x6", "parent_key": f"self:{i['sha256'][:16]}", "n_candidates": 1,
                 "graph_id": v["graph"], "input_norm_verified": v["norm"], "fp_acc_verified": v["fp_clean_acc"], "fp_asr_verified": v["fp_asr"],
                 "sha256": i["sha256"], "data_dir": str(d.relative_to(DATA))})
pd.DataFrame(rows).to_csv(Path(__file__).with_name("e08.csv"), index=False)
print(len(rows), "rows")
