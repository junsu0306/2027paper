#!/usr/bin/env python3
"""Find cohort models whose weights are identical (same float tensors), whatever their file hash.

  python3 cohorts/check_duplicates.py batch1 batch2 batch_x ...
Hashes every floating-point tensor of each model.pt (sorted by name) on CPU. Read only.
Run this on every new cohort before use; keep the first model_key (sorted) of each duplicate group.
"""
import collections
import hashlib
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/workspace/etri_iitp/JS/Data")


def state(x):
    if isinstance(x, dict):
        for k in ("state_dict", "model", "model_state_dict"):
            if k in x and isinstance(x[k], dict):
                return x[k]
        return x
    return x.state_dict()


def weight_hash(path):
    s = state(torch.load(path, map_location="cpu", weights_only=False))
    h = hashlib.sha256()
    for k in sorted(k for k in s if torch.is_tensor(s[k]) and s[k].is_floating_point()):
        h.update(k.encode())
        h.update(s[k].detach().float().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    seen = collections.defaultdict(list)
    for name in sys.argv[1:]:
        for _, r in pd.read_csv(ROOT / f"cohorts/{name}.csv").iterrows():
            seen[weight_hash(DATA / r.data_dir / "model.pt")].append((name, r.model_key))
    n = sum(len(v) for v in seen.values())
    print(f"models {n}, distinct weights {len(seen)}")
    for v in seen.values():
        if len(v) > 1:
            print(sorted(v, key=lambda x: x[1]))


if __name__ == "__main__":
    main()
