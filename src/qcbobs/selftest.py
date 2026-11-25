"""Consistency of this package with MVD's code (not a model re-verification):
for one model and each recipe, grids.quant(W) == MVD view residual, and our deployed Q == deploysim.deployed(F, view).

  CUDA_VISIBLE_DEVICES=0 python -m qcbobs.selftest [model_key]
"""
import json
import sys
import time

import pandas as pd
import torch

from . import data, deploy, env, grids
from .models import ALL_RECIPES, load


def main():
    env.setup()
    env.backend()
    b = pd.read_csv(env.ROOT / "cohorts/batch1.csv")
    row = b[b.model_key == sys.argv[1]].iloc[0] if len(sys.argv) > 1 else b[b.group == "normal-light"].iloc[0]
    dev = torch.device("cuda")
    F, norm, info = load(row.data_dir, dev)
    imgs = data.load("CIFAR-10")
    out = {"model": row.model_key}
    for r in ALL_RECIPES:
        t0 = time.time()
        rep, layers, deltas = grids.replica(F, r, dev)
        gs = grids.grids(layers, r)
        chk = grids.check(layers, gs, deltas)
        deploy.set_weights(layers, {lg.name: lg.quant(m.weight) for (n, m), lg in zip(layers, gs)})
        d = deploy.consistency(F, rep, r, imgs, norm, dev)
        out[r] = chk | {"max_logit_diff_vs_deploysim": d, "sec": round(time.time() - t0, 1)}
        print(r, out[r], flush=True)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
