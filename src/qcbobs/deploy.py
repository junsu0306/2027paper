"""Deployed models built through deploysim unchanged.

deploysim.deployed(F, view) = deploysim.build(replica + residual, spec). We call deploysim.build on our replica after
writing the weights we want into it (W, quant(W), quant(W + eta), ...), with spec["w"] = None so that deploysim adds
exactly its activation quantizers / calibration / BN handling. With quant(W) this reproduces deploysim.deployed(F, view);
`consistency()` measures that on probe images.
"""
from __future__ import annotations

import torch
from torch import nn

from . import env
from .models import RECIPES


def _ds():
    env.setup()
    import deploysim as DS
    return DS


def train_arg(recipe: str, imgs, norm):
    DS = _ds()
    return (imgs.train_u8, norm) if RECIPES[recipe]["view_key"] in DS.NEEDS_TRAIN else None


def build(rep: nn.Module, recipe: str, imgs, norm, device, act: bool = True) -> nn.Module:
    """Deploy `rep` (weights already set). act=False: weights only, FP32 activations (deep copy, eval)."""
    DS = _ds()
    spec = dict(DS.SPECS[DS.VIEW_RECIPE[RECIPES[recipe]["view_key"]]])
    spec["w"] = None
    if not act:
        spec["a"] = None
        spec["calib"] = "cal"
    cal_x = norm(imgs.cal.to(device))
    return DS.build(rep, spec, cal_x, train_arg(recipe, imgs, norm) if act else None, device=str(device))


@torch.no_grad()
def set_weights(layers, weights: dict) -> None:
    for n, m in layers:
        m.weight.copy_(weights[n])


@torch.no_grad()
def consistency(F: nn.Module, rep: nn.Module, recipe: str, imgs, norm, device, n: int = 256) -> float:
    """max |logit difference| between our Q and deploysim.deployed(F, view) on the first n probe images."""
    DS = _ds()
    vk = RECIPES[recipe]["view_key"]
    ref = DS.deployed(F, vk, norm(imgs.cal.to(device)), train_x=train_arg(recipe, imgs, norm), device=str(device)).eval()
    ours = build(rep, recipe, imgs, norm, device)
    x = norm(imgs.probe[:n].to(device))
    d = max(float((ref(x[i:i + 128]) - ours(x[i:i + 128])).abs().max()) for i in range(0, n, 128))
    del ref, ours
    return d
