"""Weight perturbations written in units of the recipe's grid step s (fixed grid, see grids.py).

Every perturbation is applied in the recipe's quantization space (the replica: BN folded for torch fbgemm / qnnpack,
unfolded otherwise) to EVERY Conv2d / Linear weight, unless `layer_mask` restricts it.

  uniform   W + c s U(-1, 1)                        then rounded on the fixed grid
  gauss     W + c s N(0, 1/3)   (variance of the uniform of the same c)
  stoch     stochastic rounding: up with probability equal to the position inside the cell (c ignored)
  gshift    the grid moved by c s: quant(W - c s) + c s  (all weights, deterministic)
"""
from __future__ import annotations

import torch

from .grids import LayerGrid

TYPES = ("uniform", "gauss", "stoch", "gshift")


@torch.no_grad()
def perturbed_q(W: dict, gs: list[LayerGrid], kind: str, c: float, seed: int, layer_mask=None) -> dict:
    """{layer: quantized weight} for Q(F + eta)."""
    g = torch.Generator(device=next(iter(W.values())).device).manual_seed(int(seed))
    out = {}
    for i, lg in enumerate(gs):
        w = W[lg.name]
        if layer_mask is not None and not layer_mask[i]:
            out[lg.name] = lg.quant(w)
            continue
        s = lg.scale
        if kind == "uniform":
            out[lg.name] = lg.quant(w + c * s * (torch.rand(w.shape, generator=g, device=w.device) * 2 - 1))
        elif kind == "gauss":
            out[lg.name] = lg.quant(w + c * s * torch.randn(w.shape, generator=g, device=w.device) / 3 ** 0.5)
        elif kind == "stoch":
            u = lg.code_coord(w)
            lo = torch.floor(u)
            up = torch.rand(w.shape, generator=g, device=w.device) < (u - lo)
            codes = torch.clamp(lo + up.float(), lg.qmin, lg.qmax)
            out[lg.name] = lg.from_codes(codes)
        elif kind == "gshift":
            out[lg.name] = lg.quant(w - c * s) + c * s
        else:
            raise ValueError(kind)
    return out
