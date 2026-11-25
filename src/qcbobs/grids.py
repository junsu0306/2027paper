"""Weight grids of each deployment recipe, so that perturbations can be written in units of the grid step s.

For a recipe we build the same weight replica deploysim builds (BN folded for the torch fbgemm / qnnpack views, unfolded
otherwise) and, per Conv2d / Linear layer, the grid that recipe rounds onto: step (scale), zero point, code range and the
exact rounding operator. The grid is computed once from the FP32 weights W and then held FIXED, so that
quant(W + eta) moves only the rounding decisions and not the scale (a re-computed scale would add a second, global
perturbation; see Research_Plan "가설 I").

`check()` compares quant(W) with the replica + residual that MVD's own view code produces (it must match exactly up to
float error); it is a consistency check of this file, not a re-verification of the models.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from . import env
from .models import RECIPES


@dataclass
class LayerGrid:
    name: str
    scale: torch.Tensor          # broadcastable to the weight
    zp: torch.Tensor             # broadcastable, float (0 for symmetric grids)
    qmin: int
    qmax: int
    quant: Callable[[torch.Tensor], torch.Tensor]   # fixed-grid fake quantization (the recipe's own rounding)

    def code_coord(self, w: torch.Tensor) -> torch.Tensor:
        """Continuous code coordinate u = w / s + zp (u integer <=> on the grid)."""
        return w / self.scale + self.zp

    def codes(self, wq: torch.Tensor) -> torch.Tensor:
        return torch.round(wq / self.scale + self.zp)

    def from_codes(self, c: torch.Tensor) -> torch.Tensor:
        return (c - self.zp) * self.scale


def _bshape(w):
    return (-1,) + (1,) * (w.dim() - 1)


def _torch_grid(name, w, per_channel: bool, bits: int):
    from torch.ao.quantization import get_default_qconfig
    from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver
    if bits == 8:  # quantization.torch_weight_quantize: the backend default weight observer
        obs = get_default_qconfig("fbgemm" if per_channel else "qnnpack").weight().to(w.device)
    else:          # ccb_stage_a_bits.torch_weight_quantize_4b
        if per_channel:
            obs = PerChannelMinMaxObserver(ch_axis=0, dtype=torch.qint8, qscheme=torch.per_channel_symmetric, quant_min=-8, quant_max=7)
        else:
            obs = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, quant_min=-8, quant_max=7)
        obs = obs.to(w.device)
    obs(w.detach())
    s, z = obs.calculate_qparams()
    qmin, qmax = int(obs.quant_min), int(obs.quant_max)
    if per_channel:
        s, z = s.to(w.device), z.to(w.device, torch.int32)
        quant = lambda x: torch.fake_quantize_per_channel_affine(x, s, z, 0, qmin, qmax)  # noqa: E731
        return LayerGrid(name, s.view(_bshape(w)).float(), z.view(_bshape(w)).float(), qmin, qmax, quant)
    sf, zi = float(s), int(z)
    quant = lambda x: torch.fake_quantize_per_tensor_affine(x, sf, zi, qmin, qmax)  # noqa: E731
    return LayerGrid(name, torch.tensor(sf, device=w.device), torch.tensor(float(zi), device=w.device), qmin, qmax, quant)


def _quanti_grid(name, w, bits: int):
    # quantization.quanti_weight_quantize (8 bit) / ccb_stage_a_bits.quanti_weight_quantize_4b (4 bit)
    q = 127.0 if bits == 8 else 7.0
    qmin, qmax = (-128, 127) if bits == 8 else (-8, 7)
    inv = q / (w.detach().abs().amax() + 1e-14)
    quant = lambda x: torch.clamp(torch.floor(x * inv + 0.5), qmin, qmax) / inv  # noqa: E731
    return LayerGrid(name, (1.0 / inv).reshape(()), torch.zeros((), device=w.device), qmin, qmax, quant)


def _efrap_grid(name, w, bits: int, E):
    flat = w.detach().flatten(1)
    s, z = E.qparams(flat.min(1).values, flat.max(1).values, bits)
    s, z = s.view(_bshape(w)), z.view(_bshape(w))
    quant = lambda x: E.fake_quantize(x, s, z, bits)  # noqa: E731
    return LayerGrid(name, s, z, 0, 2 ** bits - 1, quant)


def _iao_grid(name, w, bits: int):
    # deploysim.iao_weight: per output channel signed symmetric, scale max|w| / qmax, rounding half away from zero
    qmax = float((1 << (bits - 1)) - 1)
    flat = w.detach().reshape(w.shape[0], -1)
    m = torch.maximum(flat.min(1).values.abs(), flat.max(1).values.abs())
    s = torch.clamp(m / qmax, min=float(torch.finfo(torch.float32).eps)).view(_bshape(w))

    def quant(x):
        v = x / s
        return torch.clamp(torch.sign(v) * torch.floor(v.abs() + 0.5), -qmax, qmax) * s
    return LayerGrid(name, s, torch.zeros((), device=w.device), int(-qmax), int(qmax), quant)


def weight_spec(recipe: str) -> str:
    env.setup()
    import deploysim as DS
    return DS.SPECS[DS.VIEW_RECIPE[RECIPES[recipe]["view_key"]]]["w"]


def replica(model: nn.Module, recipe: str, device):
    """(replica in the recipe's weight space with FP32 weights, [(name, module)], reference deltas from MVD's view code)."""
    env.setup()
    import deploysim as DS
    w = weight_spec(recipe)
    if w.startswith("view:"):
        _, band, view = w.split(":")
        if band == "8bit":
            from multiview_detection.quantization import stage_a_replica as rep_of
        else:
            from multiview_detection import mvd_v3 as V3
            V3._paths()
            import paper_e9_maw4a4_dev as MW
            MW.install_views()
            from multiview_detection import ccb_stage_a_bits as BITS
            rep_of = BITS.stage_a_replica_4b
        rep, layers, deltas = rep_of(model, view, device)
    elif w.startswith("iao"):
        rep = copy.deepcopy(model).to(device).eval()
        for p in rep.parameters():
            p.requires_grad_(False)
        layers = DS.qlayers(rep)
        b = int(w[3:])
        deltas = {n: DS.iao_weight(m.weight, b) - m.weight.detach() for n, m in layers}
    else:
        raise ValueError(w)
    return rep.eval(), layers, deltas


def grids(layers, recipe: str) -> list[LayerGrid]:
    w = weight_spec(recipe)
    out = []
    E = None
    for i, (n, m) in enumerate(layers):
        W = m.weight.detach()
        if w in ("view:8bit:torch_fbgemm", "view:8bit:torch_qnnpack"):
            out.append(_torch_grid(n, W, w.endswith("fbgemm"), 8))
        elif w in ("view:W4A4:torch_fbgemm_4b", "view:W4A4:torch_qnnpack_4b"):
            out.append(_torch_grid(n, W, "fbgemm" in w, 4))
        elif w == "view:8bit:quanti":
            out.append(_quanti_grid(n, W, 8))
        elif w == "view:W4A4:quanti_4b":
            out.append(_quanti_grid(n, W, 4))
        elif w == "view:W4A4:efrap_w4":
            if E is None:
                import paper_e9_maw4a4_dev as MW
                E = MW.efrap()
            out.append(_efrap_grid(n, W, 8 if i in (0, len(layers) - 1) else 4, E))
        elif w.startswith("iao"):
            out.append(_iao_grid(n, W, int(w[3:])))
        else:
            raise ValueError(f"no grid for weight spec {w}")
    return out


@torch.no_grad()
def check(layers, gs: list[LayerGrid], deltas) -> dict:
    """quant(W) vs W + delta(MVD view code): max |difference| in grid steps and the count of differing codes."""
    worst, ndiff, n = 0.0, 0, 0
    for (name, m), g in zip(layers, gs):
        W = m.weight.detach()
        a, b = g.quant(W), W + deltas[name]
        d = ((a - b).abs() / g.scale).max().item()
        worst = max(worst, d)
        ndiff += int(((a - b).abs() > 0.5 * g.scale).sum())
        n += W.numel()
    return {"max_diff_steps": worst, "codes_differing": ndiff, "n_weights": n}


@torch.no_grad()
def changed_fraction(layers, gs, base_q: dict, new_q: dict) -> float:
    ch, n = 0, 0
    for (name, _), g in zip(layers, gs):
        ch += int(((new_q[name] - base_q[name]).abs() > 0.5 * g.scale).sum())
        n += base_q[name].numel()
    return ch / n
