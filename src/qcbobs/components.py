"""Switching deployment components on and off inside a deploysim-built model (E05 logic, shared for E06).

A deploysim model carries one ActQ forward-pre-hook per quantized module (Conv/Linear inputs, and pools for iao). After
deploysim.build has calibrated them, we can change how each one acts without recalibrating:
  off           bypass (FP32 activation)
  full          the calibrated deploysim quantizer
  clip          the calibrated range only (no rounding); clip:<alpha> scales the range by alpha
  round         the calibrated step only (no range limit); round:<bits> uses the calibrated range / (2^bits - 1)
"""
from __future__ import annotations

import torch

from . import deploy
from .models import RECIPES


def _round_away(v):
    return torch.sign(v) * torch.floor(v.abs() + 0.5)


def act_quantizers(dm):
    """[(module name, ActQ)] in module order."""
    DS = deploy._ds()
    out = []
    for name, mod in dm.named_modules():
        for h in mod._forward_pre_hooks.values():
            if isinstance(h, DS.ActQ):
                out.append((name, h))
    return out


def _range(a, x=None):
    """Calibrated real-valued range [lo, hi] that the quantizer clamps to."""
    if a.mode == "affine":
        qmax = (1 << a.bits) - 1
        return (0 - a.z) * a.s, (qmax - a.z) * a.s
    if a.mode == "quanti":
        qmax = float((1 << a.bits) - 1)
        scale = qmax / (a.hi - a.lo + 1e-14)
        zp = float(torch.round(torch.tensor(a.lo * scale)))
        return zp / scale, (qmax + zp) / scale
    if a.mode == "quanti_dyn":
        return float(x.detach().min()), float(x.detach().max())
    if a.mode == "iao":
        bits = a.bits
        qmin, qmax = -float(1 << (bits - 1)), float((1 << (bits - 1)) - 1)
        L = a.L if getattr(a, "L", None) is not None else max(abs(a.lo), abs(a.hi))
        s = max(L / ((qmax - qmin) / 2), float(torch.finfo(torch.float32).eps))
        return qmin * s, qmax * s
    raise ValueError(a.mode)


def make_q(a, variant: str):
    """Replacement for ActQ.q. variant: clip[:alpha] | round[:bits] (see module docstring). E05's clip / round are the
    alpha = 1 / recipe-bits cases, written there per mode; here through the common range."""
    kind, _, arg = variant.partition(":")
    if kind == "clip":
        alpha = float(arg) if arg else 1.0

        def f(x):
            lo, hi = _range(a, x)
            return torch.clamp(x, alpha * lo, alpha * hi)
        return f
    if kind == "round":
        bits = int(arg) if arg else a.bits

        def g(x):
            lo, hi = _range(a, x)
            step = max((hi - lo) / ((1 << bits) - 1), 1e-12)
            if a.mode == "iao":
                return _round_away(x / step) * step
            return torch.round(x / step) * step
        return g
    raise ValueError(variant)


def set_acts(acts, variant: str, only=None):
    """Set every quantizer (or only those whose index is in `only`; the rest bypass) to `variant`."""
    for i, (_, a) in enumerate(acts):
        if not hasattr(a, "_orig_q"):
            a._orig_q = a.q
        if variant == "off" or (only is not None and i not in only):
            a.state = "bypass"
            continue
        a.state = "quant"
        a.q = a._orig_q if variant == "full" else make_q(a, variant)


def build(rep, recipe, imgs, norm, dev, bn="recipe"):
    """deploysim.build with spec['w'] = None (weights already in rep). bn='stored' uses the registered *_bneval ablation."""
    DS = deploy._ds()
    spec = dict(DS.SPECS[DS.VIEW_RECIPE[RECIPES[recipe]["view_key"]]])
    spec["w"] = None
    if bn == "stored":
        spec["bn"] = "eval"
    return DS.build(rep, spec, norm(imgs.cal.to(dev)), deploy.train_arg(recipe, imgs, norm), device=str(dev))


def bn_layers(m):
    return [(n, mod) for n, mod in m.named_modules() if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)]


@torch.no_grad()
def set_bn_mix(dm, stats_a, stats_b, lam: float, only=None):
    """running stats = (1 - lam) * a + lam * b for every BN (or only BN indices in `only`; the rest get a)."""
    for i, (n, mod) in enumerate(bn_layers(dm)):
        la = lam if (only is None or i in only) else 0.0
        mod.running_mean.copy_((1 - la) * stats_a[n][0] + la * stats_b[n][0])
        mod.running_var.copy_((1 - la) * stats_a[n][1] + la * stats_b[n][1])


def bn_stats(m):
    return {n: (mod.running_mean.detach().clone(), mod.running_var.detach().clone()) for n, mod in bn_layers(m)}
