"""Measurements. Trigger-free ones read only `probe` images; ASR / accuracy read `test` images (mechanism checks).

Raw distributions are stored (histograms, K x K transition counts) so that any skew statistic can be computed later.
  M1  transition F-prediction -> prediction of the measured model (probe)
  M2  input-noise titration: x01 + N(0, sigma^2) clamped to [0,1]; transition base prediction -> noisy prediction
  M3  ProP: N(0, sigma^2) added to the input of every Conv2d / Linear except the first (MVD baselines.run_prop config:
      sigmas 100 and 2, 4 Monte-Carlo passes, probe images); histogram of predictions. On deployed models the noise hook
      runs after deploysim's activation quantizer (registered later).
  M4  inference dropout: elementwise dropout (rate p, inverted scaling) at the same inputs as M3; transition counts.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as Fn
from torch import nn

BS = 128
PROP_SIGMAS = (100.0, 2.0)
PROP_MC = 4
DROP_RATES = (0.05, 0.1, 0.2, 0.3, 0.5)
DROP_MC = 4
INPUT_SIGMAS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
INPUT_MC = 2


def qlayers(m: nn.Module):
    return [mod for mod in m.modules() if isinstance(mod, (nn.Conv2d, nn.Linear))]


@torch.no_grad()
def logits(model, xn: torch.Tensor) -> torch.Tensor:
    return torch.cat([model(xn[i:i + BS]) for i in range(0, len(xn), BS)])


def transition(a: torch.Tensor, b: torch.Tensor, K: int) -> list:
    t = torch.bincount(a.long().cpu() * K + b.long().cpu(), minlength=K * K).view(K, K)
    return t.tolist()


def hist(p: torch.Tensor, K: int) -> list:
    return torch.bincount(p.long().cpu(), minlength=K).tolist()


class _Hook:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, _m, args):
        return (self.fn(args[0]),) + tuple(args[1:])


def _with_hooks(model, fn):
    return [mod.register_forward_pre_hook(_Hook(fn)) for mod in qlayers(model)[1:]]


@torch.no_grad()
def prop(model, xn, K, seed=0) -> dict:
    out = {}
    for sigma in PROP_SIGMAS:
        g = torch.Generator(device=xn.device).manual_seed(seed)
        hs = _with_hooks(model, lambda x, s=sigma: x + torch.randn(x.shape, generator=g, device=x.device) * s)
        try:
            preds = [logits(model, xn).argmax(1) for _ in range(PROP_MC)]
        finally:
            for h in hs:
                h.remove()
        out[f"s{sigma:g}"] = hist(torch.cat(preds), K)
    return out


@torch.no_grad()
def dropout(model, xn, K, base_pred, seed=0) -> dict:
    out = {}
    for p in DROP_RATES:
        g = torch.Generator(device=xn.device).manual_seed(seed)

        def fn(x, p=p):
            keep = (torch.rand(x.shape, generator=g, device=x.device) >= p).to(x.dtype)
            return x * keep / (1 - p)
        hs = _with_hooks(model, fn)
        try:
            t = np.zeros((K, K), dtype=np.int64)
            for _ in range(DROP_MC):
                t += np.asarray(transition(base_pred, logits(model, xn).argmax(1), K))
        finally:
            for h in hs:
                h.remove()
        out[f"p{p:g}"] = t.tolist()
    return out


@torch.no_grad()
def input_noise(model, x01, norm, K, base_pred, seed=0) -> dict:
    out = {}
    for sigma in INPUT_SIGMAS:
        g = torch.Generator(device=x01.device).manual_seed(seed)
        t = np.zeros((K, K), dtype=np.int64)
        for _ in range(INPUT_MC):
            xn = norm((x01 + torch.randn(x01.shape, generator=g, device=x01.device) * sigma).clamp(0, 1))
            t += np.asarray(transition(base_pred, logits(model, xn).argmax(1), K))
        out[f"s{sigma:g}"] = t.tolist()
    return out


@torch.no_grad()
def acc_asr(model, xt_n, yt, xtrig_n, target, K) -> dict:
    """Clean accuracy on test; triggered predictions (histogram) and ASR on non-target test images when a target exists."""
    pc = logits(model, xt_n).argmax(1)
    r = {"acc": float((pc.cpu() == yt).float().mean())}
    if xtrig_n is not None:
        pt = logits(model, xtrig_n).argmax(1).cpu()
        r["trig_hist"] = hist(pt, K)
        if target is not None:
            keep = yt != int(target)
            r["asr"] = float((pt[keep] == int(target)).float().mean())
    return r


def skew(h) -> float:
    """Share of the most frequent class minus 1/K (the plan's '쏠림' on a histogram)."""
    h = np.asarray(h, dtype=float)
    return float(h.max() / max(h.sum(), 1) - 1 / len(h))


def prop_score(h) -> tuple[float, int]:
    """MVD baselines._prop_benign_score -> raw score 1 - benign (higher = more suspicious), suspect class."""
    p = np.asarray(h, dtype=float)
    p = p / max(p.sum(), 1)
    k = int(np.argmax(p))
    sd = p.std()
    ratio = float(np.delete(p, k).std() / sd) if sd > 0 else 1.0
    return float(1 - (1 - p[k]) * ratio), k


def resolve_name(model: nn.Module, name: str) -> str:
    """Module name in `model` for a CIFARResNet18 name (layerK.j[.rest] / linear). The fbgemm8 / qnnpack8 replicas
    (FusedResNet18) name the BasicBlocks blocks.N with N = 2 (K - 1) + j; linear -> fc / classifier when absent; other
    names are unchanged."""
    names = dict(model.named_modules())
    if name in names:
        return name
    if name == "linear":                                  # torchvision-style backbones (hard-Finetune-ImageNet) name the head fc
        for alt in ("fc", "classifier"):
            if alt in names:
                return alt
    parts = name.split(".")
    if parts[0].startswith("layer") and len(parts) > 1 and parts[1].isdigit():
        alt = ".".join([f"blocks.{2 * (int(parts[0][5:]) - 1) + int(parts[1])}"] + parts[2:])
        if alt in names:
            return alt
        if alt.endswith(".0") and alt[:-2] in names:     # shortcut.0 -> shortcut
            return alt[:-2]
    raise KeyError(name)
