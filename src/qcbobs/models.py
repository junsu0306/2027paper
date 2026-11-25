"""Model loading (Data/tools/load_model.py, unchanged) and the deployment recipe registry.

Recipes are the seven deployment simulators of MVD scripts/v6/supp3/deploysim.py that the Data verification used
(Data/tools/verify_models.py VIEWS). Short names are ours; the view keys are deploysim's.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from . import env

RECIPES = {
    # short     deploysim view key            nominal bits  weight grid (deploysim SPECS[..]["w"])
    "fbgemm8": {"view_key": "8bit|torch_fbgemm+clip", "bits": 8},
    "qnnpack8": {"view_key": "8bit|torch_qnnpack+clip", "bits": 8},
    "quanti8": {"view_key": "8bit|quanti+clip", "bits": 8},
    "iao8": {"view_key": "8bit|iao_train", "bits": 8},
    "efrap4": {"view_key": "W4A4|efrap", "bits": 4},
    "quanti4": {"view_key": "W4A4|quanti", "bits": 4},
    "iao4": {"view_key": "W4A4|iao_train", "bits": 4},
}
ALL_RECIPES = tuple(RECIPES)
# target_recipe token (MODEL_MANIFEST.csv) -> recipes that switch the backdoor on (Data/tools/verify_models.py VIEWS)
TARGET_RECIPES = {
    "INT8-fbgemm": ["fbgemm8"],
    "INT8-qnnpack": ["qnnpack8"],
    "QuANTI-fake-W8+W4": ["quanti8", "quanti4"],
    "QuANTI-fake-W8": ["quanti8"],
    "QuANTI-fake-W4": ["quanti4"],
    "IAO-W8A8": ["iao8"],
    "IAO-W4A4": ["iao4"],
    "W4A4-EFRAP": ["efrap4"],
}


def load(model_dir: str | Path, device: str = "cuda"):
    """(model on normalised input, norm(x01) -> model input, info). Read-only use of Data/models."""
    env.setup()
    from load_model import load as _load  # Data/tools/load_model.py
    d = Path(model_dir)
    if not d.is_absolute():
        d = env.DATA / d
    model, norm, info = _load(d, device=device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval(), norm, info


def trigger(x01: torch.Tensor, name: str) -> torch.Tensor:
    env.setup()
    from load_model import trigger as _trigger
    return _trigger(x01, name)


def info_of(model_dir: str | Path) -> dict:
    d = Path(model_dir)
    if not d.is_absolute():
        d = env.DATA / d
    return json.loads((d / "info.json").read_text())
