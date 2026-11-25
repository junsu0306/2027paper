"""Images and their roles (Data/datasets/main, split_base.json).

Roles used here (no model was trained or selected on detector_probe):
  probe   detector_probe, 1,000 images: every trigger-free measurement (M1-M4, perturbation search in E03)
  test    final_behavior_test, first 1,000: clean accuracy and ASR (mechanism checks only)
  cal     quant_cal, 1,000: activation calibration of the deployment simulators (as the Data verification)
  train   x_train (uint8): only for the QuEST train-mode IAO deployments, exactly as Data/tools/verify_models.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch

from .env import DATA

DS_FILE = {"CIFAR-10": "cifar10", "CIFAR-100": "cifar100", "GTSRB": "gtsrb", "SVHN": "svhn"}


@dataclass
class Images:
    probe: torch.Tensor      # [N,3,H,W] float in [0,1]
    test: torch.Tensor
    test_y: torch.Tensor
    cal: torch.Tensor
    train_u8: np.ndarray     # NHWC uint8


_cache: dict[str, Images] = {}


def to01(u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(u8)).permute(0, 3, 1, 2).float().div(255)


def load(dataset: str, n_probe: int = 1000, n_test: int = 1000) -> Images:
    key = f"{dataset}|{n_probe}|{n_test}"
    if key in _cache:
        return _cache[key]
    name = DS_FILE[dataset]
    z = np.load(DATA / "datasets/main" / f"{name}.npz")
    roles = json.loads((DATA / "datasets/main/manifests" / name / "split_base.json").read_text())["datasets"][name]["roles"]
    idx = lambda r, n: np.asarray(roles[r]["indices"][:n])  # noqa: E731
    pi, ti, ci = idx("detector_probe", n_probe), idx("final_behavior_test", n_test), idx("quant_cal", 1000)
    out = Images(probe=to01(z["x_test"][pi]), test=to01(z["x_test"][ti]), test_y=torch.from_numpy(z["y_test"][ti].astype(np.int64)),
                 cal=to01(z["x_test"][ci]), train_u8=z["x_train"])
    _cache[key] = out
    return out
