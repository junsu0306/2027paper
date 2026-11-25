"""Paths, the pinned MultiViewDetection (MVD) archive, and backend settings.

The models and the Data verification are used as they are (Data/README.md). The deployment simulators and model graphs
still live in the frozen MVD archive; we import them from there and record the commit in every result row.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("QCB_DATA", "/workspace/etri_iitp/JS/Data"))
MVD = Path(os.environ.get("QCB_MVD", "/workspace/etri_iitp/JS/MultiViewDetection"))
# MVD commit the code below was written against (2026-11-25). A different HEAD is recorded, not refused.
MVD_PIN = "21f16bdd0fae605df0f0d0ef036934e5cced6399"

_DONE = False


def setup() -> dict:
    """Put Data/tools and the MVD modules on sys.path (idempotent). Returns the provenance dict."""
    global _DONE
    paths = [DATA / "tools", MVD / "src", MVD / "scripts/v6/supp3", MVD / "scripts/v6", MVD / "scripts/v4", MVD / "scripts/e3"]
    for p in reversed(paths):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    _DONE = True
    return provenance()


def mvd_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(MVD), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance() -> dict:
    head = mvd_head()
    return {
        "mvd_head": head,
        "mvd_pin_match": head == MVD_PIN,
        "deploysim_sha256": sha256_file(MVD / "scripts/v6/supp3/deploysim.py")[:16],
        "manifest_sha256": sha256_file(DATA / "MODEL_MANIFEST.csv")[:16],
    }


def backend() -> None:
    """FP32 without TF32 (as the MVD verification)."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
