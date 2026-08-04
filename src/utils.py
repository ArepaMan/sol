"""Small shared helpers: seeding, device selection, parameter counting."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG this project touches.

    `deterministic=True` additionally forces deterministic cuDNN kernels. It is
    off by default because it costs throughput and, for a 20-hour training run,
    seed-level reproducibility is enough — full bitwise determinism only matters
    for the ablation seed-variance study (M7), which sets it explicitly.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model: torch.nn.Module, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in params)


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PiB"


def human_count(n: float) -> str:
    """Format a parameter/token count: 52_000_000 -> '52.0M'."""
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000.0:
            return f"{n:.1f}{unit}"
        n /= 1000.0
    return f"{n:.1f}T"


def describe_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    return (
        f"{props.name} | {human_bytes(props.total_memory)} | "
        f"sm_{cap[0]}{cap[1]} | bf16={torch.cuda.is_bf16_supported()}"
    )
