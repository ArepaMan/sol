"""Environment gates.

These exist because the two most likely M0 failures are silent: a CPU-only
torch wheel (pip resolves it happily if you forget --index-url) and a venv
accidentally built from the wrong interpreter. Both surface here in one second
instead of ten milestones later.
"""

from __future__ import annotations

import sys

import pytest
import torch


def test_python_version_supports_torch():
    # PyTorch stable wheels do not cover 3.13/3.14 yet. The machine's system
    # Python is 3.14, so this catches a venv built from the wrong interpreter.
    assert (3, 11) <= sys.version_info[:2] < (3, 13), (
        f"Python {sys.version_info.major}.{sys.version_info.minor} has no stable "
        "PyTorch wheel — rebuild the venv with py -3.12"
    )


@pytest.mark.gpu
def test_cuda_available():
    assert torch.cuda.is_available(), (
        "CUDA not available — you almost certainly installed the CPU-only torch "
        "wheel. Reinstall with --index-url https://download.pytorch.org/whl/cu124"
    )


@pytest.mark.gpu
def test_bf16_supported():
    # Sol trains in BF16, which needs Ampere or newer. The RTX 4070 Laptop
    # (Ada, sm_89) qualifies; a Turing card would not and would need the
    # float16 + GradScaler path instead.
    assert torch.cuda.is_bf16_supported(), (
        "BF16 unsupported on this GPU — set precision: float16 in the config "
        "and re-enable GradScaler in src/train.py"
    )


@pytest.mark.gpu
def test_sufficient_vram():
    total_mib = torch.cuda.get_device_properties(0).total_memory / 1024**2
    assert total_mib >= 7500, (
        f"Only {total_mib:.0f} MiB VRAM — the config assumes ~8 GB. Use the OOM "
        "fallback ladder in docs/PROJECT.md."
    )
