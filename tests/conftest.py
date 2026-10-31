"""Test configuration.

Without a CUDA device, Triton kernels run through ``TRITON_INTERPRET=1`` (pure
Python/numpy on the CPU). That validates kernel logic and numerics only; it does
not exercise compiled GPU code. With a CUDA device, kernels are compiled and run
on the GPU. The flag must be set before any ``engine`` module is imported.
"""

import os

import pytest
import torch

HAS_CUDA = torch.cuda.is_available()
if not HAS_CUDA:
    os.environ.setdefault("TRITON_INTERPRET", "1")

INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
# Tensors live on the GPU only when kernels are really compiled.
DEVICE = torch.device("cuda") if HAS_CUDA and not INTERPRETED else torch.device("cpu")

requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="requires a CUDA device (none in this environment)")
requires_slow = pytest.mark.skipif(
    os.environ.get("RUN_SLOW") != "1" and not HAS_CUDA,
    reason="very slow on CPU (Triton interpreter); set RUN_SLOW=1 to run",
)


@pytest.fixture(scope="session")
def device() -> torch.device:
    return DEVICE


def pytest_report_header(config):
    mode = "compiled Triton on CUDA" if DEVICE.type == "cuda" else "TRITON_INTERPRET=1 (CPU interpreter)"
    return f"engine tests: {mode}; torch {torch.__version__}"
