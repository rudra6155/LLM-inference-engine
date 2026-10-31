"""JIT build / load of the C++ extension in ``src/csrc`` via ``torch.utils.cpp_extension``."""

from __future__ import annotations

import platform
from pathlib import Path

import torch

_CSRC = Path(__file__).resolve().parents[1] / "csrc"
_module = None


def _x86_cflags() -> list[str]:
    """Enable hardware fp16 <-> fp32 conversion (F16C) so c10::Half is not converted in
    software. Only on x86-64 CPUs that report the features (the extension is JIT-built
    on the machine that runs it)."""
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return []
    try:
        flags = next(l for l in open("/proc/cpuinfo") if l.startswith("flags")).split()
    except (OSError, StopIteration):
        return []
    return ["-mavx2", "-mf16c", "-mfma"] if {"avx2", "f16c", "fma"} <= set(flags) else []


def load_custom_ops(verbose: bool = False):
    """Compile (first call only; cached by torch) and import the extension."""
    global _module
    if _module is None:
        from torch.utils.cpp_extension import load

        _module = load(
            name="engine_custom_ops",
            sources=[str(_CSRC / "custom_ops.cpp"), str(_CSRC / "binding.cpp")],
            extra_cflags=["-O3", *_x86_cflags()],
            verbose=verbose,
        )
    return _module


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """C++ RMSNorm (see ``custom_ops.cpp`` for what runs on which device)."""
    return load_custom_ops().rmsnorm(x, weight, eps)


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """PyTorch reference: HuggingFace Qwen2RMSNorm."""
    h = x.to(torch.float32)
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + eps)
    return weight * h.to(x.dtype)
