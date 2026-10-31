"""Unit tests: C++ RMSNorm vs the PyTorch reference (atol = 1e-4 for fp32)."""

import pytest
import torch

from engine.cpp_ops import load_custom_ops, rmsnorm, rmsnorm_reference

from .conftest import requires_cuda

SHAPES = [(1, 2048), (7, 2048), (1, 1, 2048), (3, 5, 77), (1, 11008)]


@pytest.fixture(scope="module", autouse=True)
def _build_extension():
    load_custom_ops()  # compiles on first use via torch.utils.cpp_extension


@pytest.mark.parametrize("shape", SHAPES)
def test_rmsnorm_fp32_matches_reference(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape)
    w = torch.randn(shape[-1])
    out = rmsnorm(x, w, 1e-6)
    ref = rmsnorm_reference(x, w, 1e-6)
    assert out.shape == x.shape and out.dtype == torch.float32
    assert torch.allclose(out, ref, atol=1e-4, rtol=0), (out - ref).abs().max()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
def test_rmsnorm_low_precision_matches_reference_within_rounding(dtype, shape):
    """atol=1e-4 is not attainable for fp16/bf16 outputs (their ulp is larger); the
    float32 accumulation order differs from torch's, which can flip one rounding.
    Tolerance here is 1e-3 + 1e-2 * |ref| for bf16 (8 significant bits) and
    1e-3 + 2e-3 * |ref| for fp16."""
    torch.manual_seed(1)
    x = torch.randn(*shape).to(dtype)
    w = torch.randn(shape[-1]).to(dtype)
    out = rmsnorm(x, w, 1e-6)
    ref = rmsnorm_reference(x, w, 1e-6)
    assert out.dtype == dtype
    rtol = 1e-2 if dtype == torch.bfloat16 else 2e-3
    assert torch.allclose(out.float(), ref.float(), atol=1e-3, rtol=rtol)


def test_rmsnorm_handles_non_contiguous_input():
    torch.manual_seed(2)
    x = torch.randn(2048, 4).t()  # [4, 2048], non-contiguous
    w = torch.randn(2048)
    assert torch.allclose(rmsnorm(x, w), rmsnorm_reference(x, w), atol=1e-4, rtol=0)


def test_rmsnorm_rejects_mismatched_arguments():
    x = torch.randn(2, 8)
    with pytest.raises(RuntimeError):
        rmsnorm(x, torch.randn(9))
    with pytest.raises(RuntimeError):
        rmsnorm(x, torch.randn(8, dtype=torch.float64))
    with pytest.raises(RuntimeError):
        rmsnorm(x, torch.randn(8, 1))


@requires_cuda
@pytest.mark.parametrize("shape", [(1, 2048), (7, 2048)])
def test_rmsnorm_cuda_tensor_path_matches_reference(shape):
    torch.manual_seed(3)
    x = torch.randn(*shape, device="cuda")
    w = torch.randn(shape[-1], device="cuda")
    assert torch.allclose(rmsnorm(x, w), rmsnorm_reference(x, w), atol=1e-4, rtol=0)
