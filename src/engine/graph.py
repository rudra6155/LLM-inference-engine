"""Decode loop with the token and position held on the device, optionally replayed as a CUDA graph.

Why: profiling showed that 81% of an eager decode step is host-side launch overhead (89.76 ms
eager vs 17.02 ms for the same step replayed from a CUDA graph, ``results/run_b``). A graph removes
that cost, but it freezes every kernel argument, so the per-token state (the input token and the
position) has to live in device memory that the graph reads and updates itself:

* ``tok``  int64 ``[1]``  input token; the graph overwrites it with ``argmax(logits)``
* ``pos``  int64 ``[1]``  position being processed; the graph increments it

``DeviceDecoder`` allocates those buffers, runs warm-up steps on a side stream (Triton compiles,
cuBLAS initialises), captures one decode step, and **checks the first replay against an eager step**
before it will be used. If anything fails - capture, the check, or a missing GPU - it records why
and reports ``usable = False`` so the caller falls back to the eager loop; nothing is hidden.

``use_graph=False`` runs the same device-position step eagerly (no capture). It exists for the CPU
tests and to separate "position on the device" from "replayed as a graph" when measuring.
"""

from __future__ import annotations

import torch

from .model import DevicePos, QwenEngineModel


class GraphCaptureError(RuntimeError):
    """The CUDA graph could not be captured, or its first replay disagreed with an eager step."""


def verify_replay(expected_logits: torch.Tensor, expected_tok: torch.Tensor,
                  logits: torch.Tensor, tok: torch.Tensor) -> None:
    """Raise ``GraphCaptureError`` if a replayed step disagrees with an eager step from the same state.

    The tolerance is fp16-sized, not bit-exact: captured and eager cuBLAS calls may pick different (equally valid)
    algorithms, which moves a logit by one fp16 step (about 0.016 at magnitude 16). What this check exists to catch -
    a frozen position, a stale constant - changes the logits by O(1). A token mismatch is an error only when the
    eager top-2 gap is not a near-tie, because inside a near-tie it is rounding, not a bug.
    """
    exp = expected_logits.float()
    diff = (logits.float() - exp).abs().max().item()
    tol = 0.05 + 2e-3 * exp.abs().max().item()
    if not diff <= tol:
        raise GraphCaptureError(f"self-check: replayed logits differ from eager by {diff:.3g} (tolerance {tol:.3g})")
    if not torch.equal(tok, expected_tok):
        top2 = torch.topk(exp, 2).values
        if float(top2[0] - top2[1]) > 0.1:
            raise GraphCaptureError(
                f"self-check: replay produced token {int(tok.item())}, eager produced {int(expected_tok.item())}")


class DeviceDecoder:
    def __init__(self, model: QwenEngineModel, cache, *, use_graph: bool = True, warmup: int = 3,
                 self_check: bool = True):
        self.model, self.cache = model, cache
        self.use_graph, self.warmup, self.self_check = use_graph, warmup, self_check
        self.mode = "new"        # new | graph | device-eager | fallback
        self.error: str | None = None
        self._graph = None
        self._logits: torch.Tensor | None = None
        self._tok = self._pos = None

    # -- state ---------------------------------------------------------------- #
    @property
    def usable(self) -> bool:
        return self.mode in ("graph", "device-eager")

    def describe(self) -> str:
        """One string for the benchmark's ``notes`` column."""
        if self.mode == "graph":
            return "cuda_graph=on"
        if self.mode == "device-eager":
            return "cuda_graph=off(device-position eager)"
        if self.mode == "fallback":
            return f"cuda_graph=FALLBACK to eager loop: {self.error}"
        return "cuda_graph=not initialised"

    def _reset_state(self) -> None:
        self._tok.zero_()
        self._pos.zero_()

    # -- one decode step, written so that every kernel argument is a tensor or a constant ---- #
    def _body(self) -> torch.Tensor:
        dpos = DevicePos.from_tensor(self._pos)
        logits = self.model.decode_step_device(self._tok, self.cache, dpos)
        self._tok.copy_(torch.argmax(logits).reshape(1))   # next step's input, produced on the device
        self._pos.add_(1)
        return logits

    # -- setup ------------------------------------------------------------------------------ #
    def ensure_ready(self) -> None:
        """Idempotent. May take a few seconds (warm-up + capture), so call it before timing."""
        if self.mode != "new":
            return
        device = self.model.device
        self._tok = torch.zeros(1, dtype=torch.long, device=device)
        self._pos = torch.zeros(1, dtype=torch.long, device=device)
        if not self.use_graph:
            self.mode = "device-eager"
            return
        try:
            self._capture()
            self.mode = "graph"
        except Exception as e:  # noqa: BLE001 - any failure must fall back, with its reason recorded
            self.error = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"[:300]
            self.mode, self._graph, self._logits = "fallback", None, None
            if device.type == "cuda":
                try:
                    torch.cuda.synchronize()
                except Exception:  # noqa: BLE001
                    pass

    def _capture(self) -> None:
        if self.model.device.type != "cuda":
            raise GraphCaptureError("CUDA graphs need a CUDA device")
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm-up off the default stream, as PyTorch requires before capture
            for _ in range(self.warmup):
                self._reset_state()
                self._body()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        self._reset_state()
        # thread_local: other threads (the benchmark's NVML memory monitor) must not trip the capture.
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            logits = self._body()
        torch.cuda.synchronize()
        self._graph, self._logits = graph, logits
        if self.self_check:
            self._check_first_replay()

    def _check_first_replay(self) -> None:
        """Replay once from a known state and compare with an eager step from the same state."""
        self._reset_state()
        self._pos.fill_(1)
        expected_logits = self._body().clone()          # eager; also advances tok / pos
        expected_tok = self._tok.clone()
        self._reset_state()
        self._pos.fill_(1)
        self._graph.replay()
        torch.cuda.synchronize()
        verify_replay(expected_logits, expected_tok, self._logits, self._tok)

    # -- use ----------------------------------------------------------------------------------- #
    def start(self, token: torch.Tensor, pos: int) -> None:
        """Begin decoding: ``token`` is the last generated token, ``pos`` its position."""
        if not self.usable:
            raise RuntimeError(f"decoder not usable ({self.describe()})")
        if not 0 <= pos < self.cache.max_seq_len:
            raise ValueError(f"position {pos} outside [0, {self.cache.max_seq_len})")
        self._tok.copy_(token.reshape(1))
        self._pos.fill_(pos)

    def step(self) -> int:
        """Process the current token, advance the state on the device, return the next token id."""
        if self.mode == "graph":
            self._graph.replay()
        else:
            self._body()
        return int(self._tok.item())   # the host copy is the per-token synchronization point
