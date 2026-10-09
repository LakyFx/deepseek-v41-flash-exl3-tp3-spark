"""Kernel dispatch for EXL3.

The fast path is a single fused op per linear layer,
``torch.ops.cuda_exl3_C.exl3_linear``, which internally:

1. transforms the activations once per shard -- reading ``x`` a single time and
   converting bf16 -> fp16 in the same pass, so no separate ``.to()`` kernels;
2. runs one GEMM covering every shard of a fused layer (qkv_proj, gate_up_proj)
   in one launch, with split-k when the shape needs more blocks to fill the GPU;
3. applies the output Hadamard and ``svh``, emitting the caller's activation
   dtype directly.

It is registered through ``torch.library`` (not raw pybind) so that Dynamo can
trace it, which is what makes vLLM's CUDA graph capture work.

``CUDA_EXL3_BACKEND=exllamav3`` swaps in upstream's kernels as a correctness
oracle and performance baseline.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import torch
from cuda_exl3 import env as _env

_BACKEND_ENV = _env.getenv("CUDA_EXL3_BACKEND", "auto").lower()

# Split-k reduces partial sums with fp32 atomics, so results are reproducible in
# value but not bit-exact across runs (the summation order varies). Set
# CUDA_EXL3_DETERMINISTIC=1 to disable it: slower for small batches and narrow
# layers, bit-exact everywhere.
DETERMINISTIC = _env.getenv("CUDA_EXL3_DETERMINISTIC", "0") not in ("0", "", "false")

_native: Any = None
_exl: Any = None
_backend: str | None = None


def _try_native():
    global _native
    if _native is None:
        try:
            from cuda_exl3 import _C  # noqa: F401  (registers torch.ops.cuda_exl3_C)

            _native = torch.ops.cuda_exl3_C
        except Exception:
            _native = False
    return _native or None


def _try_exllamav3():
    global _exl
    if _exl is None:
        try:
            from exllamav3.ext import exllamav3_ext as ext  # type: ignore

            _exl = ext
        except Exception:
            _exl = False
    return _exl or None


def backend() -> str:
    """Name of the active kernel backend."""
    global _backend
    if _backend is not None:
        return _backend
    if _BACKEND_ENV == "native":
        if _try_native() is None:
            raise RuntimeError("CUDA_EXL3_BACKEND=native but cuda_exl3._C failed to import")
        _backend = "native"
    elif _BACKEND_ENV == "exllamav3":
        if _try_exllamav3() is None:
            raise RuntimeError("CUDA_EXL3_BACKEND=exllamav3 but exllamav3_ext failed to import")
        _backend = "exllamav3"
    else:
        if _try_native() is not None:
            _backend = "native"
        elif _try_exllamav3() is not None:
            _backend = "exllamav3"
        else:
            raise RuntimeError(
                "cuda-exl3: no kernel backend available. Build this package's CUDA "
                "extension, or install exllamav3."
            )
    return _backend


def exl3_linear(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    group_n: Sequence[int],
    cb: int,
) -> torch.Tensor:
    """``out = had(x * suh) @ dequant(trellis) * svh``, one call per layer.

    ``x``       (..., k) fp16 or bf16
    ``trellis`` (k/16, n_total/16, 16*bits) int16 -- covers every shard
    ``suh``     (groups, k) fp16 -- one row per shard
    ``svh``     (n_total,) fp16
    ``group_n`` output width of each shard, summing to n_total

    Returns (..., n_total) in ``x``'s dtype.
    """
    if backend() == "native":
        if not x.is_contiguous():
            x = x.contiguous()
        return torch.ops.cuda_exl3_C.exl3_linear(
            x, trellis, suh, svh, list(group_n), cb, not DETERMINISTIC
        )
    return _exl_reference(x, trellis, suh, svh, group_n, cb)


def _exl_reference(x, trellis, suh, svh, group_n, cb):
    """Upstream exllamav3 kernels, one shard at a time. Oracle / baseline only."""
    ext = _try_exllamav3()
    k = x.shape[-1]
    m = x.numel() // k
    n_total = trellis.shape[1] * 16
    x2 = x.reshape(m, k)
    xh = x2.half().contiguous()
    scratch = torch.empty_like(xh)
    parts = []
    off = 0
    for g, n in enumerate(group_n):
        t = trellis[:, off // 16 : (off + n) // 16, :].contiguous()
        o = torch.empty((m, n), dtype=torch.half, device=x.device)
        ext.exl3_gemm(xh, t, o, suh[g].contiguous(), scratch,
                      svh[off : off + n].contiguous(), -1, cb == 1, cb == 2, 0)
        parts.append(o)
        off += n
    out = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
    return out.to(x.dtype).view(*x.shape[:-1], n_total)


def reserve(like: torch.Tensor, max_tokens: int, k: int, n: int, groups: int) -> None:
    """Pre-size the shared workspaces for the largest forward this model can issue.

    Must happen before vLLM captures any CUDA graph: growing a workspace later
    replaces the allocation, and a captured graph would keep writing to the old
    pointer -- which the caching allocator has by then handed to someone else.
    """
    if backend() == "native":
        torch.ops.cuda_exl3_C.exl3_reserve(like, int(max_tokens), int(k), int(n),
                                           int(groups))


def _register_fake():
    """Shape function so Dynamo can trace the op without running it."""
    if not hasattr(torch.library, "register_fake"):
        return
    try:

        @torch.library.register_fake("cuda_exl3_C::exl3_linear")
        def _(x, trellis, suh, svh, group_n, cb, split_k):
            shape = list(x.shape)
            shape[-1] = trellis.shape[1] * 16
            return x.new_empty(shape)

    except Exception:
        # Already registered (the module can be imported in several processes).
        pass


if _try_native() is not None:
    _register_fake()
