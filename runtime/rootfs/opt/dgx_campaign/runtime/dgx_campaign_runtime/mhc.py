# SPDX-License-Identifier: Apache-2.0
"""M: actual lagged post-to-next-pre fusion; Engram injection stays between.

This is distinct from the rejected narrow prenorm tuning. Same four streams,
previous pre-mix, original eps/20 Sinkhorn iterations and BF16 rounding points.
"""
from functools import cache
import os


def enabled():
    value = os.environ.get("DGX_CAMPAIGN_M", "0")
    if value not in ("0", "1"):
        raise ValueError("DGX_CAMPAIGN_M must be 0 or 1")
    return value == "1"


@cache
def _kernels(hidden, rms_eps, hc_eps, iterations, norm_eps, post_mult):
    from dgx_campaign_runtime.mhc_kernels import project_streams, finalize, BLOCK_M
    return (project_streams(hidden, "post_pre", block_M=BLOCK_M),
            finalize(hidden, 4 * hidden, rms_eps, hc_eps, iterations, norm_eps, post_mult))


@cache
def _buffers(device,hidden):
    # One strong owner per target device, reserved during eager profile.
    # Keep flat storage so every sliced runtime tensor is contiguous. The
    # graph can safely reference it even if another workspace lane grows.
    import torch
    if hidden!=5120:raise ValueError("M hidden width differs from locked target")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("M scratch must be reserved before graph capture")
    splits,maximum=hidden//128,4096
    return (torch.empty(splits*maximum*24,dtype=torch.float32,device=device),
            torch.empty(splits*maximum,dtype=torch.float32,device=device),
            torch.empty(maximum*hidden,dtype=torch.bfloat16,device=device),
            torch.empty(splits*maximum,dtype=torch.float32,device=device))


def _scratch(device,tokens,hidden):
    if not 0<tokens<=4096:raise ValueError("M rows exceed locked prefill chunk")
    splits=hidden//128
    p,s,c,q=_buffers(device,hidden)
    return (p[:splits*tokens*24].view(splits,tokens,1,24),
            s[:splits*tokens].view(splits,tokens,1),
            c[:tokens*hidden].view(tokens,hidden),
            q[:splits*tokens].view(splits,tokens))


def _execute(x, residual, previous_post, previous_comb, fn, scale, base, norm, incoming,
             residual_out, y, post, comb, pre_out, rms_eps, hc_eps, iterations, norm_eps, post_mult):
    import torch
    tokens, _, hidden = residual.shape
    if not tokens:
        return
    projection, finalize = _kernels(hidden, rms_eps, hc_eps, iterations, norm_eps, post_mult)
    p,s,collapsed,squares=_scratch(residual.device,tokens,hidden)
    projection(x, residual.view(tokens, 4 * hidden), previous_post.view(tokens, 4),
        previous_comb, incoming, fn, residual_out.view(tokens, 4 * hidden), p, s, collapsed, squares)
    finalize(p, s, collapsed, squares, scale, base, norm, post, comb, pre_out, y)


@cache
def _op():
    # Registered lazily on the first eager profiling pass. No import-time GPU
    # compilation and no replacement of any stock custom-op name.
    import torch

    @torch.library.custom_op("dgx_campaign_m_v1::post_pre", mutates_args=("residual_out", "y", "post", "comb", "pre_out"))
    def operation(x: torch.Tensor, residual: torch.Tensor, previous_post: torch.Tensor,
                  previous_comb: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor,
                  base: torch.Tensor, norm: torch.Tensor, incoming: torch.Tensor,
                  residual_out: torch.Tensor, y: torch.Tensor, post: torch.Tensor,
                  comb: torch.Tensor, pre_out: torch.Tensor, rms_eps: float, hc_eps: float,
                  iterations: int, norm_eps: float, post_mult: float) -> None:
        _execute(x, residual, previous_post, previous_comb, fn, scale, base, norm, incoming,
            residual_out, y, post, comb, pre_out, rms_eps, hc_eps, iterations, norm_eps, post_mult)

    @operation.register_fake
    def fake(*args, **kwargs):
        return None
    return operation


def post_pre(x, residual, previous_post, previous_comb, fn, scale, base,
             rms_eps, pre_eps, sinkhorn_eps, post_mult, iterations,
             *, pre_mix, norm_weight, norm_eps):
    import torch
    from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang, mhc_pre_delayed_tilelang
    if not enabled():
        updated = mhc_post_tilelang(x, residual, previous_post, previous_comb)
        post, comb, y, premix = mhc_pre_delayed_tilelang(updated, fn, scale, base,
            rms_eps, pre_eps, sinkhorn_eps, post_mult, iterations, pre_mix=pre_mix,
            norm_weight=norm_weight, norm_eps=norm_eps)
        return updated, post, comb, y, premix
    tokens, streams, hidden = residual.shape
    if streams != 4 or hidden != 5120 or pre_eps != sinkhorn_eps or iterations != 20:
        raise ValueError("M requires the locked DS4.1 four-stream/20-Sinkhorn contract")
    if pre_mix is None or norm_weight is None or residual.dtype != torch.bfloat16 or not residual.is_contiguous():
        raise ValueError("M post-to-pre requires incoming lagged mix, BF16 contiguous streams and fused norm")
    if fn.shape != (24, 4 * hidden) or fn.dtype != torch.float32:
        raise ValueError("M projection format changed")
    # Compile outside the opaque custom op during eager profile, never inside
    # graph capture. Cache identity includes EVERY numerical constexpr.
    _kernels(hidden, rms_eps, pre_eps, iterations, norm_eps, post_mult)
    if tokens:_buffers(residual.device,hidden)
    updated = torch.empty_like(residual)
    y = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=residual.device)
    post = torch.empty((tokens, 4), dtype=torch.float32, device=residual.device)
    comb = torch.empty((tokens, 4, 4), dtype=torch.float32, device=residual.device)
    premix = torch.empty((tokens, 4), dtype=torch.float32, device=residual.device)
    _op()(x, residual, previous_post, previous_comb, fn, scale, base, norm_weight, pre_mix,
        updated, y, post, comb, premix, rms_eps, pre_eps, iterations, norm_eps, post_mult)
    return updated, post.unsqueeze(-1), comb, y, premix
