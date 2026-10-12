"""N2 small-decode mHC dispatcher, opaque to Dynamo, phase-aware at capture.

The matching runner overlay forces tiny prefill/mixed batches eager so they
cannot replay a decode-specialized graph. Larger shapes use unchanged X11c.
"""
from functools import cache


@cache
def _op():
    import torch

    @torch.library.custom_op('dgx_next_mhc_v1::post_pre', mutates_args=())
    def operation(x: torch.Tensor, residual: torch.Tensor, previous_post: torch.Tensor,
                  previous_comb: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor,
                  base: torch.Tensor, incoming: torch.Tensor, norm: torch.Tensor,
                  rms_eps: float, pre_eps: float, sinkhorn_eps: float,
                  post_mult: float, iterations: int, norm_eps: float
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from vllm.forward_context import get_forward_context, is_forward_context_available
        from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang, mhc_pre_delayed_tilelang
        from dgx_next_runtime.policy import small_decode_mhc
        from dgx_next_runtime.mhc_original import post_pre as original
        metadata = get_forward_context().attn_metadata if is_forward_context_available() else None
        if small_decode_mhc(residual.shape[0], metadata):
            updated = mhc_post_tilelang(x, residual, previous_post, previous_comb)
            post, comb, y, premix = mhc_pre_delayed_tilelang(
                updated, fn, scale, base, rms_eps, pre_eps, sinkhorn_eps,
                post_mult, iterations, pre_mix=incoming, norm_weight=norm,
                norm_eps=norm_eps)
            return updated, post, comb, y, premix
        return original(x, residual, previous_post, previous_comb, fn, scale, base,
                        rms_eps, pre_eps, sinkhorn_eps, post_mult, iterations,
                        pre_mix=incoming, norm_weight=norm, norm_eps=norm_eps)

    @operation.register_fake
    def fake(x, residual, previous_post, previous_comb, fn, scale, base, incoming,
             norm, rms_eps, pre_eps, sinkhorn_eps, post_mult, iterations, norm_eps):
        n, streams, hidden = residual.shape
        return (torch.empty_like(residual),
                torch.empty((n, streams, 1), dtype=torch.float32, device=residual.device),
                torch.empty((n, streams, streams), dtype=torch.float32, device=residual.device),
                torch.empty((n, hidden), dtype=residual.dtype, device=residual.device),
                torch.empty((n, streams), dtype=torch.float32, device=residual.device))
    return operation


def enabled():
    from dgx_next_runtime.mhc_original import enabled as original
    return original()


def post_pre(x, residual, previous_post, previous_comb, fn, scale, base,
             rms_eps, pre_eps, sinkhorn_eps, post_mult, iterations,
             *, pre_mix, norm_weight, norm_eps):
    from dgx_next_runtime.mhc_original import post_pre as original
    if residual.shape[0] > 6 or residual.shape[0] == 0:
        return original(x, residual, previous_post, previous_comb, fn, scale, base,
                        rms_eps, pre_eps, sinkhorn_eps, post_mult, iterations,
                        pre_mix=pre_mix, norm_weight=norm_weight, norm_eps=norm_eps)
    if pre_mix is None or norm_weight is None or iterations != 20:
        raise ValueError('N2 requires original lagged mix and 20-iteration mHC')
    return _REGISTERED(x, residual, previous_post, previous_comb, fn, scale, base,
                       pre_mix, norm_weight, rms_eps, pre_eps, sinkhorn_eps,
                       post_mult, iterations, norm_eps)


# Register before Dynamo traces a small shape. Registration allocates no GPU
# tensors and compiles no kernels; preparation only parses this runtime file.
_REGISTERED = _op()
