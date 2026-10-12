# SPDX-License-Identifier: Apache-2.0
"""X11c row-partition transition used only by the experimental N6 backport.

The prefix remains sequence-parallel. Before the compact tail, every TP rank
recovers identical full-row states. Tail attention/MoE then uses ordinary TP
all-reduce. No local row index is ever applied to another rank's shard.
"""
from contextlib import contextmanager


@contextmanager
def replicated_rows():
    from v089x_runtime.rows import _ROWS
    token = _ROWS.set(None)
    try:
        yield
    finally:
        _ROWS.reset(token)


def enter_tail(model, plan, hidden, positions, input_ids, pre, post, mix, residual):
    # Same collective order and same optional-state branches on all three ranks.
    if plan is not None:
        hidden, pre, post, mix, residual = tuple(
            None if state is None else plan.gather(state)
            for state in (hidden, pre, post, mix, residual)
        )
    with replicated_rows():
        return model.decoder_replay_layers(
            hidden, positions, input_ids, pre, post, mix, residual
        )


def in_piecewise_cudagraph():
    # The pinned old fork exposes the mode, but not the newer convenience fn.
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available
    return (is_forward_context_available()
            and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE)


def reset_after_profiling(state):
    """Discard metadata bound to the profiler's temporary KV configuration.

    Called only from profiling teardown after the generic wrapper graph clear.
    Model weights, Engram staging and static input buffers remain owned by the
    state. The next prepare_attn rebuilds group IDs and slot mappings against
    the real cache; real capture fills fresh graph entries.
    """
    if not hasattr(state, '_replay'):
        return
    state._replay = None
    layers = getattr(state, 'decoder_replay_layers', None)
    if layers is not None:
        state._kept_slot_mappings = None
        state._replay_attn_groups = None
        layers.replay_batch = None
        layers._break_outputs = None
        manager = getattr(state, 'replay_cudagraphs', None)
        if manager is not None:
            manager._graphs_captured = False
            manager.breakable_cg_runner.clear_graphs()


def write_kv(self, x, positions, input_ids, pre_mix, post_mix, res_mix, residual):
    """Same old-fork mHC as forward, without the cut layer's query/FFN."""
    from dgx_campaign_runtime import mhc
    from v089x_runtime.rows import current_rows
    if self.engram is not None or self.use_sequence_parallel or current_rows() is not None:
        raise RuntimeError('cut-layer KV writer requires replicated rows and no Engram')
    if residual is None:
        raise RuntimeError('replay cut must follow at least one complete layer')
    _, _, _, normalized, _ = mhc.post_pre(
        x, residual, post_mix, res_mix, self.hc_attn_fn,
        self.hc_attn_scale, self.hc_attn_base, self.rms_norm_eps,
        self.hc_eps, self.hc_eps, self.hc_post_alpha, self.hc_sinkhorn_iters,
        pre_mix=pre_mix, norm_weight=self.attn_norm.weight,
        norm_eps=self.attn_norm.variance_epsilon,
    )
    self.attn.forward_kv(positions, normalized)


def forward_kv(self, positions, hidden_states):
    """Write KV with the existing fp8_ds_mla insert, not a missing new C++ op.

    The old DSpark helper passes a dummy query to the fused insert. This is
    more work than the new dedicated KV-only op, but preserves the loaded
    extension ABI and the current quantization/layout. Measure that overhead.
    """
    from vllm.forward_context import get_forward_context
    from vllm.models.deepseek_v4_1.nvidia.dspark import _insert_context_kv
    metadata = get_forward_context().attn_metadata
    qr_kv, kv_score, _ = self._run_parallel_input_projections(hidden_states)
    _, _, kv = self._split_qkv_and_norm(qr_kv)
    if isinstance(metadata, dict):
        _insert_context_kv(self, kv, positions,
                           metadata[self.swa_cache_layer.prefix].slot_mapping)
    if self.compressor is not None:
        latent = self.compressor(kv_score, positions)
        self.compressor.insert_cache(latent, positions, self.rotary_emb)
        if self.indexer is not None and self.indexer.owns_k:
            self.indexer._produce_k(latent, positions, self.indexer_rotary_emb)
