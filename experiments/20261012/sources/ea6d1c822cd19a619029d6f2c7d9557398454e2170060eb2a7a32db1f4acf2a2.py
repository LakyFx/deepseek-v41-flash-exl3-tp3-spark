"""N6's explicit model and inner-graph contract; no CUDA/model imports."""

INNER_SIZES = [128, 256, 384, 512, 768, 1024]
COMPILATION_DELTA = {
    'cudagraph_decoder_replay': True,
    'decoder_replay_cudagraph_capture_sizes': INNER_SIZES,
    'decoder_replay_trim_threshold': 768,
}


def validate_model(config, draft, parallel, cut, cache):
    """Fail before benchmarking if this is not the audited model topology.

    The old DSpark config retains the full target's layer count and appends
    three uncompressed SWA-only draft stages. It has no layer_types attribute;
    the compress_ratios are the authoritative per-layer attention selection.
    """
    if (parallel.tensor_parallel_size != 3 or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1 or parallel.enable_expert_parallel
            or parallel.prefill_context_parallel_size != 1 or parallel.use_ubatching):
        raise RuntimeError('N6 requires the reviewed TP3/DP1/PP1 non-PCP topology')
    if not cache.swa_bounded_replay or config.sliding_window != 128 or cut != 20:
        raise RuntimeError('N6 replay must be active at window128/cut20')
    if (config.num_hidden_layers != 40 or config.hidden_size != 5120
            or tuple(config.kv_source_layer_ids) != (2, 8, 14, 20)
            or tuple(config.index_source_layer_ids) != (2, 8, 14, 20, 24, 28, 32, 36)
            or tuple(config.engram_layer_ids) != (1, 14)):
        raise RuntimeError('N6 target topology changed; re-audit required')
    for model in (config, draft):
        if (model is None or model.sliding_window != 128
                or model.num_hidden_layers != 40
                or model.num_nextn_predict_layers != 3
                or tuple(model.dspark_target_layer_ids) != (37, 38, 39)
                or tuple(model.compress_ratios) != (0, 0) + (2,)*18 + (1,)*20 + (0,)*3):
            raise RuntimeError('N6 target/draft attention window or late-hidden contract changed')
    if getattr(draft, 'layer_types', None) not in (None, [], ()):
        if any(t != 'sliding_attention' for t in draft.layer_types):
            raise RuntimeError('N6 draft reads outside the retained sliding window')


def validate_inner_graphs(compilation, max_num_tokens, max_num_reqs):
    if (compilation.cudagraph_decoder_replay is not True
            or list(compilation.decoder_replay_cudagraph_capture_sizes) != INNER_SIZES
            or compilation.decoder_replay_trim_threshold != 768
            or max_num_reqs != 8 or max_num_tokens != 4096):
        raise RuntimeError('N6 inner graph plan differs from the prepared memory bound')
