"""PREPARED ONLY: temporary projection hooks inside existing FlashInfer autotune.

No hooks survive this context, no KV writes, no new inference entrypoint.
Pinned to the inspected DeepSeek V4.1 implementation; disabled by default.
"""
from contextlib import contextmanager
import logging
import os

LOG = logging.getLogger(__name__)


def _pre_hook(context, capturing, tuning, counts, name):
    def hook(module, args, kwargs):
        if not tuning() or capturing():
            raise RuntimeError("Indexer warmup must run during autotune, outside capture")
        metadata = context().attn_metadata
        if not isinstance(metadata, dict):
            raise RuntimeError("Missing real dummy attention metadata")
        item = metadata[module.k_cache.prefix]
        # Match the production forward's early-return condition exactly.
        if item.max_seq_len // module.compress_ratio <= module.topk_tokens:
            qr = kwargs["qr"] if "qr" in kwargs else args[0]
            qr_scale = kwargs.get("qr_scale", args[5] if len(args) > 5 else None)
            # Includes the actual quantized activation/scales and weight layout.
            # Discard only the dummy projection output; never change forward args.
            module._wq_b_proj(qr, qr_scale)
            counts[name] += 1
        return None
    return hook


@contextmanager
def _installed_hooks(modules, factory):
    handles = []
    try:
        for name, module in modules:
            handles.append(module.register_forward_pre_hook(factory(name), with_kwargs=True))
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()


@contextmanager
def indexer_projection_warmup(runner, world):
    import torch
    import torch.distributed as dist

    def consensus(value):
        values = [None] * world.world_size
        if world.world_size > 1:
            dist.all_gather_object(values, value, group=world.cpu_group)
        else:
            values = [value]
        if any(v != value for v in values):
            raise RuntimeError("Indexer warmup configuration differs across ranks")
        return value

    enabled = os.environ.get("DGX_INDEXER_PROJECTION_WARMUP", "0") == "1"
    consensus(enabled)
    if not enabled:
        yield
        return

    from flashinfer.autotuner import AutoTuner
    from vllm.forward_context import get_forward_context
    from vllm.models.deepseek_v4_1.attention import DeepseekV4Indexer

    modules = [(name, m) for name, m in runner.get_model().named_modules()
               if isinstance(m, DeepseekV4Indexer)]
    signature = [(name, m.q_lora_rank, m.n_head, m.head_dim, m.compress_ratio,
                  m.topk_tokens) for name, m in modules]
    consensus(signature)
    if not modules or any((m.q_lora_rank, m.n_head, m.head_dim) != (1280, 32, 128)
                          for _, m in modules):
        raise RuntimeError("Unexpected indexer inventory; re-audit warmup")
    if not AutoTuner.get().is_tuning_mode:
        raise RuntimeError("Warmup scope must be inside FlashInfer autotune context")
    counts = {name: 0 for name, _ in modules}
    factory = lambda name: _pre_hook(get_forward_context,
                                    torch.cuda.is_current_stream_capturing,
                                    lambda: AutoTuner.get().is_tuning_mode,
                                    counts, name)
    with _installed_hooks(modules, factory):
        yield
    consensus(counts)
    if not all(counts.values()):
        raise RuntimeError("Some indexer projections were not reached during warmup")
    LOG.info("DGX indexer projection warmup complete; hooks removed; calls=%s", counts)
