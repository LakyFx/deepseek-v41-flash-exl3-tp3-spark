"""P: bounded read-only L2 hints overlapped with selected target TP reductions.

No mutable CPU 'next window' state hidden inside compiled model graphs. One
opaque op owns issue->original TP collective->side-stream join, so CUDA graph
capture/replay retains both stream edges and the exact collective backend.
"""
from functools import cache
from dgx_campaign_runtime.x11_policy import prefetch_parameters as _x11_params
import os

_plans = {}
_contexts = {}


def enabled():
    value = os.environ.get("DGX_CAMPAIGN_P", "0")
    if value not in ("0", "1"):
        raise ValueError("DGX_CAMPAIGN_P must be 0 or 1")
    return value == "1"


def budget_segments(segments, budget):
    """Pure bounded geometry; address ownership is checked when collecting."""
    if budget < 0:
        raise ValueError("negative prefetch budget")
    seen, selected, remaining = set(), [], budget
    for name, ptr, size in segments:
        if ptr <= 0 or size < 0:
            raise ValueError("invalid prefetch address range")
        if ptr in seen or not size or not remaining:
            continue
        seen.add(ptr)
        count = min(size, remaining)
        selected.append((name, ptr, count))
        remaining -= count
    return selected


def _collect(owners):
    import torch
    segments, tensors, visited = [], [], set()

    def visit(name, value, depth):
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, torch.Tensor):
            if value.device.type != "cuda" or not value.numel():
                return
            if any(s < 0 for s in value.stride()):
                raise ValueError("negative strides are not qualified for L2 hints")
            span = (1 + sum((n - 1) * s for n, s in zip(value.shape, value.stride()))) * value.element_size()
            if span + value.storage_offset() * value.element_size() > value.untyped_storage().nbytes():
                raise ValueError("tensor span escapes its storage")
            segments.append((name, value.data_ptr(), span))
            tensors.append(value)
        elif depth and value is not None:
            if isinstance(value, dict):
                items = value.items()
            elif isinstance(value, (tuple, list)):
                items = enumerate(value)
            elif hasattr(value, "__dict__"):
                items = vars(value).items()
            else:
                items = ((s, getattr(value, s)) for s in getattr(type(value), "__slots__", ()) if hasattr(value, s))
            for key, child in items:
                # Packed tensor holders and module parameter dictionaries only;
                # don't walk other model children, devices or unrelated caches.
                if key in ("_modules", "_buffers"):
                    continue
                visit(f"{name}.{key}", child, depth - 1)

    for name, owner in owners:
        visit(name, owner, 4)
    return segments, tensors


def _plan(owners, budget, device):
    import torch
    segments, tensors = _collect(owners)
    chosen = budget_segments(segments, budget)
    if not chosen:
        return None
    descriptor = torch.tensor([(ptr, size) for _, ptr, size in chosen], dtype=torch.int64, device=device)
    key = id(descriptor)
    # Strong tensor references protect raw addresses through warmup and every
    # captured replay, even where a packed-weight holder has replaced params.
    _plans[key] = {"descriptor": descriptor, "owners": tensors, "segments": chosen, "device": device}
    return key


def prepare_target(model):
    if not enabled():
        return
    import torch
    from vllm.distributed import get_tensor_model_parallel_world_size
    if get_tensor_model_parallel_world_size() != 3 or torch.cuda.get_device_capability() != (12, 1):
        raise ValueError("P is prepared only for the locked SM121 TP3 target")
    layers = list(model.layers)
    for i, layer in enumerate(layers):
        device = layer.hc_ffn_fn.device
        shared = getattr(layer.ffn, "shared_experts", None)
        ff = _plan([("mhc_ffn", layer.hc_ffn_fn), ("router", getattr(layer.ffn, "gate", None)),
                    ("shared_up", getattr(shared, "gate_up_proj", None)),
                    ("shared_down", getattr(shared, "down_proj", None))], _x11_params(1, os.environ)[1], device)
        output = getattr(layer.attn, "wo_b", None)
        if output is None:
            raise ValueError("P attention output reduction owner changed")
        output._campaign_p_plan = ff
        nxt = layers[i + 1] if i + 1 < len(layers) else None
        following = None
        if nxt is not None:
            attn = nxt.attn
            indexer = getattr(attn, "indexer", None)
            following = _plan([("mhc_attn", nxt.hc_attn_fn),
                ("qkv_a", getattr(attn, "fused_wqa_wkv", None)),
                ("q_b", getattr(attn, "wq_b", None)),
                ("index_q", getattr(indexer, "wq_b", None)),
                ("index_k", getattr(indexer, "wk", None))], _x11_params(1, os.environ)[2], device)
        # The current FusedMoEFactory returns MoERunner; it owns the late
        # reduction. The decoder's outer FFN module does not perform it.
        layer.ffn.experts._campaign_p_plan = following
    if not _plans:
        raise ValueError("P selected no actual native tensor ranges")
    _op()  # register while eager, before torch.compile/capture
    for plan in _plans.values():
        device = plan["device"]
        if device not in _contexts:
            _contexts[device] = (torch.cuda.Stream(device=device, priority=0), torch.cuda.Event(), torch.cuda.Event())


def _execute(input, key):
    import torch
    from vllm.distributed import tensor_model_parallel_all_reduce
    from dgx_campaign_runtime.prefetch_kernel import hint
    plan = _plans[key]
    device = plan["device"]
    if device not in _contexts:
        raise RuntimeError("P stream context must be prepared before capture")
    side, begin, done = _contexts[device]
    main = torch.cuda.current_stream(device)
    begin.record(main)
    side.wait_event(begin)
    with torch.cuda.stream(side):
        hint(plan["descriptor"], ctas=_x11_params(input.shape[0], os.environ)[3])
        done.record(side)
    # Preserves R-vs-bulk dispatch; this is NOT a different collective engine.
    output = tensor_model_parallel_all_reduce(input)
    main.wait_event(done)
    return output


@cache
def _op():
    import torch
    @torch.library.custom_op("dgx_campaign_p_v1::reduce_with_hints", mutates_args=())
    def operation(input: torch.Tensor, key: int) -> torch.Tensor:
        return _execute(input, key)
    @operation.register_fake
    def fake(input, key):
        return torch.empty_like(input)
    return operation


def all_reduce(input, owner):
    from vllm.distributed import tensor_model_parallel_all_reduce
    key = getattr(owner, "_campaign_p_plan", None)
    # Large prefill takes the unchanged path; P is a decode experiment only.
    if not enabled() or key is None or not _x11_params(input.shape[0], os.environ)[0]:
        return tensor_model_parallel_all_reduce(input)
    return _op()(input, key)
