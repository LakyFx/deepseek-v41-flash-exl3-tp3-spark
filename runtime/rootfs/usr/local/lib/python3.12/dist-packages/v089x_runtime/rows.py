"""Per-forward prefill row distribution, independent of EP and static SP.

The target retains its full-row attention and EXL3 GEMMs. Their final TP
reduction becomes reduce-scatter; residuals, mHC and norms remain rank-local.
Decode, mixed batches, PP, DP, multimodal and capture use the original path.
"""
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import os

_ROWS = ContextVar('dgx_v089x_target_rows', default=None)

@dataclass(frozen=True)
class RowPartition:
    total: int
    world: int
    rank: int

    def __post_init__(self):
        if self.total < 1 or self.world < 1 or not 0 <= self.rank < self.world:
            raise ValueError('invalid row partition')

    @property
    def width(self):
        return (self.total + self.world - 1) // self.world

    @property
    def start(self):
        return self.rank * self.width

    @property
    def real(self):
        return max(0, min(self.width, self.total - self.start))

    def intersection(self, start, end):
        if not 0 <= start <= end <= self.total:
            raise ValueError('chunk outside token batch')
        first = max(start, self.start)
        last = min(end, self.start + self.real)
        return (first, max(first, last))

    def local(self, full, *, fill=0):
        if full.shape[0] != self.total:
            raise ValueError('full row shape changed inside forward')
        out = full.new_full((self.width, *full.shape[1:]), fill)
        if self.real:
            out[:self.real].copy_(full[self.start:self.start+self.real])
        return out

    def padded(self, full):
        if full.shape[0] != self.total:
            raise ValueError('reduction must contain the full token batch')
        if self.total == self.width * self.world:
            return full.contiguous()
        out = full.new_zeros((self.width*self.world, *full.shape[1:]))
        out[:self.total].copy_(full)
        return out

    def gather(self, local):
        from vllm.distributed import tensor_model_parallel_all_gather
        if local.shape[0] != self.width:
            raise ValueError('rank-local row shape mismatch')
        return tensor_model_parallel_all_gather(local.contiguous(), dim=0)[:self.total]

    def reduce_scatter(self, partial):
        from vllm.distributed import tensor_model_parallel_reduce_scatter
        return tensor_model_parallel_reduce_scatter(self.padded(partial), dim=0)

def current_rows():
    import torch
    # Make decode compilation specialize to the unmodified path. Dynamic SP
    # enters only eager target forwards dispatched by runner_prefill_eager.
    if torch.compiler.is_compiling():
        return None
    return _ROWS.get()

def runner_prefill_eager(runner, batch, scheduler, num_tokens, dummy):
    """CPU-only decision made before graph dispatch; identical across TP ranks."""
    if os.environ.get('DGX_V089X_SP') != '1' or dummy or batch is None:
        return False
    p = runner.vllm_config.parallel_config
    return bool(
        num_tokens >= int(os.environ.get('DGX_V089X_SP_MIN_ROWS', '2048'))
        and p.tensor_parallel_size == 3 and p.pipeline_parallel_size == 1
        and p.data_parallel_size == 1 and not p.enable_expert_parallel
        and getattr(p, 'prefill_context_parallel_size', 1) == 1
        and getattr(p, 'decode_context_parallel_size', 1) == 1
        and runner.lora_config is None
        and not scheduler.scheduled_encoder_inputs
        and len(batch.is_prefilling_np) > 0 and bool(batch.is_prefilling_np.all()))

def target_prefill_rows(fn):
    @wraps(fn)
    def call(self, input_ids, positions, intermediate_tensors, inputs_embeds=None,
             lookback_token_ids=None):
        import torch
        from vllm.forward_context import get_forward_context, is_forward_context_available
        from vllm.distributed import get_tensor_model_parallel_rank
        if torch.compiler.is_compiling():
            # No ContextVar.set in Dynamo-traced decode: that would introduce
            # a graph break into the baseline graph/capture path.
            return fn(self, input_ids, positions, intermediate_tensors, inputs_embeds,
                      lookback_token_ids)
        rows = None
        if (os.environ.get('DGX_V089X_SP') == '1' and not torch.compiler.is_compiling()
                and not torch.cuda.is_current_stream_capturing()
                and not self.use_sequence_parallel and not self.use_mega_moe
                and intermediate_tensors is None
                and (inputs_embeds is None or inputs_embeds.shape[0] == positions.shape[0])
                and positions.shape[0] >= int(os.environ.get('DGX_V089X_SP_MIN_ROWS', '2048'))
                and is_forward_context_available()):
            p = self.parallel_config
            context = get_forward_context()
            metadata = context.attn_metadata
            swa = metadata.get(self.engram_swa_prefix) if isinstance(metadata, dict) else None
            if (context.skip_compiled and getattr(context.cudagraph_runtime_mode, 'name', None) == 'NONE'
                    and p.tensor_parallel_size == 3 and p.pipeline_parallel_size == 1
                    and p.data_parallel_size == 1 and not p.enable_expert_parallel
                    and getattr(p, 'prefill_context_parallel_size', 1) == 1
                    and getattr(p, 'decode_context_parallel_size', 1) == 1
                    and swa is not None and swa.num_decodes == 0 and swa.num_prefills > 0):
                rows = RowPartition(positions.shape[0], 3, get_tensor_model_parallel_rank())
        token = _ROWS.set(rows)
        try:
            if rows is not None:
                from .telemetry import observe
                observe('sp_prefill', rows=rows.total)
            return fn(self, input_ids, positions, intermediate_tensors, inputs_embeds,
                      lookback_token_ids)
        finally:
            _ROWS.reset(token)
    return call

def attention_reduce(partial, layer, original):
    plan = current_rows()
    if plan is not None and layer.prefix.endswith('.attn.wo_b'):
        if layer.bias is not None or layer.tp_size != plan.world:
            raise RuntimeError('unsupported SP attention reduction contract')
        return plan.reduce_scatter(partial)
    return original(partial)

def moe_reduce_if_active(states, runner, output_is_reduced):
    """None means original late reduction; no mutable module reduction flags."""
    plan = current_rows()
    if plan is None:
        return None
    c = runner.moe_config
    if (c.tp_size != plan.world or c.ep_size != 1 or c.is_sequence_parallel
            or c.skip_final_all_reduce or runner.routed_output_transform is not None):
        raise RuntimeError('unsupported SP EXL3 MoE reduction contract')
    return plan.local(states) if output_is_reduced else plan.reduce_scatter(states)

def moe_result_view(states, original_shape):
    plan = current_rows()
    shape = (plan.width, *original_shape[1:]) if plan is not None else original_shape
    return states.view(shape)

def indexer_partition(metadata, total, *, use_pcp, dcp_world_size):
    if os.environ.get('DGX_V089X_INDEXER_SPLIT') != '1' or use_pcp or dcp_world_size != 1:
        return None
    plan = current_rows()
    if plan is None or plan.total != total or metadata.num_decodes or not metadata.num_prefills:
        return None
    chunks = metadata.prefill.chunks
    for chunk in chunks:
        length = chunk.token_end - chunk.token_start
        if chunk.cu_seqlen_ks.numel() != length or chunk.cu_seqlen_ke.numel() != length:
            raise RuntimeError('indexer bounds must have one value per query row')
    return plan

def gather_indexer_output(plan, topk, candidate_blocks, candidate_write):
    if plan is None:
        return
    gathered = plan.gather(plan.local(topk[:plan.total], fill=-1))
    topk[:plan.total].copy_(gathered)
    if candidate_blocks is not None and candidate_write:
        candidate_blocks[:plan.total].copy_(
            plan.gather(plan.local(candidate_blocks[:plan.total], fill=-1)))
    from .telemetry import observe
    observe('indexer_split', rows=plan.total)
