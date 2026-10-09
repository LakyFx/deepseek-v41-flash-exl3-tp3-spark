"""I1/I2: compact FP8 candidate keys, reuse the installed SM12x MQA scorer.

No change to quantization, head count or score reduction. Native scoring uses
next_n=1 over per-query packed pages. Numerical/selector equivalence MUST be
qualified on GPU before serving; this file alone is not a readiness receipt.
Only the candidate-consumer decode route changes; source layers/prefill stay.
"""
from __future__ import annotations

import os
import functools
from dataclasses import dataclass


def mode() -> int:
    value = os.environ.get("DGX_CAMPAIGN_INDEXER", "0")
    if value not in ("0", "1", "2"):
        raise ValueError("DGX_CAMPAIGN_INDEXER must be 0, 1 or 2")
    return int(value)


@dataclass
class CandidateScores:
    logits: object
    positions: object | None = None
    lengths: object | None = None


@functools.lru_cache(maxsize=8)
def _sm_count(device):
    import torch
    return torch.cuda.get_device_properties(device).multi_processor_count


def reserve_profile(device, width):
    """Conservative simultaneous extra workspace reservation at stock profiling.

    The existing full-logits reservation remains in place. No KV bytes are
    borrowed or reduced; the normal startup allocation must still succeed.
    """
    if not mode():
        return None
    import torch
    rows, k = 48, 2048  # locked C8, targetK5+1 and candidate2048
    blocks = (width + 7) // 8
    capacity = ((min(width, k * 8) + 63) // 64) * 64
    extra = rows * (12 * (blocks + 1) + capacity * (132 + 4 + 4) + (capacity // 64) * 4 + 4)
    return torch.empty(extra, dtype=torch.uint8, device=device)


def score_decode(q, cache, weights, lengths, table, request_indices, candidates,
                 *, max_model_len: int, block_size: int, use_fp4: bool,
                 source_layer: bool, scorer, schedule_builder):
    selected = mode()
    if not selected or candidates is None or source_layer:
        return None
    import torch
    if use_fp4 or q[1] is not None:
        raise ValueError("candidate port was prepared for the locked FP8 indexer, not FP4")
    values = q[0]
    b, next_n, heads, dim = values.shape
    rows = b * next_n
    if (dim, heads, block_size) != (128, 32, 8):
        raise ValueError("candidate port requires 32 heads, dimension128 and candidate block8")
    if values.dtype != torch.float8_e4m3fn or cache.dtype != torch.uint8:
        raise ValueError("candidate port requires FP8 Q and byte-backed K pages")
    if cache.ndim != 4 or cache.shape[2:] != (1, 132) or cache.shape[1] != 64:
        raise ValueError("candidate port requires SM12x segregated FP8 page64")
    if lengths.numel() != rows or candidates.shape[0] < rows:
        raise ValueError("candidate decode row mapping changed")
    if table.shape[0] != b:
        raise ValueError("native MQA block table must have one row per Q batch entry")
    if candidates.dtype != torch.int32 or table.dtype != torch.int32:
        raise ValueError("candidate/table indices must be int32")
    from dgx_campaign_runtime.candidate_kernels import pack_candidates, scatter_scores
    positions, packed_cache, packed_table, counts = pack_candidates(
        cache, table, request_indices, lengths.reshape(-1), candidates[:rows],
        next_n=next_n, width=max_model_len)
    packed_q = values.reshape(rows, 1, heads, dim)
    schedule = schedule_builder(counts[:, None], packed_cache.shape[1], _sm_count(values.device))
    logits = scorer((packed_q, None), packed_cache, weights[:rows], counts[:, None],
                    packed_table, schedule, max_model_len=positions.shape[1],
                    clean_logits=False, indices=None)
    if selected == 1:
        return CandidateScores(scatter_scores(logits, positions, counts, max_model_len))
    return CandidateScores(logits, positions, counts[:, None])


def remap_indices(indices, positions, lengths):
    from dgx_campaign_runtime.candidate_kernels import remap_topk
    remap_topk(indices, positions, lengths)
