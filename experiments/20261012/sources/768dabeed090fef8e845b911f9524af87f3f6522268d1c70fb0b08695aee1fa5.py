"""N5 reuse immutable compact-page block tables, preserve scoring/top-k.

This is an exact scheduling optimization around the existing SM12x scorer,
not an unqualified upgrade to DeepGEMM's incompatible SM100 interface.
"""
from functools import cache


@cache
def _table_owner(device):
    import torch
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('N5 immutable page table must be initialized before capture')
    # One permanent 48KiB owner per device. An LRU of shape-specific owners
    # could evict memory still referenced by an already captured CUDA graph.
    return torch.arange(48 * 256, dtype=torch.int32, device=device)


def page_table(device, rows, pages):
    if not 1 <= rows <= 48:
        raise ValueError('N5 decode rows outside locked capacity')
    if not 1 <= pages <= 256:
        raise ValueError('N5 locked 2048*8 candidate capacity exceeded')
    return _table_owner(device)[:rows * pages].view(rows, pages)

