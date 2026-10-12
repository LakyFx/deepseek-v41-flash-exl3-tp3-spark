"""Selective TP3 one-shot transport; never falls back after a backend failure.

Mounted as dgx_campaign_runtime.transport in a future immutable candidate.
The pinned sparknet package owns RDMA, dispatch votes and stream ordering.
This module deliberately does not enable transport merely on import.
"""
from __future__ import annotations

import os


def selected() -> bool:
    value = os.environ.get("DGX_CAMPAIGN_R", "0")
    if value not in ("0", "1"):
        raise ValueError("DGX_CAMPAIGN_R must be 0 or 1")
    return value == "1"


def construct(communicator, adapter_class=None):
    if not selected() or "tp" not in communicator.unique_name:
        return None
    if communicator.world_size != 3:
        raise ValueError("this transport port is qualified only for TP3")
    if communicator.device_group is None:
        raise ValueError("TP3 CUDA process group missing")
    bulk = communicator.pynccl_comm
    if bulk is None or bulk.disabled:
        raise RuntimeError("one-shot policy requires enabled bulk PyNccl")
    # We do not change the baseline NCCL library/profile. Disallow parallel
    # symmetric policies whose registration or dispatch can differ by rank.
    parallel=(("use_torch_symm_mem","symm_mem_comm"),
              ("use_flashinfer_allreduce","fi_ar_comm"),
              ("use_flashinfer_pcie_ipc_allreduce","fi_pcie_ipc_ar_comm"),
              ("use_aiter_allreduce","aiter_ar_comm"))
    # vLLM flags express a requested policy. On SM121/TP3 SymmMem and
    # FlashInfer reject capability/world-size during construction and remain
    # disabled, even though their requested flags are still true.
    if any(getattr(communicator,flag,False) and (backend:=getattr(communicator,name,None)) is not None
           and not getattr(backend,'disabled',False) for flag,name in parallel):
        raise ValueError("R requires one explicit transport policy")
    if adapter_class is None:
        from sparknet.integration.vllm import SparknetOneShotAllReduce
        adapter_class = SparknetOneShotAllReduce
    adapter = adapter_class(group=communicator.cpu_group,
                            device_group=communicator.device_group,
                            device=communicator.device,
                            global_ranks=communicator.ranks,
                            nccl_available=True)
    if adapter.disabled:
        raise RuntimeError("requested TP3 transport was not constructed")
    return adapter


def all_reduce(communicator, tensor):
    adapter = communicator.dgx_campaign_r
    if adapter.should_custom_ar(tensor):
        result = adapter.custom_all_reduce(tensor)
    else:
        # Eligibility rejection happens before launch and is shared by ranks.
        # An exception or None from an attempted collective is always fatal.
        result = communicator.pynccl_comm.all_reduce(tensor)
    if result is None:
        raise RuntimeError("selected collective returned no output")
    return result


def output_handle():
    if not selected():
        return None
    from vllm.distributed import get_tp_group
    communicator = get_tp_group().device_communicator
    handle = getattr(communicator, "dgx_campaign_r", None)
    if handle is None or handle.disabled:
        raise RuntimeError("R output cannot use a missing or disabled transport")
    return handle


def check_after_copy(handle) -> None:
    # Called AFTER AsyncOutput's existing D2H event, before any tokens escape.
    # No per-collective CUDA synchronization is introduced.
    if handle is not None:
        handle.check_health()


def finish_warmup(worker) -> None:
    if not selected():
        return
    if not worker.use_v2_model_runner:
        raise ValueError("R output health integration currently requires V2 runner")
    handle = output_handle()
    handle.check_health()
    from sparknet.oneshot import freeze_kernel_resolution
    freeze_kernel_resolution("DGX campaign warmup finished")
