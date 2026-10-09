"""H: target-only native MXFP8 dense selection, without weight conversion.

Uses the installed vLLM ModelOpt lifecycle and B12X packed-weight backend.
Routed EXL3, BMM, draft, BF16 router, and mHC are outside this selection.
No CUDA or vLLM import occurs while this helper is disabled.
"""
from __future__ import annotations

import os
import re

_TARGET = re.compile(r"^(?:language_model\.)?model\.layers\.(?:[0-9]|[12][0-9]|3[0-9])\.(?:"
                     r"attn\..+|ffn\.shared_experts\.(?:gate_up_proj|down_proj))$")


def selected(prefix: str, bmm_batch_size=None) -> bool:
    value = os.environ.get("DGX_CAMPAIGN_H", "0")
    if value not in ("0", "1"):
        raise ValueError("DGX_CAMPAIGN_H must be 0 or 1")
    return value == "1" and bmm_batch_size is None and bool(_TARGET.fullmatch(prefix))


def choose(layer, fallback):
    batch = getattr(layer, "bmm_batch_size", None)
    if not selected(getattr(layer, "prefix", ""), batch):
        return fallback(bmm_batch_size=batch)
    from vllm.model_executor.kernels.linear.mxfp8.b12x import B12xMxfp8LinearKernel
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import Mxfp8LinearLayerConfig
    from vllm.logger import init_logger
    supported, reason = B12xMxfp8LinearKernel.is_supported()
    if not supported:
        # A missing dependency must not silently turn this arm into baseline.
        raise RuntimeError("H native MXFP8 unavailable: " + str(reason))
    config = Mxfp8LinearLayerConfig(bmm_batch_size=None)
    valid, reason = B12xMxfp8LinearKernel.can_implement(config)
    if not valid:
        raise RuntimeError("H target shape unsupported: " + str(reason))
    init_logger(__name__).info_once("Campaign H native MXFP8: %s", layer.prefix, scope="global")
    return B12xMxfp8LinearKernel(config)
