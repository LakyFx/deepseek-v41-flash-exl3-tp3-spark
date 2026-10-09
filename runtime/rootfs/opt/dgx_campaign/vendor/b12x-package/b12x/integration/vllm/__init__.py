"""vLLM quantization plugin for b12x MX-FP6 (W6A6/W6A8).

Import :func:`register_b12x_fp6` from :mod:`b12x.integration.vllm.plugin`
and wire it through vLLM's ``general_plugins`` entry point (see README).
"""

from .fp6_serving import (
    QUANT_ALGO,
    QUANT_METHOD,
    B12XFP6LinearMethod,
    B12XFP6MoEMethod,
    is_b12x_fp6_checkpoint,
    is_b12x_fp6_enabled,
    load_b12x_fp6_linear_methods,
    load_b12x_fp6_moe_methods,
    should_use_b12x_fp6,
)

__all__ = [
    "QUANT_ALGO",
    "QUANT_METHOD",
    "B12XFP6LinearMethod",
    "B12XFP6MoEMethod",
    "is_b12x_fp6_checkpoint",
    "is_b12x_fp6_enabled",
    "load_b12x_fp6_linear_methods",
    "load_b12x_fp6_moe_methods",
    "should_use_b12x_fp6",
]
