"""Environment lookup with a deprecation path.

The knobs were named ``VLLM_EXL3_*`` when this was a vLLM-only plugin. They are
``CUDA_EXL3_*`` now; the old names still work so that deployment scripts written
against the old name keep behaving the same, rather than silently falling back
to a default.
"""
import os
import warnings

_LEGACY = "VLLM_EXL3_"
_PREFIX = "CUDA_EXL3_"


def getenv(name: str, default=None):
    """Read ``CUDA_EXL3_x``, falling back to ``VLLM_EXL3_x`` with a warning."""
    v = os.environ.get(name)
    if v is not None:
        return v
    if name.startswith(_PREFIX):
        legacy = _LEGACY + name[len(_PREFIX):]
        v = os.environ.get(legacy)
        if v is not None:
            warnings.warn(
                f"{legacy} is deprecated; use {name}. The old name still works.",
                DeprecationWarning, stacklevel=2,
            )
            return v
    return default
