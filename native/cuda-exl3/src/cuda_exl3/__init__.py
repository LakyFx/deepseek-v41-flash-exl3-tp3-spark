"""EXL3 (ExLlamaV3 trellis quantization) support for vLLM.

vLLM discovers this package through the ``vllm.general_plugins`` entry point and
calls :func:`register`, which makes ``quant_method: "exl3"`` checkpoints loadable.
"""

__version__ = "1.0.0"

_REGISTERED = False


def register() -> None:
    """Register the EXL3 quantization method with vLLM.

    Called by vLLM's plugin loader, potentially once per process (engine core,
    each worker), so it must be idempotent.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    # Importing the module runs the @register_quantization_config decorator.
    from cuda_exl3 import config as _config  # noqa: F401

    _register_attention_backend()
    _REGISTERED = True


def _register_attention_backend() -> None:
    """Bind the sparse-MLA decode kernel to ``--attention-backend CUSTOM``.

    vLLM's backends live in an enum and an out-of-tree package cannot add a
    member, but ``CUSTOM`` is reserved for exactly this. Binding it does not
    select it: the backend is only used when asked for by name. Missing on an
    older vLLM, and irrelevant on a machine without the compiled kernel, so
    both cases pass quietly.
    """
    try:
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
            register_backend,
        )
    except ImportError:
        return
    if not hasattr(AttentionBackendEnum, "CUSTOM"):
        return
    register_backend(
        AttentionBackendEnum.CUSTOM, "cuda_exl3.attention.Exl3MLASparseBackend"
    )


__all__ = ["register", "__version__"]
