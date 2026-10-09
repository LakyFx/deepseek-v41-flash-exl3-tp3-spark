"""Sparse-MLA attention backend backed by this plugin's SM120 decode kernel.

vLLM's attention backends live in an enum, and an out-of-tree package cannot
add a member to it. It can, however, bind the reserved ``CUSTOM`` slot, so the
backend is selected with ``--attention-backend CUSTOM``.

Only the decode (MQA) path is ours. Prefill, the top-k mask machinery and the
metadata builder are vLLM's ``SparseMLACommonImpl`` and
``FlashInferMLASparseMetadataBuilder``, unchanged.

Two differences from ``FLASHINFER_MLA_SPARSE_SM120``, which is the backend this
replaces on SM120:

* The cache is bf16 or e4m3 fp8 in its natural ``(slot, head_size)`` layout,
  not the packed 656-byte ``fp8_ds_mla`` record, so there are no per-block
  scales to unpack on the read path.
* ``head_size`` is whatever the model actually has. GLM-5.3-Flash has
  ``qk_rope_head_dim = 0`` and therefore a 512-wide latent; the kernels this
  replaces require 576 and are fed 64 zero dims per key to get there.
"""

from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionLayer,
    AttentionType,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    FlashInferMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

# Shapes the kernel is compiled for: (head_size, kv_lora_rank).
SUPPORTED_SHAPES = ((576, 512), (512, 512))


class Exl3MLASparseBackend(AttentionBackend):
    """Sparse MLA on SM120 with a bf16 cache."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto", "bfloat16", "fp8", "fp8_e4m3",
    ]

    @staticmethod
    def get_name() -> str:
        return "EXL3_MLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type[FlashInferMLASparseMetadataBuilder]:
        return FlashInferMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl]:
        return Exl3MLASparseImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [hs for hs, _ in SUPPORTED_SHAPES]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # The top-k list is converted to absolute slot indices before it reaches
        # the kernel, which reads rows, so the kernel itself does not care. But
        # advertising MultipleOf(16) lets vLLM grow the page to 1152 tokens to
        # match the indexer's page size, and the DSA indexer's own paged-MQA
        # kernel then asserts (`block_kv == 64` on sm120 without an fp4 index
        # cache). Offer the same fixed sizes the backend this replaces does.
        return [64, 256]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 12

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: "CacheDType | None",
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if not use_sparse:
            return "EXL3_MLA_SPARSE is a sparse-MLA backend"
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config()
        if cfg.model_config is None:
            return None
        hf = cfg.model_config.hf_text_config
        if getattr(hf, "index_topk", None) is None:
            return "EXL3_MLA_SPARSE requires a model with index_topk"
        rank = getattr(hf, "kv_lora_rank", None)
        if rank is not None and (head_size, rank) not in SUPPORTED_SHAPES:
            return (
                "EXL3_MLA_SPARSE supports (head_size, kv_lora_rank) in "
                f"{SUPPORTED_SHAPES}; got ({head_size}, {rank})"
            )
        return None

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)


class Exl3MLASparseImpl(SparseMLACommonImpl[FlashInferMLASparseMetadata]):
    """Decode through ``cuda_exl3_C.mla_decode``; prefill through the base."""

    # The kernel merges its own splits and does not hand back an LSE, so it
    # cannot participate in decode-context parallelism.
    can_return_lse_for_decode: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "EXL3_MLA_SPARSE does not support alibi_slopes, sliding_window "
                "or logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "EXL3_MLA_SPARSE only supports decoder self-attention"
            )
        if kv_cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
            raise NotImplementedError(
                "EXL3_MLA_SPARSE reads the cache directly and needs it in "
                f"bfloat16 or e4m3 fp8; got kv_cache_dtype={kv_cache_dtype!r}"
            )
        self._fp8_cache = kv_cache_dtype in ("fp8", "fp8_e4m3")
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
            **mla_args,
        )
        if (head_size, self.kv_lora_rank) not in SUPPORTED_SHAPES:
            raise NotImplementedError(
                f"EXL3_MLA_SPARSE supports (head_size, kv_lora_rank) in "
                f"{SUPPORTED_SHAPES}; got ({head_size}, {self.kv_lora_rank})"
            )
        assert self.topk_indices_buffer is not None
        # The kernel takes a query per token, not a fused fp8 one.
        self.supports_quant_query_input = False
        self._no_seqlens = torch.empty(0, dtype=torch.int32)

    def _kv_scale(self, layer: AttentionLayer) -> float:
        """The dequant scale for an fp8 cache, 1.0 for bf16.

        It is never applied per element: both matmuls are linear in it, so the
        kernel folds it into the softmax scale and into the merged output.
        """
        if not self._fp8_cache:
            return 1.0
        s = getattr(layer, "_k_scale_float", None)
        if s is None:
            k = getattr(layer, "_k_scale", None)
            s = float(k.item()) if k is not None else 1.0
        return float(s)

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        # Per-request token ids -> absolute cache slots. Invalid entries stay
        # negative and the kernel skips them.
        topk_slots = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_actual_toks],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
        )

        rows = kv_c_and_k_pe_cache.view(-1, kv_c_and_k_pe_cache.shape[-1])
        if self._no_seqlens.device != q.device:
            self._no_seqlens = self._no_seqlens.to(q.device)

        out = torch.ops.cuda_exl3_C.mla_decode(
            q.contiguous(),
            rows,
            topk_slots.to(torch.int32).contiguous(),
            self._no_seqlens,
            float(self.scale),
            self.kv_lora_rank,
            0,  # split_chunk: autotuned
            0,  # block shape: autotuned
            self._kv_scale(layer),
        )
        return out, None
