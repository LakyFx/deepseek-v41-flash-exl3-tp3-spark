"""Parameter classes for EXL3 weights.

Most EXL3 tensors map cleanly onto vLLM's stock parameter classes:

* ``trellis`` -> :class:`PackedvLLMParameter` with ``packed_dim=1,
  packed_factor=16``. Its shape is ``(in/16, out/16, 16*bits)``, so column-parallel
  shard offsets/sizes (given in output *features*) need dividing by 16, which is
  exactly what the packing adjustment does. Row-parallel sharding on dim 0 is
  computed from the parameter's own shape and needs no adjustment.
* ``svh`` -> :class:`ChannelQuantScaleParameter` (one value per output feature).
  For row-parallel layers the output dim is not split, and the inherited
  copy-whole ``load_row_parallel_weight`` is correct.

``suh`` is the exception. It is one value per *input* feature, but unlike a
normal per-input scale it differs between the shards of a fused layer: q/k/v (and
gate/up) are quantized as separate tensors with separately chosen input scales.
The magnitudes genuinely differ (only the sign pattern is shared), and ``suh``
sits *inside* the Hadamard transform, so the shards cannot be folded into one
vector. We therefore keep one row per shard.

That is structurally the same thing vLLM's :class:`PerTensorScaleParameter`
expresses -- "one entry per logical matrix in a fused layer" -- so we inherit it
to reuse vLLM's shard-id plumbing, including the case where one checkpoint tensor
covers several of the layer's shards (Qwen3.5 stores a pre-fused ``in_proj_qkv``
that lands on shards 0, 1 and 2 of vLLM's ``in_proj_qkvz``; vLLM replays the load
once per shard id, giving each of those shards its own copy of the same suh).
"""

from __future__ import annotations

import torch

from vllm.model_executor.parameter import PerTensorScaleParameter

__all__ = ["Exl3SuhParameter"]


class Exl3SuhParameter(PerTensorScaleParameter):
    """Per-shard input scales, shape ``(num_shards, in_features_per_partition)``."""

    def _load_into_shard_id(self, loaded_weight: torch.Tensor, shard_id, **kwargs) -> None:
        # PerTensorScaleParameter squeezes a leading 1-dim because its payload is
        # a scalar scale. suh is a full per-input-channel vector, so copy as is.
        row = self.data[self._shard_id_as_int(shard_id)]
        assert row.shape == loaded_weight.shape, (
            f"suh shard {shard_id}: expected {tuple(row.shape)}, "
            f"got {tuple(loaded_weight.shape)}"
        )
        row.copy_(loaded_weight)

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor) -> None:
        # Unfused column-parallel layer: one shard, input dim not split.
        assert self.data.shape[0] == 1
        self.data[0].copy_(loaded_weight)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor) -> None:
        # Row-parallel layers split the input dim. Slicing suh at a tp boundary
        # stays exact because the EXL3 input transform is a *block-diagonal*
        # Hadamard over 128-element blocks; the linear method asserts that each
        # rank's input size is a multiple of 128.
        assert self.data.shape[0] == 1, "row-parallel EXL3 layers are never fused"
        shard_size = self.data.shape[1]
        start = self.tp_rank * shard_size
        # The layer's input dim may be padded (a padded head count upstream makes
        # o_proj wider than the checkpoint), in which case the last rank's slice
        # runs past the end of the checkpoint tensor. Copy what exists and leave
        # the rest at zero: those input positions carry exact zeros, because the
        # producing layer's pad columns were zeroed through its own svh, so what
        # suh holds there cannot reach the output. Zero rather than garbage so a
        # NaN cannot be introduced by a value nothing should read.
        avail = max(0, min(shard_size, loaded_weight.shape[0] - start))
        if avail < shard_size:
            self.data[0].zero_()
        if avail > 0:
            self.data[0, :avail].copy_(loaded_weight.narrow(0, start, avail))
