"""EXL3 linear (and lm_head) method for vLLM."""

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    PackedvLLMParameter,
)

from cuda_exl3 import ops
from cuda_exl3.config import CB_3INST, CB_MCG, CB_MUL1
from cuda_exl3.parameter import Exl3SuhParameter

logger = init_logger(__name__)

# EXL3 packs weights in 16x16 tiles; every dim is a multiple of 16, and the
# input/output Hadamards act on 128-element blocks.
TILE = 16
HAD_BLOCK = 128


@register_weight_loader_v2_supported_method
class Exl3LinearMethod(LinearMethodBase):
    """Applies ``y = had(x * suh) @ dequant(trellis) * svh`` per shard.

    Fused layers (qkv_proj, gate_up_proj) keep one concatenated ``trellis`` and
    ``svh`` -- both concatenate cleanly along the output dim, and every shard of
    a fused group is quantized at the same bitrate -- but a *separate* ``suh``
    row per shard, because the input scales differ between shards and cannot be
    folded together through the Hadamard.
    """

    def __init__(self, quant_config, infos, prefix: str):
        self.quant_config = quant_config
        self.infos = infos
        self.prefix = prefix

        bits = {i.bits for i in infos}
        if len(bits) != 1:
            raise ValueError(
                f"EXL3 {prefix}: fused shards have mixed bitrates {sorted(bits)}; "
                "they cannot share one trellis tensor."
            )
        self.bits = bits.pop()

        cbs = {i.cb for i in infos}
        if len(cbs) != 1:
            raise ValueError(f"EXL3 {prefix}: fused shards use different codebooks")
        self.cb = cbs.pop()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight_loader = extra_weight_attrs.get("weight_loader")
        is_vocab = isinstance(layer, VocabParallelEmbedding)

        k = input_size_per_partition
        n_total = sum(output_partition_sizes)
        num_shards = len(output_partition_sizes)

        if k % TILE or n_total % TILE:
            raise ValueError(
                f"EXL3 {self.prefix}: dims must be multiples of {TILE}, got {k}x{n_total}"
            )
        # Row-parallel layers slice the input dim. The EXL3 input transform is a
        # block-diagonal Hadamard over 128-element blocks, so a per-rank slice is
        # only self-contained on a 128 boundary.
        if k != input_size and k % HAD_BLOCK:
            raise ValueError(
                f"EXL3 {self.prefix}: input_size_per_partition={k} is not a multiple "
                f"of {HAD_BLOCK}; this tensor-parallel split would break the Hadamard."
            )
        # A padded output dim (vLLM pads the vocab so it shards evenly, and a
        # padded head count does the same to q_b/o_proj). EXL3 weights cannot be
        # zero-extended -- a trellis is not a dense tensor -- but they do not
        # have to be: `svh` scales elementwise *after* the output Hadamard, so
        # zeroing it on the pad makes those columns exactly zero whatever the
        # trellis behind them holds. The decoder is total and bounded over all
        # 65,536 codes (checked exhaustively: no code decodes to inf or NaN), so
        # an uninitialised pad cannot produce a NaN through the zero.
        #
        # The condition is that the pad must not share a 128-column Hadamard
        # block with real output: the block mixes across its columns before svh
        # is applied, so a straddling pad would corrupt the real columns beside
        # it rather than merely being zero itself. Real output therefore has to
        # end on a block boundary.
        self.n_real = sum(i.out_features for i in self.infos)
        self.out_pad = 0
        if is_vocab and output_size != self.n_real:
            pad = output_size - self.n_real
            if pad < 0 or self.n_real % HAD_BLOCK or output_size % HAD_BLOCK:
                raise ValueError(
                    f"EXL3 {self.prefix}: vLLM padded the vocab to {output_size} but "
                    f"the checkpoint stores {self.n_real}, and the pad is not a whole "
                    f"number of {HAD_BLOCK}-column Hadamard blocks "
                    f"({self.n_real} % {HAD_BLOCK} = {self.n_real % HAD_BLOCK}, "
                    f"{output_size} % {HAD_BLOCK} = {output_size % HAD_BLOCK}). "
                    "A straddling pad corrupts the real columns sharing its block. "
                    "Pad the vocab to a multiple of the block size instead."
                )
            self.out_pad = pad
            logger.warning(
                "EXL3 %s: vocab padded %d -> %d; the %d pad columns are %d whole "
                "Hadamard blocks and are zeroed through svh.",
                self.prefix, self.n_real, output_size, pad, pad // HAD_BLOCK,
            )

        # The layer's shard count need not equal the checkpoint tensor count: a
        # checkpoint may store an already-fused tensor that covers several of
        # vLLM's shards (Qwen3.5's `in_proj_qkv` spans shards 0-2 of
        # `in_proj_qkvz`). vLLM replays such a load once per shard id, so we
        # keep one suh row per *layer* shard and let identical rows coalesce
        # again in process_weights_after_loading.
        expected = sum(i.out_features for i in self.infos)
        if not is_vocab and expected != output_size:
            logger.warning(
                "EXL3 %s: checkpoint tensors total %d output features but the "
                "layer declares %d; relying on per-tensor shape checks.",
                self.prefix, expected, output_size,
            )

        # suh is (num_shards, k). The v2 loader knows that; the v1 path does not,
        # and some vLLM versions have no weight_loader_v2 on ReplicatedLinear --
        # the indexer's wq_b is one -- so a bare (k,) arrives and the copy
        # asserts. Place it in row 0 when the shapes say that is what happened,
        # and delegate in every other case so nothing else changes.
        def _suh_loader(param, loaded_weight, *args, **kwargs):
            dst = param.data
            if (dst.dim() == 2 and dst.shape[0] == 1
                    and tuple(loaded_weight.shape) == tuple(dst.shape[1:])):
                dst[0].copy_(loaded_weight)
                return
            return weight_loader(param, loaded_weight, *args, **kwargs)

        loaders = (
            self._vocab_loaders(layer) if is_vocab else {"trellis": weight_loader,
                                                         "suh": _suh_loader,
                                                         "svh": weight_loader}
        )

        trellis = PackedvLLMParameter(
            data=torch.empty(
                k // TILE, n_total // TILE, TILE * self.bits,
                dtype=torch.int16, device=torch.cuda.current_device(),
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=TILE,
            weight_loader=loaders["trellis"],
        )
        suh = Exl3SuhParameter(
            data=torch.empty(num_shards, k, dtype=torch.half,
                             device=torch.cuda.current_device()),
            weight_loader=loaders["suh"],
        )
        svh = ChannelQuantScaleParameter(
            # Zeroed, not empty, when there is a pad: the loader never writes the
            # pad columns and svh = 0 is exactly what makes them vanish.
            data=(torch.zeros if self.out_pad else torch.empty)(
                n_total, dtype=torch.half, device=torch.cuda.current_device()),
            output_dim=0,
            weight_loader=loaders["svh"],
        )

        layer.register_parameter("trellis", trellis)
        layer.register_parameter("suh", suh)
        layer.register_parameter("svh", svh)

        # The procedural codebook multiplier is a per-tensor constant that the
        # kernels take as a compile-time codebook id, not as data. It is still a
        # tensor in the checkpoint, and vLLM's loader silently falls back to the
        # *layer* for any name it cannot resolve to a parameter, so it has to be
        # registered even though nothing reads it.
        cb_name = {CB_MUL1: "mul1", CB_MCG: "mcg"}.get(self.cb)
        if cb_name is not None:
            layer.register_parameter(
                cb_name,
                BasevLLMParameter(
                    data=torch.empty(1, dtype=torch.int32,
                                     device=torch.cuda.current_device()),
                    weight_loader=loaders.get(cb_name, weight_loader),
                ),
            )

        layer.exl3_bits = self.bits
        layer.exl3_cb = self.cb
        layer.exl3_k = k
        layer.exl3_n_total = n_total
        layer.exl3_shards = output_partition_sizes

    def _vocab_loaders(self, layer):
        """Loaders for ParallelLMHead.

        ``VocabParallelEmbedding.weight_loader`` assumes the vocab is dim 0 of
        the parameter, which holds for a dense ``(vocab, hidden)`` weight but not
        for an EXL3 trellis, where the output (vocab) dim is dim 1. Shard here
        instead, using the layer's own vocab shard indices.
        """

        def vocab_range():
            si = layer.shard_indices
            return si.org_vocab_start_index, si.org_vocab_end_index

        # This rank's slice of the *original* vocab. When vLLM pads the vocab so
        # it shards evenly, the parameter is the padded per-rank width while the
        # checkpoint only has the real rows, so both loaders fill a prefix and
        # leave the pad -- svh zeroed at allocation, which is what makes the pad
        # columns vanish, and the trellis behind them never read for a value.
        def load_trellis(param, loaded_weight):
            start, end = vocab_range()
            n = (end - start) // TILE
            lw = loaded_weight.narrow(1, start // TILE, n)
            param.data[:, :n].copy_(lw)

        def load_svh(param, loaded_weight):
            start, end = vocab_range()
            n = end - start
            param.data[:n].copy_(loaded_weight.narrow(0, start, n))

        def load_suh(param, loaded_weight):
            # input (hidden) dim is never vocab-sharded
            param.data[0].copy_(loaded_weight)

        def load_scalar(param, loaded_weight):
            param.data.copy_(loaded_weight.reshape(1))

        return {
            "trellis": load_trellis,
            "suh": load_suh,
            "svh": load_svh,
            "mul1": load_scalar,
            "mcg": load_scalar,
        }

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Merge neighbouring shards that ended up with the same suh. This undoes
        # the per-shard replay vLLM does for checkpoint tensors that were already
        # fused on disk, so those become one shard of the fused kernel again.
        bounds = []
        off = 0
        for n in layer.exl3_shards:
            bounds.append((off, off + n))
            off += n

        suh = layer.suh.data
        groups: list[list[int]] = []
        for i in range(len(bounds)):
            if groups and torch.equal(suh[groups[-1][0]], suh[i]):
                groups[-1].append(i)
            else:
                groups.append([i])

        # One suh row and one output width per surviving group. The kernel takes
        # the whole trellis and svh and addresses shards by offset, so nothing
        # else needs slicing (a slice along the output dim would be
        # non-contiguous and the kernel could not recover its row stride).
        layer.exl3_group_n = [bounds[g[-1]][1] - bounds[g[0]][0] for g in groups]
        compact = suh[[g[0] for g in groups]].contiguous()
        layer.suh = torch.nn.Parameter(compact, requires_grad=False)

        # Size the shared workspaces now, while nothing is captured yet.
        try:
            from vllm.config import get_current_vllm_config

            sched = get_current_vllm_config().scheduler_config
            max_tokens = max(sched.max_num_batched_tokens, sched.max_num_seqs)
            ops.reserve(layer.svh.data, max_tokens, layer.exl3_k,
                        layer.exl3_n_total, len(layer.exl3_group_n))
        except Exception as e:  # no vLLM context (unit tests) -> lazy sizing
            logger.debug("EXL3 %s: workspace pre-reserve skipped (%s)", self.prefix, e)

        if len(groups) != len(bounds):
            logger.debug(
                "EXL3 %s: coalesced %d shards into %d",
                self.prefix, len(bounds), len(groups),
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = ops.exl3_linear(
            x,
            layer.trellis.data,
            layer.suh.data,
            layer.svh.data,
            layer.exl3_group_n,
            layer.exl3_cb,
        )
        if bias is not None:
            out = out + bias
        return out
