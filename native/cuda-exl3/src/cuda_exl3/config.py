"""vLLM quantization config for the EXL3 (ExLlamaV3) format."""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from cuda_exl3 import env as _env

logger = init_logger(__name__)

# Codebook ids, matching exllamav3's `cb` template parameter.
CB_3INST = 0
CB_MCG = 1
CB_MUL1 = 2

# CUDA_EXL3_DEBUG_NAMES=1 logs every module's resolution. At warning level on
# purpose: it is opt-in, and logger.info does not reach vLLM's default log
# level, so at info it printed nothing and looked broken (reported in #5). The
# running tallies are the point -- a mapping that silently leaves half the
# attention stack in bf16 shows up as a bf16 count that keeps climbing.
_DEBUG = bool(_env.getenv("CUDA_EXL3_DEBUG_NAMES"))
_resolved: list[str] = []
_unresolved: list[str] = []

# Wrapper segments that different stacks nest in different orders.
_WRAPPER_SEGMENTS = frozenset({"model", "language_model"})


def _normalize(name: str) -> str:
    return ".".join(p for p in name.split(".") if p not in _WRAPPER_SEGMENTS)


class Exl3ModuleInfo:
    """Static description of one EXL3-quantized tensor from the checkpoint."""

    __slots__ = ("name", "in_features", "out_features", "bits", "cb", "cb_mult", "has_bias")

    def __init__(self, name, in_features, out_features, bits, cb, cb_mult, has_bias):
        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.cb = cb
        self.cb_mult = cb_mult
        self.has_bias = has_bias

    def __repr__(self):
        return (
            f"Exl3ModuleInfo({self.name}, {self.in_features}x{self.out_features}, "
            f"bits={self.bits}, cb={self.cb})"
        )


# Layers left alone by online quantization. The router decides which experts
# run at all, so its error is not averaged away like a projection's, and the
# head is the last thing before the logits -- EXL3 checkpoints conventionally
# keep both in high precision (this one records head_bits: 16).
_ONLINE_EXCLUDE = ("lm_head", "gate", "router", "e_score_correction", "shared_head")


def _is_online_excluded(prefix: str) -> bool:
    tail = prefix.rsplit(".", 1)[-1]
    if any(tail == e or tail.startswith(e) for e in _ONLINE_EXCLUDE):
        return True
    return "lm_head" in prefix


@register_quantization_config("exl3")
class Exl3Config(QuantizationConfig):
    """EXL3 trellis quantization.

    The checkpoint stores, per quantized tensor:
      ``trellis``  int16 (in/16, out/16, 16*bits)  -- packed trellis codes
      ``suh``      fp16  (in,)                     -- input scales x sign flips
      ``svh``      fp16  (out,)                    -- output scales x sign flips
      ``mcg``/``mul1`` int32 scalar                -- procedural codebook multiplier

    ``bits`` varies per tensor (5/6/7 in practice), so the per-tensor table from
    ``quantization_config.json`` is required to size parameters at build time --
    the summary block inlined into ``config.json`` does not carry it.
    """

    def __init__(self, full_config: dict[str, Any]):
        super().__init__()
        self.full_config = full_config
        self.version = full_config.get("version", "unknown")
        self.default_bits = full_config.get("bits")
        self.codebook_name = full_config.get("codebook", "mul1")

        # Fusions the model class does not declare, supplied by the checkpoint.
        # See _candidate_names. Read defensively: a malformed entry should not
        # stop a checkpoint loading, it should just not resolve.
        self.extra_packed_mapping: dict[str, list[str]] = {}
        raw = full_config.get("packed_modules_mapping")
        # A published checkpoint cannot be edited, and whether the standalone
        # quantization_config.json is even read depends on whether the inlined
        # copy carried tensor_storage -- so the same mapping can also be given
        # as JSON in CUDA_EXL3_PACKED_MAPPING, which always applies and needs no
        # write access to the weights.
        env_raw = _env.getenv("CUDA_EXL3_PACKED_MAPPING")
        if env_raw:
            try:
                parsed = json.loads(env_raw)
                if isinstance(parsed, dict):
                    raw = {**(raw if isinstance(raw, dict) else {}), **parsed}
            except ValueError as e:
                logger.warning("EXL3: CUDA_EXL3_PACKED_MAPPING is not valid JSON "
                               "(%s); ignoring it", e)
        if isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(k, str) and isinstance(v, (list, tuple)) \
                        and all(isinstance(x, str) for x in v):
                    self.extra_packed_mapping[k] = list(v)

        storage = full_config.get("tensor_storage")
        if not storage:
            # vLLM hands us the summary block inlined into config.json, which
            # omits the per-tensor table (bitrates differ per layer, so we need
            # it to size parameters). Pull in the standalone file written next
            # to the weights.
            full_config = self._merge_tensor_storage(full_config)
            self.full_config = full_config
            storage = full_config.get("tensor_storage")
        if not storage:
            raise ValueError(
                "EXL3: could not find `tensor_storage`. cuda-exl3 needs the full "
                "`quantization_config.json` written by exllamav3 (the copy inlined "
                "into config.json is only a summary). Pass it explicitly with "
                "--hf-overrides '{\"quantization_config_file\": \"/path/to/"
                "quantization_config.json\"}' if it is not next to the weights."
            )
        self.modules: dict[str, Exl3ModuleInfo] = {}
        for name, entry in storage.items():
            info = self._parse_module(name, entry)
            if info is not None:
                self.modules[name] = info

        # vLLM's module paths do not always match the checkpoint's. Qwen3.5, for
        # instance, is `model.language_model.layers.N...` on disk but
        # `language_model.model.layers.N...` in vLLM. Index on a form with the
        # `model`/`language_model` wrapper segments dropped so either spelling
        # resolves.
        self.modules_norm: dict[str, Exl3ModuleInfo] = {}
        for name, info in self.modules.items():
            key = _normalize(name)
            if key in self.modules_norm:
                logger.warning(
                    "EXL3: normalized name collision on %r (%s vs %s); "
                    "falling back to exact matching for these.",
                    key, self.modules_norm[key].name, name,
                )
                self.modules_norm[key] = None  # type: ignore[assignment]
            else:
                self.modules_norm[key] = info
        # Some checkpoints omit modules from quantization_config.json even
        # though the tensors are present -- Qwen3.5 ships an EXL3-quantized MTP
        # head that the config never mentions. Recover those from the
        # safetensors headers so speculative decoding can use them.
        recovered = self._augment_from_checkpoint()

        logger.info(
            "EXL3: %d quantized tensors (format v%s, codebook=%s)%s",
            len(self.modules),
            self.version,
            self.codebook_name,
            f", {recovered} recovered from checkpoint headers" if recovered else "",
        )

    @staticmethod
    def _parse_module(name: str, entry: dict) -> Exl3ModuleInfo | None:
        if entry.get("quant_format") != "exl3":
            return None
        tensors = entry.get("stored_tensors", {})
        trellis = tensors.get(f"{name}.trellis")
        if trellis is None:
            return None
        k_tiles, n_tiles, packed = trellis["shape"]
        bits = packed // 16

        if f"{name}.mul1" in tensors:
            cb, cb_mult = CB_MUL1, entry.get("mul1_multiplier", 0)
        elif f"{name}.mcg" in tensors:
            cb, cb_mult = CB_MCG, entry.get("mcg_multiplier", 0)
        else:
            cb, cb_mult = CB_3INST, 0

        return Exl3ModuleInfo(
            name=name,
            in_features=k_tiles * 16,
            out_features=n_tiles * 16,
            bits=bits,
            cb=cb,
            cb_mult=cb_mult,
            has_bias=f"{name}.bias" in tensors,
        )

    # -- locating the full per-tensor config -----------------------------

    # Stashed by override_quantization_method(), which vLLM calls with the HF
    # config before it builds the quantization config.
    _model_path_hint: str | None = None

    @classmethod
    def _merge_tensor_storage(cls, config: dict[str, Any]) -> dict[str, Any]:
        raw = cls._load_config_file(cls._model_path_hint)
        if raw is None:
            return config
        merged = dict(raw)
        merged.update({k: v for k, v in config.items() if k != "tensor_storage"})
        return merged

    @staticmethod
    def _load_config_file(model_path: str | None) -> dict[str, Any] | None:
        if not model_path:
            return None
        fname = "quantization_config.json"
        try:
            if os.path.isdir(model_path):
                path = os.path.join(model_path, fname)
                if not os.path.exists(path):
                    return None
                with open(path) as f:
                    return json.load(f)
            # Remote repo id: the file sits next to the weights on the Hub.
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id=model_path, filename=fname)
            with open(path) as f:
                return json.load(f)
        except Exception as e:  # pragma: no cover - diagnostic path
            logger.warning("EXL3: could not load %s from %s: %s", fname, model_path, e)
            return None

    @classmethod
    def from_config_file(cls, path: str) -> "Exl3Config":
        with open(path) as f:
            return cls(json.load(f))

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        # We never override another method; this hook just gives us the one
        # chance to see where the model was loaded from.
        if hf_config is not None:
            path = getattr(hf_config, "_name_or_path", None)
            if path:
                cls._model_path_hint = path
        return None


    def _augment_from_checkpoint(self) -> int:
        """Add EXL3 modules present in the weights but missing from the config.

        Reads only safetensors headers (shape/dtype), never tensor data.
        """
        path = self._model_path_hint
        if not path or not os.path.isdir(path):
            return 0
        try:
            index_path = os.path.join(path, "model.safetensors.index.json")
            if os.path.exists(index_path):
                with open(index_path) as f:
                    weight_map = json.load(f)["weight_map"]
            elif os.path.exists(os.path.join(path, "model.safetensors")):
                weight_map = None
            else:
                return 0

            from safetensors import safe_open

            if weight_map is None:
                shards = {"model.safetensors": None}
                with safe_open(os.path.join(path, "model.safetensors"),
                               framework="pt") as f:
                    weight_map = {k: "model.safetensors" for k in f.keys()}

            # Group by the module each tensor belongs to.
            by_module: dict[str, set[str]] = {}
            for key in weight_map:
                mod, _, leaf = key.rpartition(".")
                if leaf in ("trellis", "suh", "svh", "mcg", "mul1", "bias"):
                    by_module.setdefault(mod, set()).add(leaf)

            missing = [
                m for m, leaves in by_module.items()
                if "trellis" in leaves and m not in self.modules
            ]
            if not missing:
                return 0

            handles: dict[str, Any] = {}
            added = 0
            for mod in missing:
                key = f"{mod}.trellis"
                fname = weight_map[key]
                if fname not in handles:
                    handles[fname] = safe_open(os.path.join(path, fname), framework="pt")
                shape = handles[fname].get_slice(key).get_shape()
                if len(shape) != 3:
                    continue
                k_tiles, n_tiles, packed = shape
                leaves = by_module[mod]
                if "mul1" in leaves:
                    cb = CB_MUL1
                elif "mcg" in leaves:
                    cb = CB_MCG
                else:
                    cb = CB_3INST
                info = Exl3ModuleInfo(
                    name=mod,
                    in_features=k_tiles * 16,
                    out_features=n_tiles * 16,
                    bits=packed // 16,
                    cb=cb,
                    cb_mult=0,
                    has_bias="bias" in leaves,
                )
                self.modules[mod] = info
                self.modules_norm.setdefault(_normalize(mod), info)
                added += 1
            return added
        except Exception as e:  # pragma: no cover - diagnostic path
            logger.warning("EXL3: could not scan checkpoint headers: %s", e)
            return 0

    # -- QuantizationConfig interface ------------------------------------

    def get_name(self):
        return "exl3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # The trellis codebook decodes natively to fp16 and every EXL3 kernel is
        # fp16; bf16 activations are converted at the layer boundary.
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # mma.m16n8k16 + cp.async
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        # NOT config.json: only the standalone file carries `tensor_storage`.
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        return cls(config)

    # -- module lookup ---------------------------------------------------

    def _candidate_names(self, prefix: str) -> list[list[str]]:
        """Checkpoint module names that could back a vLLM module `prefix`.

        Returns a list of candidate *groups*; a group has more than one entry
        when vLLM fuses several checkpoint tensors into one layer (qkv_proj,
        gate_up_proj). Ordered most-likely first.
        """
        out: list[list[str]] = [[prefix]]

        # vLLM fills packed_modules_mapping from the *model class*, so a fusion
        # the model does not declare -- or declares under a different name than
        # the checkpoint used -- cannot be resolved. A checkpoint may therefore
        # carry supplementary entries of its own in its quantization config:
        #
        #   "packed_modules_mapping": {"in_proj_qkvbfg_a": ["qkv_proj", ...]}
        #
        # so a fusion peculiar to one model travels with the weights instead of
        # requiring a patch to the model file. Checkpoint entries are merged
        # under, never over, whatever the model declared.
        base, _, last = prefix.rpartition(".")
        merged = dict(self.extra_packed_mapping)
        merged.update(self.packed_modules_mapping)
        for packed_name, sub_names in merged.items():
            if last == packed_name:
                out.append([f"{base}.{s}" if base else s for s in sub_names])

        # vLLM sometimes drops or adds a leading "model." relative to the
        # checkpoint (e.g. lm_head, or multimodal wrappers).
        extra = []
        for group in out:
            if all(g.startswith("model.") for g in group):
                extra.append([g[len("model.") :] for g in group])
            else:
                extra.append([f"model.{g}" for g in group])
        out.extend(extra)
        return out

    def resolve(self, prefix: str) -> list[Exl3ModuleInfo] | None:
        """Resolve a vLLM module prefix to its EXL3 shards, or None."""
        groups = self._candidate_names(prefix)
        for lookup in (self.modules, self.modules_norm):
            key = (lambda n: n) if lookup is self.modules else _normalize
            for group in groups:
                infos = [lookup.get(key(n)) for n in group]
                if all(i is not None for i in infos):
                    return infos  # type: ignore[return-value]
        return None

    def resolve_moe(self, prefix: str) -> tuple | None:
        """Resolve a routed-experts module to expert 0's gate/up/down tensors.

        All experts of a layer share a bitrate and codebook (checked at load), so
        one expert is enough to size the fused weights.
        """
        out = []
        for proj in ("gate_proj", "up_proj", "down_proj"):
            info = self.resolve(f"{prefix}.0.{proj}")
            if info is None or len(info) != 1:
                return None
            out.append(info[0])
        return tuple(out)

    def _has_exl3_under(self, prefix: str) -> bool:
        """Any EXL3 tensor nested under this module prefix (e.g. MoE experts)."""
        key = _normalize(prefix) + "."
        return any(n.startswith(key) for n in self.modules_norm)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        from cuda_exl3.linear import Exl3LinearMethod

        infos = self.resolve(prefix)

        # Mixture-of-experts layers are not implemented. vLLM silently falls back
        # to UnquantizedFusedMoEMethod when get_quant_method returns None, which
        # then fails deep inside weight loading with an unrelated-looking error,
        # so fail here with the actual reason. Fused experts need a grouped GEMM
        # (one trellis per expert, selected per token), which is a separate
        # kernel from the dense path.
        cls_name = type(layer).__name__
        if cls_name in ("RoutedExperts", "FusedMoE", "SharedFusedMoE") or (
            "MoE" in cls_name and "Method" not in cls_name
        ):
            moe_infos = self.resolve_moe(prefix)
            if moe_infos is not None:
                from cuda_exl3.moe import Exl3MoEMethod

                moe_cfg = getattr(layer, "moe_config", None)
                return Exl3MoEMethod(moe_cfg, self, *moe_infos, prefix=prefix)
            if infos is not None or self._has_exl3_under(prefix):
                raise NotImplementedError(
                    f"EXL3: {prefix} has quantized experts that could not be "
                    "resolved to per-expert gate/up/down tensors."
                )
            return None

        if isinstance(layer, (LinearBase, VocabParallelEmbedding)):
            if infos is not None:
                if _DEBUG:
                    _resolved.append(prefix)
                    logger.warning("EXL3: %s -> EXL3 (%d shard%s)  [%d exl3 / %d bf16]",
                                   prefix, len(infos), "" if len(infos) == 1 else "s",
                                   len(_resolved), len(_unresolved))
                return Exl3LinearMethod(self, infos, prefix)
            # A linear layer with no EXL3 tensors is genuinely unquantized in
            # this checkpoint (e.g. the bf16 vision tower, or a checkpoint that
            # only quantized its routed experts). Optionally encode it here
            # instead of leaving it in bf16 -- see cuda_exl3.online.
            if isinstance(layer, LinearBase):
                from cuda_exl3.online import (
                    Exl3OnlineLinearMethod,
                    online_bits,
                    shape_supported,
                )

                bits = online_bits()
                if bits is not None and not _is_online_excluded(prefix):
                    k = getattr(layer, "input_size_per_partition", None)
                    n = sum(getattr(layer, "output_partition_sizes", []) or [0])
                    if k and n and shape_supported(k, n):
                        logger.info_once(
                            "EXL3: quantizing unquantized linears online at "
                            "%d bits (CUDA_EXL3_ONLINE_BITS)", bits
                        )
                        return Exl3OnlineLinearMethod(self, prefix, bits)
                    if _DEBUG:
                        logger.warning("EXL3 online: %s shape %sx%s unsupported",
                                    prefix, k, n)
                if _DEBUG:
                    _unresolved.append(prefix)
                    logger.warning("EXL3: %s -> unquantized  [%d exl3 / %d bf16]",
                                   prefix, len(_resolved), len(_unresolved))
                return UnquantizedLinearMethod()
            return None

        return None
