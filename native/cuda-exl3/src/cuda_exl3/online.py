"""Online EXL3 quantization for layers the checkpoint left in bf16.

Some EXL3 checkpoints quantize only part of the model. GLM-5.3-Flash's is the
extreme case: `scope: glm53_routed_experts_only` leaves every attention
projection, the shared experts and the head in bf16. Those are only 11% of the
file, but the routed experts are read eight-of-288 at a time while the bf16
weights are read *in full, every token* -- about 16.8 GiB against 4.0 GiB, so
they dominate decode even though they look small on disk.

This encodes them to EXL3 at load time, uncalibrated: no Hessian, just the
incoherence transform and the trellis search that ExLlamaV3 already falls back
to when a layer has no captured activations. The result feeds exactly the same
kernel as a checkpoint-quantized layer.

Enable with CUDA_EXL3_ONLINE_BITS=<2..8>. Off by default: it changes the model's
numerics, and that is the user's call to make.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import types

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.parameter import BasevLLMParameter

from cuda_exl3 import ops
from cuda_exl3 import env as _env

logger = init_logger(__name__)

TILE = 16
_CB_MCG = 1
_MCG_MULTIPLIER = 0xCBAC1FED


def online_bits() -> int | None:
    """Bitrate for online quantization, or None when the feature is off."""
    raw = _env.getenv("CUDA_EXL3_ONLINE_BITS")
    if not raw:
        return None
    try:
        bits = int(raw)
    except ValueError:
        raise ValueError(f"CUDA_EXL3_ONLINE_BITS must be an integer, got {raw!r}")
    if not 2 <= bits <= 8:
        raise ValueError(f"CUDA_EXL3_ONLINE_BITS must be 2..8, got {bits}")
    return bits


def shape_supported(k: int, n: int) -> bool:
    """EXL3 tiles are 16x16 and the input transform is a 128-wide Hadamard."""
    return k % 128 == 0 and n % TILE == 0 and k >= 128 and n >= TILE


def _load_quantizer():
    """Import ExLlamaV3's trellis encoder without its generator dependencies.

    The package __init__ pulls in filter machinery (kbnf, formatron) that is
    unrelated to quantization and often absent, so load the one module directly.
    """
    root = _env.getenv("CUDA_EXL3_EXLLAMAV3_PATH")
    if root is None:
        for cand in ("/opt/exllamav3-python", "/home/shadeform/vllm/exllamav3",
                     "/home/shadeform/exllamav3"):
            if os.path.isfile(os.path.join(
                    cand, "exllamav3/modules/quant/exl3_lib/quantize.py")):
                root = cand
                break
    if root is None:
        try:
            spec = importlib.util.find_spec("exllamav3")
        except (ValueError, ImportError):
            spec = None                     # namespace stub with no __spec__
        if spec and spec.submodule_search_locations:
            root = os.path.dirname(list(spec.submodule_search_locations)[0])
    if root is None:
        raise RuntimeError(
            "CUDA_EXL3_ONLINE_BITS is set but ExLlamaV3 was not found. Install it, "
            "or point CUDA_EXL3_EXLLAMAV3_PATH at a checkout."
        )

    name = "exllamav3.modules.quant.exl3_lib.quantize"
    if name in sys.modules:
        return sys.modules[name].quantize_exl3

    if root not in sys.path:
        sys.path.insert(0, root)
    for pkg, rel in (
        ("exllamav3", "exllamav3"),
        ("exllamav3.modules", "exllamav3/modules"),
        ("exllamav3.modules.quant", "exllamav3/modules/quant"),
        ("exllamav3.modules.quant.exl3_lib", "exllamav3/modules/quant/exl3_lib"),
    ):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [os.path.join(root, rel)]
            sys.modules[pkg] = m
    path = os.path.join(root, "exllamav3/modules/quant/exl3_lib/quantize.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.quantize_exl3


def _cache_path(w: torch.Tensor, bits: int, prefix: str) -> str | None:
    d = _env.getenv("CUDA_EXL3_ONLINE_CACHE")
    if not d:
        return None
    os.makedirs(d, exist_ok=True)
    # Hash the weight itself: the same tensor always encodes the same way, and
    # this stays correct across renames, shard layouts and tp sizes.
    h = hashlib.blake2b(digest_size=16)
    h.update(str(tuple(w.shape)).encode())
    h.update(str(w.dtype).encode())
    flat = w.reshape(-1)
    probe = torch.cat([flat[:4096], flat[-4096:]]).to(torch.float32).cpu()
    h.update(probe.numpy().tobytes())
    h.update(str(bits).encode())
    return os.path.join(d, f"exl3_online_{h.hexdigest()}.pt")


def quantize_weight(w: torch.Tensor, bits: int, prefix: str) -> dict:
    """Encode one (out, in) bf16 weight to EXL3, uncalibrated."""
    cache = _cache_path(w, bits, prefix)
    if cache and os.path.exists(cache):
        try:
            return {k: v.cuda() for k, v in torch.load(cache, map_location="cpu").items()}
        except Exception as e:                       # pragma: no cover
            logger.warning("EXL3 online: cache read failed for %s (%s)", prefix, e)

    quantize_exl3 = _load_quantizer()
    dev = w.device
    # ExLlamaV3 wants (in_features, out_features) fp32.
    wt = w.t().contiguous().to(torch.float32)
    k = wt.shape[0]
    h_data = {
        "count": 0,
        "finalized": False,
        # A meta H selects the uncalibrated path -- no activations were captured.
        "H": torch.empty(k, k, device="meta", dtype=torch.float32),
        "device": dev,
    }
    args = {
        "K": bits,
        "seed": 0,
        "devices": [dev],
        "apply_out_scales": False,
        "zeros": False,
        "mcg": _MCG_MULTIPLIER,
        "sigma_reg": 0.025,
    }
    _, _, packed = quantize_exl3(wt, h_data, args, return_weight_q=False)[:3]
    out = {
        "trellis": packed["trellis"].contiguous(),
        "suh": packed["suh"].to(torch.float16).contiguous(),
        "svh": packed["svh"].to(torch.float16).contiguous(),
    }
    if cache:
        try:
            torch.save({k_: v.cpu() for k_, v in out.items()}, cache)
        except Exception as e:                       # pragma: no cover
            logger.warning("EXL3 online: cache write failed for %s (%s)", prefix, e)
    return out


class Exl3OnlineLinearMethod(LinearMethodBase):
    """Holds a bf16 weight through loading, then encodes it to EXL3 in place.

    vLLM's loader fills an ordinary weight; the encode happens in
    process_weights_after_loading, after which the bf16 copy is dropped and the
    layer runs the same kernel as a checkpoint-quantized one.
    """

    def __init__(self, quant_config, prefix: str, bits: int):
        self.quant_config = quant_config
        self.prefix = prefix
        self.bits = bits

    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        weight = torch.nn.Parameter(
            torch.empty(sum(output_partition_sizes), input_size_per_partition,
                        dtype=params_dtype),
            requires_grad=False,
        )
        # input_dim/output_dim let vLLM shard it exactly as an unquantized layer.
        from vllm.model_executor.utils import set_weight_attrs

        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        layer.exl3_online_shards = list(output_partition_sizes)

    def process_weights_after_loading(self, layer) -> None:
        w = layer.weight.data
        n_total, k = w.shape
        freed = w.numel() * w.element_size()
        shards = getattr(layer, "exl3_online_shards", [n_total])

        # Each shard is encoded on its own: they are separate matrices that the
        # checkpoint would have quantized separately, and suh differs per shard.
        parts, suhs, svhs, off = [], [], [], 0
        for n in shards:
            enc = quantize_weight(w[off:off + n].contiguous(), self.bits,
                                  f"{self.prefix}[{off}:{off + n}]")
            parts.append(enc["trellis"])
            suhs.append(enc["suh"])
            svhs.append(enc["svh"])
            off += n

        trellis = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        svh = torch.cat(svhs) if len(svhs) > 1 else svhs[0]
        suh = torch.stack(suhs)

        # Drop every reference to the bf16 weight and return its blocks to the
        # driver. Without the empty_cache the memory stays in torch's caching
        # allocator, vLLM's profiler still counts it, and the whole point of the
        # exercise -- the freed VRAM -- never shows up in the KV cache size.
        del layer.weight
        del w
        torch.cuda.empty_cache()
        layer.exl3_online_freed_bytes = freed

        layer.register_parameter(
            "trellis", torch.nn.Parameter(trellis, requires_grad=False))
        layer.register_parameter(
            "suh", torch.nn.Parameter(suh, requires_grad=False))
        layer.register_parameter(
            "svh", torch.nn.Parameter(svh, requires_grad=False))
        layer.exl3_group_n = list(shards)
        layer.exl3_cb = _CB_MCG
        layer.exl3_k = k
        layer.exl3_n_total = n_total

        try:
            from vllm.config import get_current_vllm_config

            sched = get_current_vllm_config().scheduler_config
            max_tokens = max(sched.max_num_batched_tokens, sched.max_num_seqs)
            ops.reserve(svh, max_tokens, k, n_total, len(shards))
        except Exception as e:
            logger.debug("EXL3 online %s: pre-reserve skipped (%s)", self.prefix, e)

    def apply(self, layer, x, bias=None):
        out = ops.exl3_linear(x, layer.trellis.data, layer.suh.data,
                              layer.svh.data, layer.exl3_group_n, layer.exl3_cb)
        return out if bias is None else out + bias
