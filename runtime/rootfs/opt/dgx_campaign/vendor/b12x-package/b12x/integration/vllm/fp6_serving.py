"""Framework enablement layer for serving b12x MX-FP6 (W6A6/W6A8) checkpoints.

This is the reference surface a vLLM / SGLang quantization backend calls to run an
FP6 checkpoint produced by :mod:`b12x.quantization.mxfp6.fp6_safetensors_export`.
It is framework-agnostic (no vLLM/SGLang imports) so it can be wired into any
fork; :mod:`b12x.integration.vllm.plugin` is the concrete vLLM adapter.

Enablement is twofold (mirrors how the FP4 b12x path is selected framework-side):

1. **Env gate** — ``B12X_ENABLE_FP6=1`` must be set. If unset,
   :func:`should_use_b12x_fp6` returns ``False``
   and the framework falls back to its native path.
2. **Checkpoint detection** — ``config.json`` carries
   ``quantization_config={"quant_method": "modelopt", "quant_algo": "W6A6", ...}``.

Exact FP6 call contract
-----------------------

**MoE** (gated SiLU). All tensors on CUDA; ``E`` experts, hidden ``K``,
intermediate ``N``; FP6 codes pack 4 values into 3 bytes (``3*dim/4``); block
scales are UE8M0 (``float8_e8m0fnu`` bytes) at ``sf_vec_size=32``:

* ``hidden_states``  ``(M, K)``      bfloat16 activations (quantized to FP6 in-kernel)
* ``topk_weights``   ``(M, topk)``   float32 router weights
* ``topk_ids``       ``(M, topk)``   int32 expert ids
* prepared experts from :func:`b12x.moe.fused_moe.prepare_weights` with an
  ``MXFP6_E8M0_K32`` source and FC1 rows in ``[up; gate]``
* output ``(M, K)`` bfloat16.

Routing is the framework's responsibility.  :class:`B12XFP6MoEMethod.apply`
consumes ``topk_ids``/``topk_weights`` and returns the routed-and-combined output.

**Dense linear** ``y = x @ W.T``:

* ``x`` ``(M, in_features)`` bfloat16 -> ``y`` ``(M, out_features)`` bfloat16
* weight from :func:`b12x.quantization.mxfp6.load_fp6_dense_checkpoint` as an
  :class:`~b12x.quantization.mxfp6.fp6_dense_weights.FP6DenseWeight`.

End-to-end flow
---------------

    pip install b12x
    python scripts/quantize_model_fp6.py --model <bf16 model> --out <fp6 model> --arch auto
    export B12X_ENABLE_FP6=1
    vllm serve <fp6 model>      # plugin detects W6A6 and calls b12x
"""
from __future__ import annotations

import os
from typing import Any, Optional

import torch

ENABLE_ENV = "B12X_ENABLE_FP6"
QUANT_METHOD = "modelopt"
QUANT_ALGO = "W6A6"

# Weight planning is shared because these immutable checkpoint geometry records
# are identical across layers. Each registered layer generation holds its own
# exact-M plans and runtime scratch, filled in place by preparation.

# Deduped fused_moe weight plans, keyed by geometry.  Every MoE layer of a
# model shares one plan object, which is what makes the scratch cache above
# actually shared (its key includes ``id(weight_plan)``) and keeps
# the prepared expert owners and execution plans reference the same canonical
# weight plan.
_WEIGHT_PLAN_CACHE: dict[tuple, Any] = {}


def get_fp6_moe_weight_plan(
    *,
    source_format: str,
    activation: str,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> Any:
    """One shared ``fused_moe.plan_weights`` result per (geometry, activation)."""
    from b12x.moe import fused_moe

    key = (
        source_format,
        activation,
        int(num_experts),
        int(hidden_size),
        int(intermediate_size),
    )
    plan = _WEIGHT_PLAN_CACHE.get(key)
    if plan is None:
        source = fused_moe.PackedSource(
            format=fused_moe.PackedSourceFormat(source_format),
            w13_layout=fused_moe.W13Layout.W13,
        )
        plan = fused_moe.plan_weights(
            source=source,
            activation=fused_moe.ActivationSpec(
                mode=fused_moe.ActivationMode.A8,
                nonlinearity=activation,
                io_dtype=torch.bfloat16,
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            ),
        )
        _WEIGHT_PLAN_CACHE[key] = plan
    return plan


def _env_truthy(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in ("1", "true", "yes", "on")


def is_b12x_fp6_enabled() -> bool:
    """True iff ``B12X_ENABLE_FP6`` is set."""
    return _env_truthy(ENABLE_ENV)


def _quant_config(config: Any) -> Optional[dict]:
    qc = (
        config.get("quantization_config")
        if isinstance(config, dict)
        else getattr(config, "quantization_config", None)
    )
    if qc is None:
        return None
    if isinstance(qc, dict):
        return qc
    if hasattr(qc, "to_dict"):
        return qc.to_dict()
    try:
        return dict(vars(qc))
    except TypeError:
        return None


def is_b12x_fp6_checkpoint(config: Any) -> bool:
    """True iff ``config`` declares a b12x FP6 (modelopt + W6A6) quantization."""
    qc = _quant_config(config)
    if not qc:
        return False
    return (
        str(qc.get("quant_method", "")).lower() == QUANT_METHOD
        and str(qc.get("quant_algo", "")).upper() == QUANT_ALGO
    )


def should_use_b12x_fp6(config: Any) -> bool:
    """Gate: env enabled AND the checkpoint is an FP6 checkpoint."""
    return is_b12x_fp6_enabled() and is_b12x_fp6_checkpoint(config)


def kernel_source_format_for_moe(_checkpoint_source_format: str) -> str:
    """Map a checkpoint ``source_format`` to the ``w6a8_mx`` fused_moe tag."""
    # The w6a8_mx preparation path requires the mxfp6_e2m3 source tag regardless
    # of whether the checkpoint was exported as mxfp6_default or mxfp6_w6a8.
    return "mxfp6_e2m3"


class B12XFP6MoEMethod:
    """FP6 routed-MoE owner holding exact-M plans for its lifetime.

    Plans are declared once, on the first :meth:`get_b12x_preparation_units`
    call, and cached on this instance; later calls reuse them. ``apply`` runs
    inside vLLM's own MoE custom op, so it may read the cached plans and
    scratch directly instead of resolving them through a custom op body.
    """

    def __init__(
        self,
        experts_prepared: Any,
        weight_plan: Any,
        *,
        input_scales_static: bool = True,
        apply_router_weight_on_input: bool = False,
    ):
        self.experts_prepared = experts_prepared
        self.weight_plan = weight_plan
        self.input_scales_static = input_scales_static
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self._declaration: Any = None
        self._topk: Optional[int] = None
        self._plans: dict[int, Any] = {}
        self._scratch: dict[int, tuple[torch.Tensor, ...]] = {}
        self._request_name = f"fp6_moe:{id(self)}"

    def get_b12x_preparation_units(self, layer, workload):
        """Declare this MoE layer's exact-M plans for the weights stage."""
        del layer
        if workload.stage != "weights":
            return ()
        from b12x.moe import fused_moe
        from b12x.preparation import PreparedCall
        from vllm.utils.b12x import B12xPreparationUnit

        if workload.output_dtype != self.weight_plan.activation.io_dtype:
            raise ValueError("FP6 MoE output dtype differs from its loaded contract")
        if self._declaration is None:
            counts = tuple(sorted({int(count) for count in workload.token_counts}))
            if not counts or counts[0] <= 0:
                raise ValueError("FP6 MoE preparation requires positive token counts")
            self._topk = int(getattr(self.weight_plan.geometry, "top_k", 0) or 8)
            self._declaration = fused_moe.plan_execution(
                experts=self.experts_prepared,
                capacity=fused_moe.ExecutionCapacity(
                    max_tokens=counts[-1], top_k=self._topk, warmup_token_counts=counts,
                    route_num_experts=0,
                ),
                routing=fused_moe.RoutingSpec(
                    apply_router_weight_on_input=self.apply_router_weight_on_input,
                ),
            )
            variants = getattr(self._declaration, "variants", None)
            self._plans = (
                {int(count): child for count, child in variants.items()}
                if variants is not None
                else {counts[-1]: self._declaration}
            )
        declaration = self._declaration
        topk = self._topk

        def call(tokens):
            def prepare(state):
                device = self.experts_prepared.device
                scratch = tuple(
                    torch.empty(spec.shape, dtype=spec.dtype, device=device)
                    for spec in state.scratch.scratch_specs()
                )
                hidden = torch.empty(
                    (tokens, self.experts_prepared.hidden_size),
                    dtype=self.weight_plan.activation.io_dtype,
                    device=device,
                )
                activation_source = torch.empty_like(hidden).normal_(mean=0.0, std=0.125)
                output = torch.empty_like(hidden)
                route_rows = torch.arange(
                    tokens, dtype=torch.int32, device=device
                ).unsqueeze(1)
                route_columns = torch.arange(
                    topk, dtype=torch.int32, device=device
                ).unsqueeze(0)
                route_ids = (route_rows + route_columns).remainder_(
                    self.experts_prepared.num_experts
                ).contiguous()
                route_logits = (
                    route_rows.to(dtype=torch.float32) * 0.03125
                    + route_columns.to(dtype=torch.float32) * 0.125
                )
                route_weights = torch.softmax(route_logits, dim=-1).contiguous()
                ids = torch.empty_like(route_ids)
                weights = torch.empty_like(route_weights)

                def reset() -> None:
                    output.zero_()
                    for buffer in scratch:
                        buffer.zero_()

                def produce() -> None:
                    hidden.copy_(activation_source)
                    ids.copy_(route_ids)
                    weights.copy_(route_weights)

                def restore() -> None:
                    reset()
                    produce()

                binding = state.bind(
                    scratch=scratch,
                    a=hidden,
                    experts=self.experts_prepared,
                    topk_weights=weights,
                    topk_ids=ids,
                    output=output,
                    input_scales_static=self.input_scales_static,
                )
                return PreparedCall(
                    run=binding.run,
                    output=output,
                    produce=produce,
                    reset=reset,
                    restore=restore,
                    owners=(
                        scratch,
                        hidden,
                        activation_source,
                        output,
                        route_ids,
                        route_weights,
                        ids,
                        weights,
                        binding,
                    ),
                )
            return prepare

        if hasattr(declaration, "token_counts"):
            calls = {count: call(count) for count in declaration.token_counts}
            benchmark_calls = {
                count: call(count) for count in declaration.token_counts
            }
            request = declaration.request(
                name=self._request_name,
                prepare_calls=calls,
                benchmark_calls=benchmark_calls,
            )
        else:
            (tokens,) = self._plans
            request = declaration.request(
                name=self._request_name,
                prepare_call=call(tokens),
                benchmark_call=call(tokens),
            )
        return (B12xPreparationUnit(
            name="FP6_MOE", key=(self._request_name, tuple(sorted(self._plans))),
            requests=(request,), stage="weights",
            autotune=not workload.eager_only,
        ),)

    def _scratch_for(self, tokens: int, plan: Any) -> tuple[torch.Tensor, ...]:
        """Runtime scratch for an exact-M plan; allocated once and reused.

        Called only from :meth:`apply`, after the plan has been prepared, so
        ``require_prepared`` always resolves a live state here.
        """
        scratch = self._scratch.get(tokens)
        if scratch is None:
            from b12x.preparation import require_prepared

            state = require_prepared(plan, "moe.decode", self.experts_prepared.device)
            scratch = tuple(
                torch.empty(spec.shape, dtype=spec.dtype, device=self.experts_prepared.device)
                for spec in state.scratch.scratch_specs()
            )
            self._scratch[tokens] = scratch
        return scratch

    def apply(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        apply_router_weight_on_input: bool = False,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run only an exact-M plan prepared before eager/capture work."""
        from b12x.moe import fused_moe

        m = int(hidden_states.shape[0])
        if bool(apply_router_weight_on_input) != self.apply_router_weight_on_input:
            raise ValueError("apply_router_weight_on_input differs from prepared FP6 MoE")
        try:
            plan = self._plans[m]
        except KeyError:
            raise RuntimeError(f"FP6 MoE exact-M={m} was not prepared") from None
        scratch = self._scratch_for(m, plan)
        if output is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("B12X FP6 MoE requires caller output during CUDA capture")
            output = torch.zeros(
                m, hidden_states.shape[1], dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        if topk_ids.dtype not in (torch.int32, torch.int64) or not topk_ids.is_contiguous():
            raise TypeError("B12X FP6 MoE requires contiguous int32 or int64 topk_ids")
        if topk_weights.dtype != torch.float32 or not topk_weights.is_contiguous():
            raise TypeError("B12X FP6 MoE requires contiguous float32 topk_weights")
        binding = fused_moe.bind(
            plan, scratch=scratch, a=hidden_states,
            experts=self.experts_prepared, topk_weights=topk_weights,
            topk_ids=topk_ids, output=output,
            input_scales_static=self.input_scales_static,
        )
        return fused_moe.run(binding=binding)



class B12XFP6LinearMethod:
    """Reference dense-linear method backed by :func:`dense_fp6_linear`."""

    def __init__(self, weight):
        self.weight = weight

    def apply(
        self, x: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute ``y = x @ W.T`` in MX-FP6; returns ``(M, out_features)`` bf16."""
        from b12x.quantization.mxfp6 import dense_fp6_linear

        return dense_fp6_linear(x, self.weight, out=out)


def load_b12x_fp6_moe_methods(
    model_path: str,
    *,
    activation: str = "silu",
    device: torch.device | str = "cuda",
    limit_layers: Optional[int] = None,
) -> dict[int, B12XFP6MoEMethod]:
    """Load every routed-MoE layer as ``{layer_index: B12XFP6MoEMethod}``."""
    from b12x.moe import fused_moe
    from b12x.quantization.mxfp6 import load_fp6_moe_checkpoint

    layers = load_fp6_moe_checkpoint(
        model_path, activation=activation, device=device, limit_layers=limit_layers
    )
    out: dict[int, B12XFP6MoEMethod] = {}
    for layer_idx, weights in layers.items():
        kernel_src = kernel_source_format_for_moe(weights.source_format)
        weight_plan = get_fp6_moe_weight_plan(
            source_format=kernel_src,
            activation=weights.activation,
            num_experts=weights.num_experts,
            hidden_size=weights.k,
            intermediate_size=weights.n,
        )
        # Artifact blockscales are swizzled; prepare_weights wants unswizzled.
        from b12x._lib.fp6 import unswizzle_mxfp6_scales

        def _unswizzle_grid(swizzled: torch.Tensor, rows: int, blocks: int) -> torch.Tensor:
            return torch.stack(
                [
                    unswizzle_mxfp6_scales(swizzled[eid], rows, blocks)
                    for eid in range(swizzled.shape[0])
                ]
            ).contiguous()

        prepared = fused_moe.prepare_weights(
            plan=weight_plan,
            weights=fused_moe.PackedWeights(
                w13=weights.w1_fp6,
                w2=weights.w2_fp6,
                w13_block_scales=_unswizzle_grid(
                    weights.w1_blockscale, 2 * weights.n, weights.k // 32
                ),
                w2_block_scales=_unswizzle_grid(
                    weights.w2_blockscale, weights.k, weights.n // 32
                ),
                w13_global_scales=weights.w1_alphas,
                w2_global_scales=weights.w2_alphas,
                input_scale=weights.a1_gscale,
                intermediate_scale=weights.a2_gscale,
            ),
        )
        method = B12XFP6MoEMethod(prepared, weight_plan)
        from vllm.utils.b12x import register_b12x_unit_provider
        register_b12x_unit_provider(method)
        out[layer_idx] = method


def load_b12x_fp6_linear_methods(
    model_path: str,
    *,
    device: torch.device | str = "cuda",
) -> dict[str, B12XFP6LinearMethod]:
    """Load every FP6 dense linear as ``{module_name: B12XFP6LinearMethod}``."""
    from b12x.quantization.mxfp6 import load_fp6_dense_checkpoint

    weights = load_fp6_dense_checkpoint(model_path, device=device)
    return {name: B12XFP6LinearMethod(w) for name, w in weights.items()}
