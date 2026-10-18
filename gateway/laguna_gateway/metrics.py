"""Privacy-safe Prometheus metrics for the gateway."""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


class GatewayMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self.active = Gauge(
            "laguna_gateway_active_requests",
            "Requests currently holding an upstream inference slot.",
            registry=self.registry,
        )
        self.waiting = Gauge(
            "laguna_gateway_waiting_requests",
            "Requests waiting for an upstream inference slot.",
            registry=self.registry,
        )
        self.admitted_tokens = Gauge(
            "laguna_gateway_admitted_tokens",
            "Prompt plus output-reserve tokens held by active requests.",
            registry=self.registry,
        )
        self.requests = Counter(
            "laguna_gateway_requests_total",
            "Completed gateway requests.",
            ("outcome", "stream"),
            registry=self.registry,
        )
        self.profile_requests = Counter(
            "laguna_gateway_profile_requests_total",
            "Completed requests by emulated harness effort profile.",
            ("profile", "stream"),
            registry=self.registry,
        )
        self.finish_reasons = Counter(
            "laguna_gateway_finish_reasons_total",
            "Original upstream finish reasons before any safety repair.",
            ("reason", "profile"),
            registry=self.registry,
        )
        token_buckets = (
            64,
            256,
            512,
            1_024,
            2_048,
            4_096,
            8_192,
            12_288,
            16_384,
            24_576,
            32_768,
        )
        self.completion_tokens = Histogram(
            "laguna_gateway_completion_tokens",
            "Completion tokens returned by the upstream and safety passes.",
            ("profile",),
            buckets=token_buckets,
            registry=self.registry,
        )
        self.reasoning_tokens = Histogram(
            "laguna_gateway_reasoning_tokens",
            "Tokenizer-backed hidden reasoning tokens per non-streaming response.",
            ("profile",),
            buckets=token_buckets,
            registry=self.registry,
        )
        self.output_budget_fraction = Histogram(
            "laguna_gateway_output_budget_fraction",
            "Completion tokens divided by the effective per-turn output cap.",
            ("profile",),
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98, 1.0),
            registry=self.registry,
        )
        self.tool_call_responses = Counter(
            "laguna_gateway_tool_call_responses_total",
            "Responses containing at least one model tool call.",
            ("profile",),
            registry=self.registry,
        )
        self.queue_wait = Histogram(
            "laguna_gateway_queue_wait_seconds",
            "Time spent waiting for an upstream slot.",
            buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60, 180, 600, 1800),
            registry=self.registry,
        )
        self.stream_heartbeats = Counter(
            "laguna_gateway_stream_heartbeats_total",
            "SSE keepalive comments emitted while a streaming request was idle.",
            ("api", "phase"),
            registry=self.registry,
        )
        self.stream_pending_cancellations = Counter(
            "laguna_gateway_stream_pending_cancellations_total",
            "Clients disconnected before their streaming request reached upstream.",
            ("api",),
            registry=self.registry,
        )
        self.upstream_aborts = Counter(
            "laguna_gateway_upstream_aborts_total",
            "Explicit upstream aborts issued for interrupted streaming requests.",
            ("api", "outcome"),
            registry=self.registry,
        )
        self.upstream_latency = Histogram(
            "laguna_gateway_upstream_seconds",
            "Wall time of upstream model calls.",
            ("purpose",),
            buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 180, 600, 1800),
            registry=self.registry,
        )
        self.original_prompt_tokens = Histogram(
            "laguna_gateway_original_prompt_tokens",
            "Tokenizer-backed prompt size before emergency compaction.",
            buckets=(
                1_000,
                8_000,
                32_000,
                64_000,
                100_000,
                160_000,
                200_000,
                220_000,
                240_000,
                262_144,
            ),
            registry=self.registry,
        )
        self.compactions = Counter(
            "laguna_gateway_compactions_total",
            "Emergency context compaction outcomes.",
            ("outcome",),
            registry=self.registry,
        )
        self.responses_output_adjustments = Counter(
            "laguna_gateway_responses_output_adjustments_total",
            "Responses requests whose output cap was adapted to remaining context.",
            registry=self.registry,
        )
        self.responses_context_retries = Counter(
            "laguna_gateway_responses_context_retries_total",
            "Authoritative upstream context rejections handled by a checkpoint retry.",
            ("outcome",),
            registry=self.registry,
        )
        self.responses_input_estimate_error = Histogram(
            "laguna_gateway_responses_input_estimate_absolute_error_tokens",
            "Absolute delta between the preflight estimate and upstream usage.",
            buckets=(0, 16, 64, 256, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768),
            registry=self.registry,
        )
        self.responses_input_estimate_direction = Counter(
            "laguna_gateway_responses_input_estimate_direction_total",
            "Direction of Responses preflight token-estimation error.",
            ("direction",),
            registry=self.registry,
        )
        self.responses_checkpoint_observations = Counter(
            "laguna_gateway_responses_checkpoint_observations_total",
            "Responses-only structured checkpoint shadow outcomes.",
            ("outcome",),
            registry=self.registry,
        )
        checkpoint_buckets = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
        self.responses_checkpoint_evidence_coverage = Histogram(
            "laguna_gateway_responses_checkpoint_evidence_coverage",
            "Fraction of source evidence referenced by retained checkpoint claims.",
            buckets=checkpoint_buckets,
            registry=self.registry,
        )
        self.responses_checkpoint_critical_coverage = Histogram(
            "laguna_gateway_responses_checkpoint_critical_coverage",
            "Fraction of deterministic critical evidence referenced by checkpoint claims.",
            buckets=checkpoint_buckets,
            registry=self.registry,
        )
        self.responses_checkpoint_literal_recall = Histogram(
            "laguna_gateway_responses_checkpoint_literal_recall",
            "Exact recall of deterministic task-critical literals in checkpoints.",
            buckets=checkpoint_buckets,
            registry=self.registry,
        )
        self.language_repairs = Counter(
            "laguna_gateway_language_repairs_total",
            "Czech final-answer repair outcomes.",
            ("outcome",),
            registry=self.registry,
        )
        self.reasoning_restores = Counter(
            "laguna_gateway_reasoning_restores_total",
            "Assistant tool turns whose hidden reasoning was restored in memory.",
            registry=self.registry,
        )
        self.reasoning_captures = Counter(
            "laguna_gateway_reasoning_captures_total",
            "Assistant tool turns whose hidden reasoning was cached in memory.",
            registry=self.registry,
        )
        self.reasoning_tokenize_failures = Counter(
            "laguna_gateway_reasoning_tokenize_failures_total",
            "Hidden-reasoning token counts that could not be measured.",
            registry=self.registry,
        )
        self.truncation_repairs = Counter(
            "laguna_gateway_truncation_repairs_total",
            "Reasoning-only length truncation repair outcomes.",
            ("outcome", "profile"),
            registry=self.registry,
        )
        self.tokenize_failures = Counter(
            "laguna_gateway_tokenize_failures_total",
            "Failed exact tokenizer calls that used a conservative estimate.",
            registry=self.registry,
        )
        # Materialize the bounded label space so a fresh post-boot audit sees
        # the full metric contract before the first real Jarvis request.
        for profile in ("standard", "xhigh", "max"):
            for stream in ("false", "true"):
                self.profile_requests.labels(
                    profile=profile,
                    stream=stream,
                ).inc(0)
            for reason in (
                "stop",
                "length",
                "tool_calls",
                "content_filter",
                "error",
                "abort",
                "repetition",
                "unknown",
            ):
                self.finish_reasons.labels(
                    reason=reason,
                    profile=profile,
                ).inc(0)
            self.completion_tokens.labels(profile=profile)
            self.reasoning_tokens.labels(profile=profile)
            self.output_budget_fraction.labels(profile=profile)
            self.tool_call_responses.labels(profile=profile).inc(0)
            for outcome in ("success", "failed", "empty"):
                self.truncation_repairs.labels(
                    outcome=outcome,
                    profile=profile,
                ).inc(0)
        for outcome in ("valid", "invalid", "failed", "busy", "not_sampled"):
            self.responses_checkpoint_observations.labels(outcome=outcome).inc(0)
        for outcome in ("success", "failed"):
            self.responses_context_retries.labels(outcome=outcome).inc(0)
        for direction in ("exact", "over", "under"):
            self.responses_input_estimate_direction.labels(direction=direction).inc(0)

    def update_queue(
        self,
        active: int,
        waiting: int,
        admitted_tokens: int = 0,
    ) -> None:
        self.active.set(active)
        self.waiting.set(waiting)
        self.admitted_tokens.set(admitted_tokens)

    def record_completion(
        self,
        *,
        profile: str,
        finish_reason: str,
        completion_tokens: int,
        reasoning_tokens: int | None,
        effective_max_tokens: int,
        has_tool_calls: bool,
    ) -> None:
        self.profile_requests.labels(profile=profile, stream="false").inc()
        self.finish_reasons.labels(
            reason=finish_reason,
            profile=profile,
        ).inc()
        safe_completion_tokens = max(0, int(completion_tokens))
        self.completion_tokens.labels(profile=profile).observe(safe_completion_tokens)
        if reasoning_tokens is not None:
            self.reasoning_tokens.labels(profile=profile).observe(
                max(0, int(reasoning_tokens))
            )
        if effective_max_tokens > 0:
            self.output_budget_fraction.labels(profile=profile).observe(
                min(1.0, safe_completion_tokens / effective_max_tokens)
            )
        if has_tool_calls:
            self.tool_call_responses.labels(profile=profile).inc()

    def record_responses_checkpoint(
        self,
        *,
        outcome: str,
        evidence_coverage: float,
        critical_coverage: float,
        literal_recall: float,
    ) -> None:
        self.responses_checkpoint_observations.labels(outcome=outcome).inc()
        if outcome not in {"valid", "invalid"}:
            return
        self.responses_checkpoint_evidence_coverage.observe(
            min(1.0, max(0.0, float(evidence_coverage)))
        )
        self.responses_checkpoint_critical_coverage.observe(
            min(1.0, max(0.0, float(critical_coverage)))
        )
        self.responses_checkpoint_literal_recall.observe(
            min(1.0, max(0.0, float(literal_recall)))
        )

    def record_responses_input_estimate(
        self,
        *,
        estimated_tokens: int,
        actual_tokens: int,
    ) -> None:
        delta = int(estimated_tokens) - int(actual_tokens)
        direction = "exact" if delta == 0 else "over" if delta > 0 else "under"
        self.responses_input_estimate_direction.labels(direction=direction).inc()
        self.responses_input_estimate_error.observe(abs(delta))

    def render(self) -> bytes:
        return generate_latest(self.registry)
