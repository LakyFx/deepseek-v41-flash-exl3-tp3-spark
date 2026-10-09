"""Native engine-counter deltas; absent counters are unknown, never zero.

Only meaningful under the controller's exclusive ingress and unchanged engine
generation. Component rates divide summed request times, not GPU wall time.
"""
import math
import re

PREFIX = "vllm:"
COUNTERS = (
    "request_generation_tokens_count", "request_generation_tokens_sum",
    "request_prefill_kv_computed_tokens_sum", "request_prefill_time_seconds_sum",
    "request_decode_time_seconds_sum", "request_success_total",
    "spec_decode_num_accepted_tokens_total", "spec_decode_num_draft_tokens_total",
    "spec_decode_num_drafts_total")
GAUGES = ("num_requests_running", "num_requests_waiting")


def parse(text):
    wanted = set(COUNTERS + GAUGES)
    values = {}
    for line in text.splitlines():
        found = re.fullmatch(r"vllm:([^\s{]+)(?:\{.*\})?\s+([-+0-9.eE]+)(?:\s+\d+)?", line)
        if not found or found[1] not in wanted:
            continue
        number = float(found[2])
        if not math.isfinite(number) or number < 0:
            raise ValueError("invalid native engine counter")
        values[found[1]] = values.get(found[1], 0)+number
    if not set(GAUGES).issubset(values):
        raise ValueError("engine idle gauges absent")
    return values


def compare(before, after, *, expected_requests, same_generation, exclusive):
    if not same_generation or not exclusive:
        raise ValueError("native deltas require same exclusive engine generation")
    if expected_requests < 1:
        raise ValueError("positive request count required")
    deltas = {}
    for name in COUNTERS:
        if name not in before or name not in after:
            deltas[name] = None
            continue
        change = after[name]-before[name]
        if not math.isfinite(change) or change < 0:
            raise ValueError("engine counter reset or invalid delta")
        deltas[name] = change
    completed = deltas["request_generation_tokens_count"]
    idle_before = all(before.get(name) == 0 for name in GAUGES)
    idle_after = all(after.get(name) == 0 for name in GAUGES)
    coverage = idle_before and idle_after and completed == expected_requests
    def rate(numerator, denominator):
        a, b = deltas[numerator], deltas[denominator]
        return a/b if coverage and a is not None and b is not None and b > 0 else None
    return {"native_deltas": deltas, "coverage_verified": coverage,
            "idle_before": idle_before, "idle_after": idle_after,
            "expected_requests": expected_requests, "observed_requests": completed,
            "weighted_request_computed_prefill_tokens_per_second": rate(
                "request_prefill_kv_computed_tokens_sum", "request_prefill_time_seconds_sum"),
            "weighted_request_decode_tokens_per_second": rate(
                "request_generation_tokens_sum", "request_decode_time_seconds_sum"),
            "draft_acceptance_fraction": rate(
                "spec_decode_num_accepted_tokens_total", "spec_decode_num_draft_tokens_total"),
            "scope": "summed per-request engine time; report cohort wall throughput separately",
            "missing_or_overlapping_metrics": "UNKNOWN_NOT_ZERO_GAIN" if not coverage else None}
