# 1.2.7 live prefix-cache results

These observations were collected from **real mixed Hermes work**, while the
release still carried its development name **1.2.6x4**. Production 1.2.7 uses
the same native payload. This report separates measured token accounting from
the estimated effective-input speedup on the community flyer.

## Observed window and cohorts

28 October 2026, **07:59:13.869769–10:11:25 CET**: approximately **2h12m**.
There were **937 successful requests**, with 850 exact client/session joins.
No twelve-hour measurement is claimed. Runtime evidence contains aggregates
only; private prompts, responses, session IDs and client mappings are excluded.

| Population | Requests | Input tokens | Cached tokens | Cache hit | Median prefill |
| --- | ---: | ---: | ---: | ---: | ---: |
| All successful requests, including first loads | 937 | 90,856,475 | 86,395,378 | **95.09%** | 0.723 s |
| Returning work sessions | 201 | 34,019,979 | 33,727,210 | **99.14%** | **0.641 s** |
| Returning autonomous cron sessions | 613 | 53,531,742 | 51,832,776 | **96.83%** | **0.672 s** |
| Returning work + cron | 814 | 87,551,721 | 85,559,986 | **97.73%** | **0.661 s** |
| Long work contexts, above 131,072 input tokens | 121 | 26,244,766 | 26,148,370 | **99.63%** | **0.656 s** |

Cache hit is `sum(cached input tokens) / sum(input tokens)`, not the average of
per-request percentages. Returning rows exclude the first observed request of
each session. The long-context row is a **subset** of work sessions, not another
population to add. The combined row excludes six returning TUI requests. Other
unjoined traffic and first loads explain why the global figure is lower.

Prefill times are request prefill durations, not raw GPU throughput, decode
latency or a controlled equal-input comparison. Source counts are available in
[LIVE-WORKFLOW.json](../results/20261028-1.2.7/LIVE-WORKFLOW.json).

## 47% lower uncached input share

| Observed period | Requests | Input | Newly computed input | Uncached share |
| --- | ---: | ---: | ---: | ---: |
| Start → 09:17:47 CET | 509 | 47,727,606 | 3,011,116 | 6.309% |
| 09:17:47 → 10:11:25 CET | 428 | 43,128,869 | 1,449,981 | 3.362% |

`1 - (3.36197316% / 6.30896090%) = 46.71%`, rounded to **47% lower uncached input
share**. This compares two observed periods of the same running version.
Warmup and task mix also changed; it is not a randomized X3-versus-X4 effect or
proof that learning alone caused the difference.

## Estimated +67% effective prefill throughput

The flyer models the time to handle a mixture of recomputed and reused input,
using **2,000 tok/s** for computed tokens and **200,000 tok/s** for effective
cached-input handling. The latter is an accounting assumption, not measured GPU
prefill or a claim that RAM lookup itself runs at a fixed token speed.

```text
u = newly_computed_input / total_input
effective_input_tps(u) = 1 / (u / 2000 + (1-u) / 200000)
gain = effective_input_tps(later) / effective_input_tps(earlier) - 1
     = 67.4%  →  ESTIMATED +67%
```

This is **effective input throughput**, not decode TPS or complete-workflow
speed. It depends on the assumed two rates, ignores fixed overhead, and includes
warmup/workload differences. It demonstrates why avoiding repeated long
prefills can matter, even when the absolute cache-hit percentage changes only
slightly. No fabricated before/after benchmark is used.

## Runtime costs and remaining losses

| Observed item | Value |
| --- | ---: |
| Pressure placement decisions | 294 |
| Adaptive placements | 68 |
| Bounded donor moves | 18 |
| Planner CPU time, all sampled decisions | **46.08 ms total** |
| Planner median / p95 / max | **0.123 / 0.244 / 1.936 ms** |
| X4-specific short rescue / eligible fresh-short victims | **0 / 0** |
| Bounded ghost-history returns after physical eviction | 23 |
| Confirmed lost positions associated with those returns | 511,555 |

Planner timings exclude GPU copies and other engine work. The 68 adaptive
placements include inherited adaptive logic; they are not 68 X4 short rescues.
The new special short-rescue branch had no eligible event in this window.
The benefit of that branch must not be inferred from the overall hit rate.

Lost positions account for about **0.563 percentage points** of all input in
this window. Some repeat losses were medium-size contexts (~59K–87K), beyond
the ≤32K short-rescue scope. A finite pool cannot retain every paused job, and
the bounded ghost ring does not enumerate every possible miss. No automatic
native restart, failed restore or serving error was observed in this audit;
one output-budget warning and an interleaved malformed diagnostic record were
separate observations, not evidence of perfect logging or unlimited retention.

## Validation context

The previous deployment canary verified first load → exact repeat → next turn,
namespace isolation, reader cancellation followed by a successful request, and
both image routes. Those are correctness checks, not a new throughput campaign.
Offline planner tests compare fixed geometries separately:
[CPU timing](../results/20261028-1.2.7/PLANNER-TIMING.json).
They do not establish production decode gains.

All historic decode measurements remain accessible under their original
versions. The release is a balanced retention policy for mixed workloads;
neither cache-hit percentage nor one favorable decode cell alone defines the
quality of that policy.
