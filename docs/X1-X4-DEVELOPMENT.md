# From 1.2.6 through X1–X4 to 1.2.7

Each row below preserves an independently buildable source recipe and launcher
profile. The trials share the same pinned 1017 weights, EXL3 quantization,
sampling, C12 capacity, R48 ceiling, 7M KV pool, 3 GiB checkpoints and MAX/8K
standalone vision. **1.2.7 is X4 with release metadata changed.**

## What changed, and why

| Stage | Change | Purpose and constraints |
| --- | --- | --- |
| 1.2.6 / 5c | Bounded pressure-aware idle KV relocation; 320 MiB/rank credit | Retain useful long prefixes when fragmentation prevents a contiguous reservation. No background compactor. |
| ABCD additions | A: activity-limited Engram NVMe keepalive. B: reusable checkpoint selection. C: CUDA-graph storage ownership guard. D: short-prefix continuation reuse. | Avoid device wakeup overhead, retain compatible states, protect buffer lifetime and improve mixed-prefix handling. These letters are **local 1.2.6 tuning labels**, distinct from the earlier 1.2.4 A–G campaign. |
| **X1** | Native-aware checkpoint marks and balanced retention; bounded fresh-state admission; ghost diagnostics | Balance repeated long sessions and short first-use agent work within the same checkpoint budget. Three soft fresh entries per serving slot (36 at C12), scan bound 512. No cron-name priorities. |
| **X2** | Remove periodic expensive legacy shadow comparisons; add eligible verified-continuation donors and rejection counters | Keep the functional legacy fallback. Additional donor search runs only after original direct-donor/D paths; a continuation hint is never counted as a completed restore. |
| **X3** | Bounded CPU online prefix history and cost-aware physical placement | Learn verified returns, event spacing and successful prefill cost by size. Refine legal placement under pressure with hysteresis; keep existing restore authority and snapshot victim rules. |
| **X4** | Narrow rescue for recently returning ≤32K prefixes; native-compatible ghost filtering; first-failing rejection reasons | Relax only the fractional extra-loss allowance within the absolute 32K cap, without new proven long victims against baseline/X3. Reuse the existing candidate search and copy paths. |
| **1.2.7** | Promote the exact deployed X4 | No native code, settings, weights, gateway or sampling change during promotion. |

The earlier [14-recipe campaign](WORKFLOW-BENCHMARK-20261021.md), 1.2.5/K2a and
1.2.6 records remain unchanged. Label reuse across campaigns does not make two
different “B” patches equivalent.

## Buildable historical selectors

| Version | Native recipe | Launcher / template selector | Source files | Reference native SHA prefix |
| --- | --- | --- | ---: | --- |
| X1 | [`x1`](../tensorfold/variants/recipes/x1.json) | [`1.2.6x1`](../tensorfold/profiles/1.2.6x1.json) | 1,142 | `670ff128ca42` |
| X2 | [`x2`](../tensorfold/variants/recipes/x2.json) | [`1.2.6x2`](../tensorfold/profiles/1.2.6x2.json) | 1,142 | `068b2ff07533` |
| X3 | [`x3`](../tensorfold/variants/recipes/x3.json) | [`1.2.6x3`](../tensorfold/profiles/1.2.6x3.json) | 1,143 | `c82bed2cd9d9` |
| X4 | [`x4`](../tensorfold/variants/recipes/x4.json) | [`1.2.6x4`](../tensorfold/profiles/1.2.6x4.json) | 1,143 | `93bfa145f256` |
| Production | [`v127`](../tensorfold/variants/recipes/v127.json) | [`1.2.7`](../tensorfold/profiles/1.2.7.json) | 1,143 | `93bfa145f256` |

Every manifest binds a complete source-file map against the same frozen public
base archive; full changed files are stored as SHA256-addressed objects. X4 and
v127 have identical source maps and reference binary hashes. No old manifest
was overwritten. Historical trial receipts are under
[results/history](../results/20261028-1.2.7/history/).

To rebuild X2, for example, use the same installation sequence as 1.2.7, replacing
`--recipe v127` with `--recipe x2` and `--variant 1.2.7` with `--variant 1.2.6x2`.
Use new build/output directories and switch all three native ranks together.
These are real engine snapshots, not aliases pointing every old selector to X4.

## How the prefix learner works

1. Match real token-prefix lineage and namespace using existing verified state.
   A session label or diagnostic ghost entry cannot authorize a restore.
2. Keep bounded history: 2,048 four-way records and 64 namespace models within a
   256 KiB bound. Track verified return frequency/spacing, recency in request
   events and successful prefill-cost EWMAs in length buckets.
3. Use ghost-return feedback to adjust the balance of recency and frequency.
   Entries age by request progress, with no wall-clock eviction TTL. A numeric
   score is an estimated retention value, not a calibrated return probability.
4. At allocation pressure, compare only legal alternatives. The extra adaptive
   candidate limit is 128, inside a global 2,048-evaluation and cooperative 2 ms
   planner budget. Require a 5% modeled cost advantage before changing the plan.
5. Retain the old compatible plan when geometry or evidence is unsuitable. The
   ordinary extra-loss allowance is bounded by the lesser of 32K and the
   baseline-dependent allowance (floor 2K, fraction 1/8). X4 may use the absolute
   32K ceiling for a verified recent short return only with long-victim guards.

This bookkeeping is CPU-side and pressure/request driven. It adds no GPU
training, prompt classification, new save points or per-decoded-token work.
Physical moves still pay real copy cost and debit the existing 320 MiB/rank
credit. The 2 ms budget is cooperative, not a hard real-time preemption guarantee.

Neither learned metadata nor saved checkpoint objects reserve unlimited
physical KV. A compatible state can outlive its physical interval; diagnostics
distinguish that loss from a successful reusable prefix. Unknown new child-agent
prompts are naturally cold even when their parent's context remains reusable.

## Tests, observations and interpretation

The archive includes CPU/module check exit codes, build/log hashes, source
identities and the exact reference binaries for all four trials. X4 adds fixed
geometry timing/oracle checks, and its deployment canary verifies text, exact and
growing prefixes, namespace isolation, cancellation and both vision routes.
Published receipts remove local paths and private client data.

The X1–X4 development stage was assessed through real mixed operation rather
than repeatedly re-running a short synthetic cache test. Earlier audits found
physically evicted returns and costly shadow comparisons; X2 removed the latter,
while X3/X4 refined decisions and diagnostics. That progression does not imply
every old loss became avoidable. The final [live evidence](LIVE-CACHE-1.2.7.md)
publishes the remaining losses, denominators and the fact that the X4-specific
short-rescue branch had no eligible event in the selected window.

The new flyer reports **99.14% work-session reuse**, **96.83% cron reuse** and
**97.73% combined returning-workload reuse**. Its **47% lower uncached share**
compares observed time windows; **estimated +67% effective prefill throughput**
uses explicitly stated accounting rates. Neither is a controlled causal
comparison of all four trial versions or a new decode benchmark.
