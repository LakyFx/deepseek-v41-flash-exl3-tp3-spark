# Completed-workflow benchmark: 1.2.2TF → 1.2.4

**Main recipe: [1.2.4 / CDEF](DS41F-1.2.4.md).** One matched-initial-input pass
per recipe, C1–C12, 78 scored tool workflows per recipe. All 14 recipes completed
78/78 tasks: **168 cells, 1,092 correct tasks, 4,846 model requests**.
Reasoning and tool-call tokens are included. No new model run was needed to
produce this report. [Choose a recipe](RECIPES.md) · [exact model, quant and
sampling](BENCHMARK-SETTINGS-20261020.md) · [CSV evidence](../results/20261021-workflow/).

## What changed in practice

Against the oldest matched benchmark, **1.2.4 used 14.35% less wave time**
(1,154.48 vs 1,347.87 s), or **16.75% more correct tasks/minute**. At C6,
weighted native decode increased **21.66%**, from 23.61 to 28.72 tok/s per request.
Against the new 1.2.3 reference, summed wave time fell **10.58%**.

These are observed whole-recipe results. Different output lengths and repair
paths contribute to elapsed time. CDEF wins total completed-work time here;
it does not lead every concurrency or every token-rate metric. C and CF remain
useful alternatives. Rejected tool attempts are counted, including ones repaired
before successful completion. They are not evidence of changed model quality.

## Aggregate concurrent work

All rates below are **generated completion tokens / complete wave wall time**.
The denominator includes queueing, prefill, tool execution and repairs. A wave
starts C workflows together; the number of active model streams varies as tools
run. “Peak wave” is the fastest **whole completed wave mean**, not an instantaneous
GPU decode peak. “Median wave” gives each of the 12 waves equal weight.

| Recipe | All waves, tok/s | Median wave, tok/s | Peak wave, tok/s (C) | 78 tasks, min:s | Correct | Rejected tools |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.2.2TF | 121.85 | 117.73 | 135.22 (C11) | 22:27.87 | 78/78 | 30 |
| Bert1 | 118.06 | 114.17 | 132.34 (C11) | 23:38.94 | 78/78 | 31 |
| xTF / 1.2.3 (historical) | 124.62 | 119.89 | 139.49 (C12) | 20:29.80 | 78/78 | 16 |
| 1.2.3 (new reference) | 122.41 | 122.71 | 135.08 (C11) | 21:31.15 | 78/78 | 24 |
| A | 125.72 | 124.44 | 136.22 (C11) | 20:43.51 | 78/78 | 18 |
| B | 126.64 | 123.79 | 138.20 (C11) | 21:31.21 | 78/78 | 28 |
| C | 137.76 | 134.39 | 153.16 (C11) | 19:41.29 | 78/78 | 29 |
| D | 120.28 | 120.56 | 135.70 (C11) | 22:58.62 | 78/78 | 27 |
| E | 122.92 | 120.61 | 135.27 (C10) | 22:39.96 | 78/78 | 33 |
| F | 122.04 | 121.09 | 132.83 (C12) | 21:57.75 | 78/78 | 29 |
| G | 122.50 | 120.95 | 137.58 (C12) | 21:19.77 | 78/78 | 22 |
| CF | 138.74 | 137.86 | 159.04 (C11) | 19:50.39 | 78/78 | 25 |
| DEF | 122.09 | 121.68 | 133.48 (C11) | 22:34.75 | 78/78 | 27 |
| **1.2.4 / CDEF** | 139.33 | 136.98 | 157.18 (C11) | 19:14.48 | 78/78 | 26 |

## Decode per request and first token after prefill

Native request decode excludes prefill. **Weighted** divides native generated
tokens by summed request decode seconds; overlapping requests contribute separate
seconds. **Median** and **peak** are over completed requests, not GPU samples.
Peak requests can be short; their lengths and concurrency are in aggregates.csv.

**Post-prefill latency** is the comparable median **upper bound**, in milliseconds:
`U = native_TTFT − native_total + (completion_tokens − 1) / native_decode_tps`.
It excludes scheduler queueing and prefill, and includes server preparation
before submission. It is not end-to-end TTFT. Native “prefilled” occurs after
first-token selection: neither the microsecond host event nor U measures exact
GPU prefill-finish → first-token computation time. That duration was not recorded.

| Recipe | Weighted decode, tok/s | Median request, tok/s | Peak request, tok/s | Post-prefill bound, ms | Prefill¹, tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.2.2TF | 19.68 | 20.78 | 107.88 | 4.14 | 847.64 |
| Bert1 | 19.91 | 20.73 | 98.36 | 3.91 | 841.00 |
| xTF / 1.2.3 (historical) | 20.08 | 20.98 | 108.65 | 3.72 | 860.99 |
| 1.2.3 (new reference) | 20.09 | 21.44 | 110.16 | 4.12 | 848.42 |
| A | 19.39 | 20.84 | 108.40 | 4.03 | 855.44 |
| B | 20.28 | 22.05 | 115.54 | 4.05 | 847.21 |
| C | 23.17 | 24.46 | 117.93 | 4.13 | 853.56 |
| D | 20.62 | 20.91 | 108.86 | 3.90 | 857.92 |
| E | 20.38 | 20.91 | 109.81 | 4.08 | 848.14 |
| F | 19.78 | 20.97 | 109.94 | 3.98 | 853.65 |
| G | 20.11 | 20.39 | 110.17 | 3.79 | 859.42 |
| CF | 23.14 | 23.75 | 117.42 | 4.04 | 856.10 |
| DEF | 20.10 | 20.66 | 111.06 | 3.91 | 855.62 |
| **1.2.4 / CDEF** | 22.58 | 23.30 | 117.63 | 4.06 | 855.82 |

¹ Computed prompt tokens / native cumulative prefill time. This workload includes
short cached continuations; these are **not cold long-context prefill** rates.
Its request-level prefill medians answer a different question and remain in CSV.

## C1–C12: the direct decode comparison

**Weighted native decode tok/s per request** in every cell. Final column is
computed-prefill throughput pooled across all 12 cells, not an arithmetic mean
of rates. Compare rows at the same C. [All cell medians, request peaks, aggregate
rates, elapsed times and post-prefill latencies](WORKFLOW-CELLS-20261021.md).

| Recipe | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | C12 | Prefill¹ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.2.2TF | 84.37 | 56.66 | 42.63 | 32.93 | 27.67 | 23.61 | 21.66 | 19.53 | 18.63 | 18.14 | 15.15 | 15.26 | 847.64 |
| Bert1 | 82.36 | 55.13 | 44.91 | 32.92 | 26.82 | 25.37 | 20.65 | 19.93 | 18.91 | 19.79 | 15.10 | 15.21 | 841.00 |
| xTF / 1.2.3 (historical) | 84.95 | 56.37 | 46.65 | 34.47 | 30.63 | 26.23 | 22.80 | 19.82 | 17.62 | 16.50 | 15.76 | 14.31 | 860.99 |
| 1.2.3 (new reference) | 84.96 | 57.50 | 43.41 | 36.96 | 30.75 | 24.31 | 21.96 | 20.52 | 18.89 | 16.96 | 15.70 | 15.50 | 848.42 |
| A | 83.69 | 56.67 | 41.21 | 33.40 | 27.58 | 24.54 | 23.48 | 19.05 | 17.85 | 16.48 | 15.22 | 14.69 | 855.44 |
| B | 86.16 | 57.34 | 43.49 | 33.82 | 30.50 | 25.61 | 24.44 | 19.44 | 18.77 | 16.61 | 15.60 | 16.05 | 847.21 |
| C | 91.46 | 61.90 | 49.48 | 40.51 | 31.65 | 32.26 | 28.47 | 22.73 | 21.99 | 20.47 | 17.36 | 17.62 | 853.56 |
| D | 84.96 | 56.56 | 52.31 | 39.42 | 29.36 | 23.72 | 22.76 | 21.14 | 18.79 | 17.91 | 15.67 | 15.39 | 857.92 |
| E | 84.54 | 56.97 | 48.44 | 34.28 | 29.04 | 26.76 | 21.74 | 21.02 | 18.68 | 16.25 | 17.85 | 14.81 | 848.14 |
| F | 84.15 | 56.65 | 44.52 | 35.56 | 29.02 | 24.77 | 21.71 | 19.90 | 18.78 | 17.04 | 16.40 | 15.22 | 853.65 |
| G | 83.83 | 56.75 | 41.76 | 37.83 | 29.67 | 24.85 | 23.44 | 20.61 | 19.44 | 16.30 | 15.43 | 14.36 | 859.42 |
| CF | 92.03 | 62.61 | 46.13 | 38.84 | 34.62 | 28.01 | 25.90 | 23.63 | 23.52 | 19.45 | 17.61 | 18.37 | 856.10 |
| DEF | 84.34 | 56.77 | 41.19 | 39.32 | 30.51 | 25.10 | 22.08 | 19.55 | 17.75 | 16.49 | 16.23 | 15.72 | 855.62 |
| **1.2.4 / CDEF** | 91.01 | 61.54 | 44.56 | 38.38 | 33.49 | 28.72 | 24.60 | 22.24 | 21.41 | 20.73 | 17.57 | 17.62 | 855.82 |

## Comparison contract and limits

- Same pinned DeepSeek-V4.1-Flash-1017 weights, EXL3 routed experts ~3.5101 bpw,
  TP3 on three DGX Sparks, 2,200 MHz GPU clocks, MAX reasoning, temperature 0.6,
  seed 42, effective top-k 0 / top-p 1 / min-p 0, max_tokens 65,536, chunk 2,048.
  The request body explicitly sets temperature/seed/max_tokens; the remaining
  sampler values are engine defaults. [Full settings](BENCHMARK-SETTINGS-20261020.md).
- Original 1.2.2TF: C12/R32, 8M configured KV, 2 GiB snapshots. Bert1: C16/R48,
  7M, 2 GiB. xTF/1.2.3 and new A–G/combinations: C12/R48, 7M, 3 GiB. C is the
  workload concurrency; configured C16 in Bert1 is capacity, not a C16 test.
- Historical original/Bert1/xTF rows retain the earlier pass. The separate
  new 1.2.3 reference uses common timing instrumentation. They are not duplicate
  measurements averaged into a single winner. All 78 initial body hashes are
  frozen and verified by the [public fixture](../benchmarks/workflow/).
- One pass per recipe; no confidence intervals. Initial inputs match, subsequent
  output lengths, tool traces and context growth vary. Do not infer an isolated
  cache, speculation, learning or quality effect from recipe comparisons.
- Native counter coverage matched request/token totals in every scored cell.
  `requests.csv`, `tasks.csv` and `native-counters.csv` permit independent
  aggregation. Original source-cell SHA256 digests bind the export to retained
  evidence; private conversations, addresses and runtime state are excluded.
- The older “~50% fewer errors” observation was 31 → 16 rejected tool attempts
  in Bert1 → historical xTF. Here the new reference has 24 and CDEF 26; all final
  tasks pass. This does not establish any accuracy improvement or degradation.
- D/E/F/G mechanism counters were not captured. Their code is present, but
  how often each path fired is not proven by this archive. CDEF contains E;
  sustained-workload gains beyond this pass remain unmeasured.
- Two preliminary G attempts stopped at the external monitor. Its guard was
  repaired to use the primary local monotonic timeline; the initial trigger's
  root cause remained unproven. Only the complete G pass is scored here.
- Wave time sums exclude startup and between-cell overhead. No full 7M capacity,
  vision throughput, instantaneous aggregate GPU peak or 12-hour stability result
  is claimed. API/gateway behavior is outside this native workflow benchmark.

## Reproduce and inspect

[All recipes and mechanisms](RECIPES.md) · [1.2.4 source/build/install](DS41F-1.2.4.md)
· [online-learning design](ADAPTIVE-SPECULATION.md) · [public benchmark procedure](../benchmarks/workflow/README.md).
Rebuild these Markdown tables with `python3 tools/render_workflow_report.py`.
The preserved [earlier three-way report](WORKFLOW-BENCHMARK-20261020.md) and
[older benchmark index](BENCHMARKS.md) retain their original scopes.
