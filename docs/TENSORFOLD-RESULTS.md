# Complete TensorFold TP3 test results

## New production observations and correctness repair

The [14 October real Hermes workload export](HERMES-REAL-WORKLOAD.md) contains X11c/C4/TF351 active-time aggregate rates across separate five-hour windows, including all available C1–C8 data. This is a passive production sample with different tasks and cache states, not another exclusive benchmark run. Its TF351 window used 10M and precedes the [short-prefill AOT repair](TENSORFOLD-HC-PREFILL-FIX.md). The saved controlled campaigns below remain unchanged.

[Recipe overview](TENSORFOLD-TP3.md) · [Run the same public adapters](TENSORFOLD-BENCHMARK.md) · [Numerical summary](../results/20261013-tensorfold/summary.json) · [Every exported receipt and hash](../results/20261013-tensorfold/evidence-manifest.json)

TP3, MAX, context 401,408, eight streams. Full V3 uses source `6b7c3ac6df05e8041a4a77c26152723c795fad42`; final V7 uses `c4844adfef8df07f2bd299a6a23e5d04a6ec43c0`. The published runtime snapshot `61357debe730ac9ffe20e760edcddbc253b0e525` adds monitor observations. All speed/quality runs used the 3.3M pool. TF351 10M qualification is a separate phase. C4 reference files were frozen and SHA-verified, not rerun at every native iteration.

**14 October correction:** all 270 measured Owen V3 decode requests and 36 final V7 requests generated only reasoning and hit the 256-token limit with zero final content. The old code/prose labels identify the prompt, not a completed answer. The numbers below remain valid short-reasoning decode measurements; do not advertise them as completed coding/prose throughput. Prefer the [audited completed Hermes C1/C3 comparison](BENCHMARK-CORRECTION-20261014.md) and the README's fresh C4 baseline. No new inference was performed for this correction.

## Relevant Hermes C1/C3 and cold-prefill comparison

| Metric / stage | C4 fresh reference | TF351 | TF29 |
|---|---:|---:|---:|
| Completed C1 workflow wall, s; final | 22.965 | 15.207 | 22.284 |
| Completed C3 workflow cohort wall, s; final | 39.003 | 35.601 | 43.666 |
| C1 request decode incl. reasoning/tools, tok/s; final | 62.319 | 85.001 | 88.900 |
| C3 request decode incl. reasoning/tools, tok/s; final | 31.826 | 34.491 | 41.809 |
| Cold 64K-C1 computed input/native prefill time, tok/s; V3 | 1,710.851 | 2,111.258 | 2,118.820 |
| Cold 128K-C3 computed input/native prefill time, tok/s; V3 | 1,638.884 | 2,040.969 | 2,056.489 |
| Hot 8K-C8 short probe aggregate output/wall, tok/s; V3 | 150.590 | 199.245 | 221.017 |

TF351 completed C1/C3 tasks in **33.8%/8.7% less time** than C4, with different generated output lengths. Its request decode rates increased **36.4%/8.4%**; these are weighted per-request rates, not aggregate concurrency scaling. Cold prefill increased **23.4%/24.5%** in the two Hermes regimes. C3 was measured both as scored agent workflows and as 128K cold/hot load probes; Owen has no C3 cell. [Receipts, hashes and calculations](../results/20261013-tensorfold/comparison-audit.json), [offline recalculation](../tools/audit_tensorfold_results.py).

## Full V3 matrix: three decode repeats, two cold repeats

| Version | Code C1 | Code C2 | Code C4 | Code C8 | Prose C1 | Prose C2 | Prose C4 | Prose C8 | Cold 32K | Cold 256K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C4-reference | 53.51 | 85.18 | 128.43 | 208.62 | 37.42 | 64.63 | 99.80 | 152.51 | 1,610.05 | 1,680.63 |
| TF351 | 84.40 | 161.45 | 255.80 | 392.87 | 83.13 | 159.71 | 259.21 | 393.33 | 2,129.50 | 1,976.48 |
| TF29 | 93.15 | 181.12 | 306.40 | 413.59 | 86.41 | 162.90 | 242.64 | 405.61 | 2,152.20 | 1,989.41 |

Decode values are **reasoning-only** aggregate tok/s over the shared first/last-token window after a discarded warmup. Code/prose name the input task. Cold values are actual rendered input / HTTP TTFT, not a native kernel bandwidth or output metric. Each receipt retains actual counts and TTFT. Native Hermes prefill is separate.

## Final V7 subset after short-continuation and tool-history repairs

| Version | Code C1 | Code C8 | Prose C1 | Prose C8 | Cold 32K | Workflow C1 wall | Workflow C3 wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF351 | 84.470 | 396.030 | 83.513 | 394.949 | 2,147.643 | 15.207 | 35.601 |
| TF29 | 93.327 | 416.233 | 86.963 | 408.805 | 2,161.993 | 22.284 | 43.666 |

One measured wave after warmup. C2/C4/256K were not rerun on final V7 and must not be represented as fresh final measurements. Full V3 results are retained rather than silently mixed into the final subset. Both final arms completed four of four Hermes workflows and three of three tool quality deltas.

## Cache semantics and the repaired failures

Early native runs invalidated retained neighbors, reused only rounded chunks and lost DSML tool separators during assistant-history serialization. Repairs protect retained extents, preserve complete accepted snapshots, support checked short continuations and roundtrip tool separators. V3 already fixes identical-prefix reuse; V7 fixes continuation/tool history. Strict partial-prefix replay can still differ numerically from cold execution; retained mismatches are failures, not evidence of bit-exact equivalence.

| Version / phase | Workflow C1 cached / newly computed | C1 output tokens | Workflow C3 cached / newly computed | C3 output tokens | Correct workflows |
|---|---|---:|---|---:|---|
| C4-reference V3 | 26,368 / 8,667 | 1003 | 50,688 / 25,807 | 2185 | 4/4 |
| TF351 V3 | 25,552 / 9,330 | 911 | 50,582 / 26,227 | 2650 | 4/4 |
| TF29 V3 | 8,182 / 8,354 | 550 | 33,204 / 25,251 | 2612 | 2/4 |
| TF351 final V7 | 26,446 / 8,352 | 877 | 51,476 / 25,249 | 2616 | 4/4 |
| TF29 final V7 | 27,818 / 8,352 | 1559 | 52,840 / 25,249 | 3621 | 4/4 |

Final TF351 hits 76.00%/67.09%; TF29 hits 76.91%/67.67%. **Both compute exactly 8,352/25,249 new input tokens.** The different percentage denominator includes different generated histories. TF29 emits 1,559/3,621 output tokens versus TF351 877/2,616, explaining why its faster raw decode does not guarantee shorter workflow time. Compare completed work, computed input, output length and end-to-end time together.

## Identical-prefix hot tests, V3

| Version | 64K-C1 hit / hot wall | 128K-C3 hit / hot wall | 8K-C8 hit / hot wall |
|---|---|---|---|
| C4-reference | 99.81% / 2.928 s | 99.92% / 5.153 s | 98.60% / 13.600 s |
| TF351 | 100.00% / 1.499 s | 100.00% / 3.005 s | 100.00% / 10.279 s |
| TF29 | 100.00% / 1.388 s | 100.00% / 2.682 s | 100.00% / 9.266 s |

## Bounded Q20 quality screen

| Version | Passed / attempted | Incomplete | Six code tasks | Inference time |
|---|---:|---:|---:|---:|
| C4-reference | 20/24 | 4 | 5/6 | 360.120 s |
| TF351 | 20/24 | 4 | 5/6 | 271.638 s |
| TF29 | 21/24 | 3 | 5/6 | 256.418 s |

No completed-but-failed noncode attempts were found in that captured subset. Length caps and one C4 timeout account for incompletes; these remain failures in the denominator. All six code attempts, including truncated ones, remain visible. TF29 scoring one additional completion does not establish better quantization quality. The native cache format/ancillary tensors/sampler also change, so this is not a clean bits-only comparison.

## C4 reference

[Preserved C4 / DS41F 1.0.0 recipe](DS41F-1.0.0.md). vLLM, Pollard experts, native FP8 KV. This fresh reference differs from the earlier host64k campaign; both are retained under their own dates.

| Incomplete attempt | Finish / error |
|---|---|
| evalplus-36-greedy | length / no transport error |
| ifeval-1001-greedy | length / no transport error |
| ifeval-1322-greedy | length / no transport error |
| long-digest-1-greedy | none / TimeoutError: timed out |

[Full C4 reference per-request receipts](../results/20261013-tensorfold/evidence/v3/results/C4-reference) · [Quality attempts](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference)

| Attempt | Finish | Noncode grade / code sandbox pending | Input / output |
|---|---|---|---:|
| [cs-deadline-order-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference/cs-deadline-order-greedy.json) | stop | pass | 168 / 165 |
| [digest-correction-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference/digest-correction-greedy.json) | stop | pass | 231 / 263 |
| [digest-correction-sample-1708](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference/digest-correction-sample-1708.json) | stop | pass | 231 / 300 |
| [digest-correction-sample-2026](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference/digest-correction-sample-2026.json) | stop | pass | 231 / 157 |
| [digest-duplicates-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference/digest-duplicates-greedy.json) | stop | pass | 243 / 604 |
| [digest-source-injection-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference/digest-source-injection-greedy.json) | stop | pass | 255 / 533 |
| [digest-uncertainty-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference/digest-uncertainty-greedy.json) | stop | pass | 231 / 461 |
| [evalplus-0-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference/evalplus-0-greedy.json) | stop | [code pass](../results/20261013-tensorfold/evidence/v3/quality-q20-code/C4-reference/evalplus-0-greedy.json) | 189 / 719 |
| [evalplus-11-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference/evalplus-11-greedy.json) | stop | [code pass](../results/20261013-tensorfold/evidence/v3/quality-q20-code/C4-reference/evalplus-11-greedy.json) | 139 / 1,006 |
| [evalplus-12-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference/evalplus-12-greedy.json) | stop | [code pass](../results/20261013-tensorfold/evidence/v3/quality-q20-code/C4-reference/evalplus-12-greedy.json) | 164 / 369 |
| [evalplus-3-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference/evalplus-3-greedy.json) | stop | [code pass](../results/20261013-tensorfold/evidence/v3/quality-q20-code/C4-reference/evalplus-3-greedy.json) | 180 / 320 |
| [evalplus-36-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference/evalplus-36-greedy.json) | length | fail / incomplete | 141 / 4,096 |
| [evalplus-7-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference/evalplus-7-greedy.json) | stop | [code pass](../results/20261013-tensorfold/evidence/v3/quality-q20-code/C4-reference/evalplus-7-greedy.json) | 156 / 160 |
| [ifeval-1001-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference/ifeval-1001-greedy.json) | length | fail / incomplete | 68 / 4,096 |
| [ifeval-102-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference/ifeval-102-greedy.json) | stop | pass | 103 / 1,687 |
| [ifeval-1075-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference/ifeval-1075-greedy.json) | stop | pass | 65 / 1,825 |
| [ifeval-122-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference/ifeval-122-greedy.json) | stop | pass | 64 / 3,834 |
| [ifeval-1307-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference/ifeval-1307-greedy.json) | stop | pass | 74 / 3,154 |
| [ifeval-1322-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference/ifeval-1322-greedy.json) | length | fail / incomplete | 63 / 4,096 |
| [long-digest-0-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference/long-digest-0-greedy.json) | stop | pass | 65,635 / 607 |
| [long-digest-1-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference/long-digest-1-greedy.json) | none | fail / incomplete | n/a / n/a |
| [tools-digest-0-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference/tools-digest-0-greedy.json) | stop | pass | 2,389 / 338 |
| [tools-digest-0-sample-1708](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference/tools-digest-0-sample-1708.json) | stop | pass | 2,479 / 591 |
| [tools-digest-0-sample-2026](../results/20261013-tensorfold/evidence/v3/quality-q20/C4-reference/tools-digest-0-sample-2026.json) | stop | pass | 2,467 / 421 |

## TF351

[Exact TF351 recipe](TF351.md). Pollard routed experts plus Mia ancillary tensors. Speed profile 3.3M; separately qualified selected pool 10M. Final short reasoning-only Owen C1/C8 decode is 84.470/396.030 for the code prompt and 83.513/394.949 for the prose prompt. Final completed Hermes C1/C3 workflows take 15.207/35.601 s.

| Incomplete attempt | Finish / error |
|---|---|
| evalplus-36-greedy | length / no transport error |
| ifeval-1001-greedy | length / no transport error |
| ifeval-122-greedy | length / no transport error |
| ifeval-1307-greedy | length / no transport error |

[Full TF351 per-request receipts](../results/20261013-tensorfold/evidence/v3/results/TF351) · [Quality attempts](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351)

| Attempt | Finish | Noncode grade / code sandbox pending | Input / output |
|---|---|---|---:|
| [cs-deadline-order-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351/cs-deadline-order-greedy.json) | stop | pass | 168 / 161 |
| [digest-correction-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351/digest-correction-greedy.json) | stop | pass | 231 / 229 |
| [digest-correction-sample-1708](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351/digest-correction-sample-1708.json) | stop | pass | 231 / 333 |
| [digest-correction-sample-2026](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351/digest-correction-sample-2026.json) | stop | pass | 231 / 222 |
| [digest-duplicates-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351/digest-duplicates-greedy.json) | stop | pass | 243 / 653 |
| [digest-source-injection-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351/digest-source-injection-greedy.json) | stop | pass | 255 / 206 |
| [digest-uncertainty-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351/digest-uncertainty-greedy.json) | stop | pass | 231 / 458 |
| [evalplus-0-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351/evalplus-0-greedy.json) | stop | [code pass](../results/20261013-tensorfold/evidence/v3/quality-q20-code/TF351/evalplus-0-greedy.json) | 189 / 601 |
| [evalplus-11-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351/evalplus-11-greedy.json) | stop | [code pass](../results/20261013-tensorfold/evidence/v3/quality-q20-code/TF351/evalplus-11-greedy.json) | 139 / 787 |
| [evalplus-12-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351/evalplus-12-greedy.json) | stop | [code pass](../results/20261013-tensorfold/evidence/v3/quality-q20-code/TF351/evalplus-12-greedy.json) | 164 / 396 |
| [evalplus-3-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351/evalplus-3-greedy.json) | stop | [code pass](../results/20261013-tensorfold/evidence/v3/quality-q20-code/TF351/evalplus-3-greedy.json) | 180 / 191 |
| [evalplus-36-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351/evalplus-36-greedy.json) | length | fail / incomplete | 141 / 4,096 |
| [evalplus-7-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351/evalplus-7-greedy.json) | stop | [code pass](../results/20261013-tensorfold/evidence/v3/quality-q20-code/TF351/evalplus-7-greedy.json) | 156 / 189 |
| [ifeval-1001-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351/ifeval-1001-greedy.json) | length | fail / incomplete | 68 / 4,096 |
| [ifeval-102-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351/ifeval-102-greedy.json) | stop | pass | 103 / 1,075 |
| [ifeval-1075-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351/ifeval-1075-greedy.json) | stop | pass | 65 / 2,059 |
| [ifeval-122-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351/ifeval-122-greedy.json) | length | fail / incomplete | 64 / 4,096 |
| [ifeval-1307-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351/ifeval-1307-greedy.json) | length | fail / incomplete | 74 / 4,096 |
| [ifeval-1322-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351/ifeval-1322-greedy.json) | stop | pass | 63 / 2,988 |
| [long-digest-0-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351/long-digest-0-greedy.json) | stop | pass | 65,635 / 344 |
| [long-digest-1-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351/long-digest-1-greedy.json) | stop | pass | 65,635 / 310 |
| [tools-digest-0-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351/tools-digest-0-greedy.json) | stop | pass | 2,503 / 479 |
| [tools-digest-0-sample-1708](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351/tools-digest-0-sample-1708.json) | stop | pass | 2,490 / 694 |
| [tools-digest-0-sample-2026](../results/20261013-tensorfold/evidence/v3/quality-q20/TF351/tools-digest-0-sample-2026.json) | stop | pass | 2,481 / 423 |

## TF29

[Exact TF29 recipe](TF29.md). Mia 2.9 bpw routed experts and ancillary tensors. Qualified speed/pool profile 3.3M; no TF29 10M claim. Earlier V3 strict workflows failed when the model requested the nonexistent `test_events.py` fixture path; that failure is retained. Final repaired history passed all four workflows.

| Incomplete attempt | Finish / error |
|---|---|
| evalplus-36-greedy | length / no transport error |
| ifeval-1001-greedy | length / no transport error |
| ifeval-1307-greedy | length / no transport error |

[Full TF29 per-request receipts](../results/20261013-tensorfold/evidence/v3/results/TF29) · [Quality attempts](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29)

| Attempt | Finish | Noncode grade / code sandbox pending | Input / output |
|---|---|---|---:|
| [cs-deadline-order-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29/cs-deadline-order-greedy.json) | stop | pass | 168 / 106 |
| [digest-correction-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29/digest-correction-greedy.json) | stop | pass | 231 / 367 |
| [digest-correction-sample-1708](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29/digest-correction-sample-1708.json) | stop | pass | 231 / 343 |
| [digest-correction-sample-2026](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29/digest-correction-sample-2026.json) | stop | pass | 231 / 345 |
| [digest-duplicates-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29/digest-duplicates-greedy.json) | stop | pass | 243 / 546 |
| [digest-source-injection-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29/digest-source-injection-greedy.json) | stop | pass | 255 / 944 |
| [digest-uncertainty-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29/digest-uncertainty-greedy.json) | stop | pass | 231 / 451 |
| [evalplus-0-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29/evalplus-0-greedy.json) | stop | [code pass](../results/20261013-tensorfold/evidence/v3/quality-q20-code/TF29/evalplus-0-greedy.json) | 189 / 665 |
| [evalplus-11-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29/evalplus-11-greedy.json) | stop | [code pass](../results/20261013-tensorfold/evidence/v3/quality-q20-code/TF29/evalplus-11-greedy.json) | 139 / 709 |
| [evalplus-12-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29/evalplus-12-greedy.json) | stop | [code pass](../results/20261013-tensorfold/evidence/v3/quality-q20-code/TF29/evalplus-12-greedy.json) | 164 / 304 |
| [evalplus-3-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29/evalplus-3-greedy.json) | stop | [code pass](../results/20261013-tensorfold/evidence/v3/quality-q20-code/TF29/evalplus-3-greedy.json) | 180 / 246 |
| [evalplus-36-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29/evalplus-36-greedy.json) | length | fail / incomplete | 141 / 4,096 |
| [evalplus-7-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29/evalplus-7-greedy.json) | stop | [code pass](../results/20261013-tensorfold/evidence/v3/quality-q20-code/TF29/evalplus-7-greedy.json) | 156 / 238 |
| [ifeval-1001-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29/ifeval-1001-greedy.json) | length | fail / incomplete | 68 / 4,096 |
| [ifeval-102-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29/ifeval-102-greedy.json) | stop | pass | 103 / 1,224 |
| [ifeval-1075-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29/ifeval-1075-greedy.json) | stop | pass | 65 / 3,456 |
| [ifeval-122-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29/ifeval-122-greedy.json) | stop | pass | 64 / 3,533 |
| [ifeval-1307-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29/ifeval-1307-greedy.json) | length | fail / incomplete | 74 / 4,096 |
| [ifeval-1322-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29/ifeval-1322-greedy.json) | stop | pass | 63 / 2,999 |
| [long-digest-0-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29/long-digest-0-greedy.json) | stop | pass | 65,635 / 510 |
| [long-digest-1-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29/long-digest-1-greedy.json) | stop | pass | 65,635 / 494 |
| [tools-digest-0-greedy](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29/tools-digest-0-greedy.json) | stop | pass | 2,381 / 326 |
| [tools-digest-0-sample-1708](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29/tools-digest-0-sample-1708.json) | stop | pass | 2,477 / 424 |
| [tools-digest-0-sample-2026](../results/20261013-tensorfold/evidence/v3/quality-q20/TF29/tools-digest-0-sample-2026.json) | stop | pass | 2,504 / 442 |

## TF351 10M capacity: separate stress qualification

| Test | Recorded outcome |
|---|---|
| Short C1 | 44 input / 44 output; 0.765 s |
| Cold long input | 131,939 input / 96 output; 60.875 s end-to-end |
| Eight simultaneous long inputs | 384,063 total input; last completed at 172.641 s |
| Direct vision, medium | Passed; 2.734 s |
| Gateway vision, medium | HTTP 200; 14.863 s; 16,567 input / 280 output |
| Gateway chat, Responses, default MAX, tool smoke | HTTP 200 / passed |
| Minimum MemAvailable by rank | 8.843 / 11.926 / 10.840 decimal GB |
| Restart / swap condition | No rank restart; no swap increase over 256 MiB |

[Capacity receipts, including the failed first path attempt](../results/20261013-tensorfold/evidence/capacity-10m). The first attempt failed on a local generated-kit path before new model loading and rolled back to 3.3M; the corrected preflight passed. Minimum memory values can occur at different times; their 31.61 GB sum is a conservative reserve bound for this test, not a promise for every workload. Neither the 60.875 s nor the C8 wall ratio is a pure prefill metric. Eight 401K requests, cluster power-cycle recovery and indefinite cache fill were not qualified.

## Historical X11c comparison and poster values

[All historical host64k measurements](CAMPAIGN-20261012.md), [earlier X/A/B history](HISTORY.md), and [the superseded poster's source values](../results/20261013-tensorfold/poster-values.json). The historical X11c B0 baseline is from an earlier three-repeat campaign. Percentages compare that baseline with final V7 one-wave decode/32K and explicitly V3 256K/64K; they are descriptive cross-campaign differences, not confidence intervals or a fresh controlled paired test. **The old poster is superseded:** its short reasoning-only code/prose rates could be read as completed-answer performance. The [corrected poster values](../results/20261013-tensorfold/poster-values-corrected.json) use fresh C4 for completed Hermes work and cold prefill; the old data are retained as history.

| Metric | X11c historical | TF351 stage | Change |
|---|---:|---:|---:|
| Hermes cold 64K | 1,318.0 | 2,111.3 V3, derived actual uncached input/native time | +60.2% |
| Owen cold 32K | 1,254.1 | 2,147.6 final V7 | +71.2% |
| Owen cold 256K | 1,230.2 | 1,976.5 V3 | +60.7% |
| Short reasoning decode, code prompt C1 / C8 | 59.8 / 192.9 | 84.47 / 396.03 final V7 | +41.3% / +105.3% |
| Short reasoning decode, prose prompt C1 / C8 | 40.9 / 153.3 | 83.51 / 394.95 final V7 | +104.2% / +157.6% |

## Initial full comparison, before cache repairs

Source `178f28aee9d403ad9b7a52e7b15d999227a7eb93`, initial campaign `94579cbe19c0`. This is a historical failure/diagnostic phase, not the selected recipe. Its [complete numerical receipts](../results/20261013-tensorfold/evidence/initial-v22) and [original frozen summary](../results/20261013-tensorfold/evidence/initial-v22/FINAL-SUMMARY-v22.json) are preserved alongside later repaired phases.

| Initial measurement | C4 | TF351 | TF29 |
|---|---:|---:|---:|
| Owen code C1 / C2 / C4 / C8 | 53.51 / 85.18 / 128.43 / 208.62 | 83.55 / 149.43 / 216.90 / 297.04 | 93.28 / 166.76 / 253.91 / 310.91 |
| Owen prose C1 / C2 / C4 / C8 | 37.42 / 64.63 / 99.80 / 152.51 | 82.30 / 147.28 / 219.27 / 299.36 | 85.22 / 150.46 / 208.54 / 308.40 |
| Owen cold 32K / 256K | 1,610.05 / 1,680.63 | 2,145.69 / 1,978.39 | 2,162.50 / 1,991.50 |
| Hermes C1 / C3 complete workflow wall | 22.97 / 39.00 s | 16.92 / 40.12 s | 19.34 / 40.37 s |
| Initial quality: greedy / seed2026 / seed1708 | 18/26 + 4/6 + 4/6 | 16/26 + 4/6 + 4/6 | 20/26 + 5/6 + 5/6 |
| Initial strict total | 26/38 | 24/38 | 30/38 |
| Actual code sandbox | 5/6 | 5/6 | 5/6 |
| Long-context facts, greedy | 4/4 | 4/4 | 4/4 |

The original quality suite has **26 greedy cases and two six-case sampling subsets, 38 attempts per arm**, with six translation tasks, four digests, four tool tasks, four code tasks, four Czech instruction tasks and four long-context digests. Its frozen fixture and code are in `tensorfold/benchmark-kit/experiments/hybrid/`; it is a separate, longer suite from Q20. Final original comparator states were **TF351 REGRESSION** and **TF29 INCOMPLETE** for all three seeds. INCOMPLETE here is an unproved strict quality gate, not missing generations. The completed anonymous reviews and sandbox grades remain in the numerical receipts. These failures are not regraded into passes.

Several checks were overspecified: exact `C,A,B` rejected semantically correct spacing, tool `locale=cs-CZ` rejected schema-valid `cs`, and JSON fences failed the direct parser. In all 18 initial tool attempts, the runner stopped after that locale mismatch and never tested final save confirmation. Diagnosing that fixture issue does not prove the whole tool task passed. Q20 was separately frozen before new generation, with explicit types/format/completion rules and public IFEval/EvalPlus checks; its scores must not replace the original suite's scores.

Initial native cache qualification failed: legacy hot Code C1 reused 16,384/18,365 tokens (89.21%), below the frozen 90% floor, so the remaining hot legacy cells were not executed. C8 hot reused roughly 75% in five slots and zero in three. Native Hermes `complete=false` and its shorter suite runtime were **not a valid complete-suite speedup**. Later V3/V7 receipts show the repaired phases separately. Early row48 exact-probe failures and a head16/follower48 ring mismatch are historical technical failures; the selected profile consistently uses env/CLI16. The mismatch, not an assumed fork problem, explained the observed HTTP/RDMA protection errors.

## Complete evidence and reproducibility boundaries

The export includes **8,265 numerical receipts** across the initial full comparison, retained cache iterations, full V3, V7, V7r1, V7r2 and 10M qualification. The [manifest](../results/20261013-tensorfold/evidence-manifest.json) binds each original SHA, public redacted SHA and byte length. Original request/response bodies, exact private token dumps, operator addresses/paths, environment and private content are excluded. Timings, usage, counter coverage, seeds/attempt IDs, grades, statuses and failures remain. Public Owen/Q20 fixtures and independent source pins are included for replay.

Small-cell speed variability, different output lengths, historical corpus privacy, independent repeated source phases and incomplete quality cases limit inference. Reports distinguish full V3 matrices, final subsets and separate capacity tests. Publication checks source/package integrity and portable CPU clients; it does not rerun a model or assert a new GPU qualification of a rebuilt public image.

The exact public source archive was extracted on Linux and cross-built with Zig 0.17.0 for `aarch64-linux-gnu`, steps `native dsv41-tools`. Head, follower and check-tool binaries were produced successfully. [Publication build receipt](../tensorfold/publication-checks.json) records their sizes/hashes; build bytes can depend on environment and do not certify destination GPU behavior. The nine new CPU tests also verify coordinated pool/MAX settings, refuse incomplete topology, validate every source/archive/result hash and check the frozen quality fixture and token accounting.
