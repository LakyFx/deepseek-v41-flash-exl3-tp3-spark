# Comparable results from the saved TensorFold benchmarks

This is an interpretation and numerical audit of **existing** results. No model requests, benchmark reruns, weight loads or production changes were made. Raw measurements, unsuccessful variants, earlier quality failures and source hashes remain intact.

## What changed

Our Owen adapter sent different code/prose prompts but forced MAX reasoning and a 256-token output cap. **Every measured V3 decode request (90 per arm, 270 across C4/TF351/TF29) and every final V7 request (18 per native arm, 36 total) ended at that cap with reasoning and zero final-content characters.** The timing matches the streamed-token window and native generation counters. These are short reasoning decode measurements, not completed coding/prose rates.

The previous poster emphasized those figures without making the absence of a final answer clear enough. It is superseded by a poster based on completed Hermes work and verified cold prefill. We keep the original numbers labelled by their actual scope instead of inventing replacement decode rates.

The main reference is now **the fresh C4 / DS41F 1.0.0 capture from the TensorFold comparison**. Historical X11c is still available, but its earlier campaign is not substituted for C4. The larger +60.2% historical 64K difference and the +23.4% fresh-C4 difference describe different baselines.

## Completed Hermes work, including C3

C1 is the frozen code-event-dedup workflow. C3 starts that code task, the tool-usage-accounting task and the analysis-deployment-evidence task together. They include real model tool calls, fixed fixture tool results and scored completion. All four final workflows pass in each arm. The final TF351/TF29 stages use the repaired V7 history; C4 is the frozen reference. Each is one pass, not a confidence interval.

| Saved metric | C4 reference | TF351 final | TF29 final |
|---|---:|---:|---:|
| C1 workflow wall, seconds | 22.965 | 15.207 | 22.284 |
| C3 workflow cohort wall, seconds | 39.003 | 35.601 | 43.666 |
| C1 weighted request decode, tok/s | 62.319 | 85.001 | 88.900 |
| C3 weighted request decode, tok/s | 31.826 | 34.491 | 41.809 |
| C1 generated tokens | 1,003 | 877 | 1,559 |
| C3 generated tokens | 2,185 | 2,616 | 3,621 |
| C1 newly computed input | 8,667 | 8,352 | 8,352 |
| C3 newly computed input | 25,807 | 25,249 | 25,249 |

Against C4, TF351 uses **33.8% less C1 task time and 8.7% less C3 cohort time**. Its request decode rate rises **36.4%/8.4%**. TF29 uses 3.0% less C1 time but **12.0% more C3 time**, despite higher token rates. Generated histories and output lengths differ; completed tasks, total time and token rates should be read together.

Request decode is `generated tokens / summed per-request engine decode seconds`. It includes reasoning, tool-call tokens and final output. It is neither visible-answer-only speed nor aggregate C3 throughput. The published final workflow wall includes the monitor phase boundary; the separately retained cohort timers differ by under 0.02 s. Both timers are exported in the audit.

**C3 was measured.** It appears in the scored workflows, the original six-cell legacy screen and the 128K-C3 cold/hot regime. Owen's frozen subset has C1/C2/C4/C8, so there is no Owen C3 result to invent. The prepared full16 legacy matrix was not run in these coverage20 captures; only its six selected cells were run. Final V7 reran C1/C3 workflows and selected Owen cells; it did not rerun the full V3 load suite.

## Verified long-input measurements

These Hermes values are recalculated from full V3. Each arm received exactly the same cold token counts, with verified zero cached tokens: **65,534 at 64K-C1**, and **393,162 total across three 128K-C3 requests**. Native prefill time is the sum of that engine's reported request prefill timers. The three-request metric is not total input divided by cohort wall time.

| Cold measurement, tok/s | C4 reference | TF351 V3 | TF29 V3 | TF351 vs C4 |
|---|---:|---:|---:|---:|
| Hermes 64K-C1 computed input/native prefill | 1,710.851 | 2,111.258 | 2,118.820 | +23.4% |
| Hermes 128K-C3 computed input/native prefill | 1,638.884 | 2,040.969 | 2,056.489 | +24.5% |
| Owen nominal 32K actual input/HTTP TTFT; C4 V3, TF final V7 | 1,610.054 | 2,147.643 | 2,161.993 | +33.4% |
| Owen nominal 256K actual input/HTTP TTFT; V3 | 1,680.628 | 1,976.480 | 1,989.405 | +17.6% |

The Owen prefill definitions remain independent of the 256-token decode limitation: these cold cells request **one output token**, and measure input/first-token latency. C4 and V3 have two cold repetitions; final V7 has one 32K measurement. Do not replace HTTP TTFT with engine prefill time or merge them into one rate.

## C8 and cache

The retained **Hermes 8K-C8 hot probe** has eight different short fixture bodies, a 256-token cap per request and verified prefix reuse. Its aggregate output / cohort wall rate is **150.590 / 199.245 / 221.017 tok/s** for C4/TF351/TF29, a TF351 increase of **32.3%**. All are V3, not rerun on final V7. This is a bounded token-throughput probe, not eight completed scored workflows.

The separate Owen C8 figures **396.030 / 394.949** for final TF351 code/prose prompts are reasoning-only aggregate decode-window rates, approximately 49.5 tok/s per stream. They cannot replace the Hermes C8 wall rate or predict the time to complete eight agent tasks.

Identical-prefix hot cache remains **99.81%/99.92%/98.60%** for C4 and **100%/100%/100%** for both native arms at 64K-C1/128K-C3/8K-C8. This is not the workflow cache percentage; changing generated histories changes that percentage's denominator.

All comparative speed/quality data used a **3.3M pool**. TF351's **10M capacity** was separately stress-qualified. TF29 10M is untested. The 10M badge describes capacity, not a measured 10M decode result.

## Quality remains a separate check

The Q20 screen is **20/24, 20/24 and 21/24** for C4/TF351/TF29, with **5/6** code tasks passed in each arm. This small bounded screen does not establish quality equivalence or prove that 2.9 bpw is better. Incompletes and all earlier strict regressions remain in [the full results](TENSORFOLD-RESULTS.md).

## Audit the receipts without inference

```sh
python3 tools/audit_tensorfold_results.py
```

The script checks the public export hashes, request coverage, workflow correctness, usage/counter agreement, cold input counts and the original Owen timing formula. It writes [comparison-audit.json](../results/20261013-tensorfold/comparison-audit.json) with the input hashes and recalculated rates. It has no networking or inference code.

Original evidence: [V3 reference and full native captures](../results/20261013-tensorfold/evidence/v3/results), [TF351 final C1/C3](../results/20261013-tensorfold/evidence/v7/TF351-workflow_delta_probe/RESULT.json), [TF29 final C1/C3](../results/20261013-tensorfold/evidence/v7r2/TF29-workflow_delta_probe/RESULT.json), [final numerical summary](../results/20261013-tensorfold/summary.json), [inventory and hashes](../results/20261013-tensorfold/evidence-manifest.json), [corrected poster values](../results/20261013-tensorfold/poster-values-corrected.json).
