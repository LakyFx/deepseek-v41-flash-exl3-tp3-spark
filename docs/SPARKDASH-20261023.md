# Stock SparkDash — TensorFold 1.2.5 / TP3

23 October 2026. Fresh measurements, using the original unmodified SparkDash collectors at
[`9e3f01729a4e63dccc2b6b608e9e67d3168891b6`](https://github.com/MiaAI-Lab/sparkDash/tree/9e3f01729a4e63dccc2b6b608e9e67d3168891b6).
This historical pin matches the stock protocol around the supplied 17 October cards; the other operator’s exact installed revision is not independently known.

**3× DGX Spark 128 GB, TP3, DeepSeek-V4.1-Flash-1017, LakyFx EXL3 ~3.51 bpw routed experts, TensorFold 1.2.5 (measured runtime alias 1.2.4k2a).**
**Temperature 0, top_p 1, thinking OFF, 400 tokens per decode stream.** One wave at each selected C, original warmups, original prompts; no cache reset or model reload.

## Decode

Rates are tok/s. Stream and TTFT are arithmetic means, matching the upstream cards. TTFT includes prefill and waiting.

| C | Prose aggregate | Prose stream | Prose TTFT ms | Code aggregate | Code stream | Code TTFT ms |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 87.71 | 87.71 | 5.40 | 126.06 | 126.06 | 325.60 |
| 2 | 111.79 | 60.93 | 572.83 | 169.36 | 88.39 | 337.15 |
| 3 | 123.04 | 45.27 | 768.15 | 174.00 | 64.20 | 148.13 |
| 4 | 138.33 | 38.38 | 951.57 | 187.34 | 50.73 | 155.33 |
| 5 | 146.40 | 33.50 | 1142.18 | 207.70 | 44.90 | 303.19 |
| 6 | 154.10 | 29.37 | 1343.21 | 218.05 | 38.86 | 195.30 |
| 8 | 164.43 | 23.92 | 1731.69 | 226.19 | 30.94 | 371.57 |
| 10 | 169.73 | 20.00 | 2081.71 | 241.80 | 26.44 | 796.35 |
| 12 | 183.57 | 18.00 | 2323.38 | 246.21 | 23.12 | 574.38 |

## Prefill

| Target | Actual tokens | Prefill tok/s | TTFT ms | Cached | Samples | Low confidence |
|---|---:|---:|---:|---:|---:|---|
| 1K | 1042 | 1922.05 | 809.70 | 0 | 3/3 | Yes |
| 2K | 2065 | 2224.18 | 1196.00 | 0 | 3/3 | No |
| 4K | 4117 | 2097.48 | 2230.40 | 0 | 3/3 | No |
| 8K | 8212 | 2228.93 | 3951.85 | 0 | 3/3 | No |

Only 1K–8K was measured in this SparkDash pass; 16K–64K has no SparkDash result here. Cache hits are excluded, not counted as fresh prefill. Warm-prefix results from RigMark are separate.

## Metric definitions

- Stream decode = `(completion_tokens - 1) / (last_token_time - first_token_time)`.
- Aggregate = `sum(completion_tokens_i - 1) / (latest_last_token_time - earliest_first_token_time)`.
- Card Stream = mean of individual stream rates; it is not aggregate divided by C.
- Card TTFT = request to first token, including prefill/queue, averaged over streams.
- Prefill uses server prompt timing if available. Here all rows use the fallback `(prompt_tokens - cached_tokens) / max(TTFT - calibrated_overhead, TTFT * 0.5)` in seconds. Calibration was 267.57 ms; correction is capped at 50%.
- Prefill retains the median-rate sample with that sample’s token count and TTFT; the other columns are not independent medians. The stock low-confidence flag remains on 1K.
- C1 prose reused the warmup prompt and reached 5.40 ms TTFT. This is a warm-prefix observation, not a cold-start guarantee.

## Results and reproduction

102/102 decode streams returned 400 tokens; 12/12 prefill samples passed. Native logs record **121 thinking=False, zero thinking=True** requests including warmup/calibration. Total campaign: **263.186 s**. All three inference executables and gateway generations stayed unchanged; public inference was closed during the run and restored afterwards.

The supplied comparison cards use TP4, while this recipe uses TP3. Their exact quantization, clocks and serving settings are not independently verified; matching a benchmark protocol does not make these a controlled same-hardware software comparison.

[Portable benchmark runner, original sources and source hashes](../benchmarks/sparkdash/) · [raw jobs and native validation summary](../results/20261023-sparkdash/).

Original SparkDash exports: [prose](../assets/1.2.5/sparkdash-original-prose.png), [code](../assets/1.2.5/sparkdash-original-code.png), [prefill](../assets/1.2.5/sparkdash-original-prefill.png).
Portrait artwork: [prose](../assets/1.2.5/01-prose.png), [code](../assets/1.2.5/02-code.png), [prefill](../assets/1.2.5/03-prefill.png).
