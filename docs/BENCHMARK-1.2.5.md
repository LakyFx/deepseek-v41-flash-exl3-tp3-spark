# TensorFold 1.2.5 — workflow and RigMark

The measured build identifies itself as 1.2.4k2a. Public 1.2.5 names that same native payload and production settings. The newer [SparkDash pass](SPARKDASH-20261023.md) uses its own protocol; do not mix these rates.

## Unchanged C1–C12 agent workflow

78/78 tasks correct, 355 HTTP requests, 37 repaired tool errors, zero final task failures. Sum of measured wave times **1,258.809 s**, workflow wall **1,261.243 s**. Same frozen 78 initial request-body hashes, evaluator and sampler as the existing campaign. Historical October 22 runs preceded the direct-output formatter, so full rendered-input parity is not implied by identical initial JSON bodies.

| C | Native request decode median tok/s | Native request decode weighted tok/s | Correct | Repaired tool errors | Wave s |
|---|---:|---:|---:|---:|---:|
| 1 | 91.173 | 91.603 | 1/1 | 1 | 34.259 |
| 2 | 68.932 | 62.726 | 2/2 | 0 | 32.342 |
| 3 | 49.810 | 51.017 | 3/3 | 3 | 66.917 |
| 4 | 39.159 | 38.178 | 4/4 | 1 | 66.860 |
| 5 | 31.145 | 33.554 | 5/5 | 1 | 80.248 |
| 6 | 31.022 | 31.739 | 6/6 | 6 | 116.869 |
| 7 | 23.795 | 24.196 | 7/7 | 0 | 70.513 |
| 8 | 22.326 | 23.457 | 8/8 | 1 | 118.342 |
| 9 | 20.383 | 23.588 | 9/9 | 8 | 173.511 |
| 10 | 20.337 | 22.890 | 10/10 | 6 | 176.612 |
| 11 | 17.193 | 17.520 | 11/11 | 5 | 150.018 |
| 12 | 16.769 | 17.406 | 12/12 | 5 | 172.316 |

Raw numeric metrics: [WORKFLOW.json](../results/20261023-1.2.5/WORKFLOW.json). The existing public [workflow fixtures, runner and evaluator](../benchmarks/workflow/) reproduce the standard workload. Native callback `prefilled` occurs after initial token/draft preparation; its following microsecond token-callback interval is not an isolated GPU prefill-end-to-first-token measurement. TTFT includes prefill; a GPU-only boundary was not separately instrumented.

## RigMark protocol 1.3 — separate suite

Pinned upstream [`770c3106729d5b73543afa2d9b498cebda6456f3`](https://github.com/alexellis/rigmark/tree/770c3106729d5b73543afa2d9b498cebda6456f3). Five code/prose/JSON samples, three cold/replay pairs at 8K/32K/64K, three C1/C2/C4 rounds. T=0, top_p=1, seed=20260905, MAX reasoning. Standard decode/prefill/concurrency caps: 8192/8/256 tokens. Wall: **779.263 s**.

| Workload | Median decode tok/s | Median TTFT s | Output gate |
|---|---:|---:|---|
| code | 105.345 | 0.362 | 0/5 |
| prose | 83.324 | 0.355 | 5/5 |
| structured | 139.291 | 0.318 | 5/5 |

RigMark basic output gates: 10/15. The five code samples used the standard 8192-token budget for reasoning without a visible final answer. Their decode timing remains measured, but it is not completed-code throughput or a recipe-quality score. Prose and structured gates passed. The separate agent workflow completed all 78 tasks.

| Input tokens | Cold effective prefill tok/s | Cold TTFT s | Warm replay TTFT ms |
|---|---:|---:|---:|
| 32768 | 2339.592 | 14.006 | 45 |
| 65536 | 2315.700 | 28.301 | 46 |
| 8192 | 2228.473 | 3.676 | 42 |

All nine immediate replays report the full prompt cached; cold runs report zero cached tokens. Warm replay is prefix-reuse latency, not fresh GPU prefill throughput. Cold rate is all prompt tokens / request-to-first-token time, including overhead. This differs from SparkDash’s calibrated fallback.

[Full measured RigMark receipt](../results/20261023-1.2.5/rigmark.json) · [upstream ASCII card](../results/20261023-1.2.5/rigmark.card.txt). Descriptive metadata was corrected to FP4 compressed latent/index codes, BF16 window rings and FP32 staging; measured sections and output gates are retained. Only local platform/appliance paths were omitted from the public receipt, with the original receipt SHA recorded.
