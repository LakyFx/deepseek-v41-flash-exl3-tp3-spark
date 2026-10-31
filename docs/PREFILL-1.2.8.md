# 1.2.8: measured TP3 cold-prefill improvements

**1.2.8 is the exact AF-4K source/profile selected at the end of the 30 October
2026 TP3 prefill campaign.** The version rename runs no new benchmark and changes
no model weights. It retains the smart KV and speculative controllers from 1.2.7.

| Input tokens | 1.2.7 tok/s | **1.2.8 / AF tok/s** | Gain |
| --- | ---: | ---: | ---: |
| 1K | 1,511.1 | **1,826.5** | **+20.9%** |
| 2K | 1,816.6 | **2,366.6** | **+30.3%** |
| 4K | 2,027.4 | **2,801.3** | **+38.2%** |
| 8K | 2,207.2 | **3,169.1** | **+43.6%** |
| 32K | 2,330.6 | **3,525.2** | **+51.3%** |
| 64K | 2,310.9 | **3,552.8** | **+53.7%** |
| 128K | 2,215.8 | **3,499.2** | **+57.9%** |

All values are **medians of three C1 cold-KV samples**, on the same frozen
tokenized inputs and EXL3 checkpoint. K means 1,024 tokens. Every counted input
token was actually computed: cached tokens were **zero**. These are native
prefill rates, not cached-input throughput, decode TPS or monitor peaks.

## Protocol and sampling

| Item | Recorded setting |
| --- | --- |
| Hardware | Three NVIDIA DGX Spark / GB10 nodes, TP3; switchless RoCE; automatic four network-peer channels |
| Model | DeepSeek-V4.1-Flash-1017, source revision `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Quantization | LakyFx EXL3 hybrid, ~3.510072157 bpw routed experts; weight revision `8621801c1fc74aece1fa83e67c6ade3fde1e7894` |
| Capacity | C12 configured slots, adaptive shared R48 ceiling, 7M physical KV, 3 GiB checkpoints, 401,408-token context |
| Benchmark concurrency | **C1**, sequential requests; three samples per size |
| Inputs | Deterministic synthetic English filler; frozen rendered token IDs reused across arms |
| Sampler | Thinking OFF, temperature 0, top_p 1, seed 42, max_tokens 8; greedy output |
| KV isolation | Independent cache namespace per case; cached_tokens=0 checked against native counters |
| Compute chunks | 1.2.7: 2,048; AF/1.2.8: 4,096; physical KV allocation quantum remains 2,048 |
| Primary rate | Computed prompt tokens / native prefill seconds |
| TTFT | Measured separately at the client; includes preparation and prefill |

Cold KV does not mean dropping Linux page cache, forcing a disk-cold load, or
restarting before each sample. The model is already loaded. This test isolates
computed prefill; it does not test hourly retention or guarantee a gain under
every mixed workload. Serving still defaults to MAX reasoning, as before.
No new logits/sampling parameters were introduced for this release.

| Input | Median native prefill | Median client TTFT | Min–max native tok/s |
| --- | ---: | ---: | ---: |
| 1K | 0.561 s | 0.564 s | 1,790.9–1,900.0 |
| 2K | 0.865 s | 0.870 s | 2,362.4–2,441.4 |
| 4K | 1.462 s | 1.468 s | 2,798.7–2,813.4 |
| 8K | 2.585 s | 2.592 s | 3,158.3–3,178.2 |
| 32K | 9.295 s | 9.335 s | 3,522.1–3,525.4 |
| 64K | 18.446 s | 18.511 s | 3,551.5–3,556.8 |
| 128K | 37.457 s | 37.571 s | 3,498.0–3,499.5 |

TTFT and native prefill seconds are independent medians. Their difference is not
a precise post-prefill latency measurement. The short outputs do not constitute
a decode-throughput benchmark. The earlier SparkDash 1.2.6 and heavy-workflow
results remain published under their original version labels.

## What changed

The accepted cumulative path was **1.2.7 → D → E → F → K → R → U → W → AC → AE → AF**.
These letters belong to the October 30 prefill campaign and are unrelated to the
older A–G speculative-controller campaign.

| Layer of work | Concrete change |
| --- | --- |
| D/E/F: distribute actual work | Keep row ownership through residual/mHC/norm/Engram gates; partition index projections, RoPE, packing and selection; exchange completed results. On 17/20 encoder layers, exchange 1,792 finished QR/KV columns rather than 5,120 input columns. Ordered FP32 accumulation is retained. |
| K: attention tiles | Use masked HB16 tiles for TP3's 24 local prompt heads; check every output BF16 value against the original. |
| R/U/W: EXL3 prompt MoE | Adapt upstream transposed MMA and live-16 groups, use 128 output columns/eight warps, and a selective down-projection occupancy hint. Preserve FP32 sum order, original decode and fallback paths. |
| AC: 4K compute | Raise actual compute chunk to 4,096; full-chunk row ownership is 1,376 / 1,376 / 1,344. Keep 2K physical KV quantum and the existing pool/growth policy. |
| AE: Engram scheduling | Predict exact replay-tail boundaries and avoid duplicate inherited second-layer prefetch. Retain the existing second worker, error handling and cancellation joins. |
| AF: prompt scratch | Size prompt-only workspace to 403,456 positions, covering the full request window plus verification/alignment margin. Physical KV/RoPE remains 7M. Add startup rank agreement and early bounds checks. |

This is distributed row work and reduced redundant work inside the existing TP3
engine. It does not split three independent requests across separate models.
Weights, expert partitioning, target verifier/sampling, adaptive K1–K5/R48
selection, X4 smart KV retention, bounded 320 MiB relocation, NVMe companion and
the existing vision policy are retained.

## Why AF instead of AE?

AE's long-input medians were 3,538.8 / 3,570.7 / 3,520.1 tok/s at 32K / 64K /
128K. AF was about **0.4–0.6% lower**, while native startup model accounting on
rank 0 fell from **103.60 to 100.20 GiB**. AF was the selected final tradeoff:
nearly the same throughput, materially more headroom, and a full-window check.
We do not describe AF as the highest observed value in every cell.

The candidate mask shrank from 3,584,000,000 to 206,569,472 bytes per rank.
Minimum sampled MemAvailable under AF was 8.49 / 11.04 / 8.38 GiB. Sampled model
swap and cgroup swap were zero; host-wide swapout deltas were 2 / 0 / 2 pages.
Those host counters do not identify ownership and are not a global zero-swap claim.

## Correctness and source identity

- 21/21 cold samples and 3/3 exact-answer probes completed.
- All-rank exchange, row activation, masked BF16 attention and MoE FP32 equality checks passed.
- 8/8 row and 162/162 native/planner/cache CPU checks; 1,052 AOT launch cases, zero missing; ARM build passed.
- One separate cold **401,344-token** prompt returned the expected answer and the engine returned healthy idle. This capacity probe is excluded from the 1K–128K table.
- AF source manifest: `8083d0ebd68192204ca2da1c478ba6cabcaf930f8a93406091acb725cb1a04d3`.

The full AF heavy-workflow/SparkDash decode-concurrency campaign was **not run**
before the hardware work moved to TP4. This release publishes its completed
TP3 prefill qualification, not TP4 results or a new decode claim. A local rebuild
still needs local serving qualification.

## Evidence and reproduction

[Install the exact recipe](DS41F-1.2.8.md) ·
[all 21 numerical samples](../results/20261030-1.2.8/SAMPLES.json) ·
[summary and per-rank proof](../results/20261030-1.2.8/PREFILL.json) ·
[full campaign median archive](../results/20261030-1.2.8/CAMPAIGN.json) ·
[qualification](../results/20261030-1.2.8/QUALIFICATION.json).

For a paired local reproduction, use the exact same pre-tokenized synthetic
fixtures on both builds, fresh cache namespaces, no competing traffic, the
sampler above and three samples at each exact token count. Reject nonzero KV
hits and cross-check native input/computed/prefill-duration counters. Never
substitute prompt character counts, request wall time or cached token rates
for the native metric. Private agent prompts and generated content are excluded
from the public export; input body/token hashes identify the historical samples.

The engine and source kernels are upstream work with local TP3 adaptations.
[Sources and credits](SOURCES.md#128-tp3-prefill-work) identify the pinned inputs.
