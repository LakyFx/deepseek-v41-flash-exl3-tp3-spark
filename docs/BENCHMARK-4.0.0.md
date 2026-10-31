# 4.0.0 / TP4 P3 - measured results

4.0.0 is the release label for the exact production P3 profile, retaining the
selected P2 compute path. Publication did not run a new benchmark or restart
production. [Recipe and installation](DS41F-4.0.0.md).

## Frozen tool workflow - final P3 settings

Four Sparks, switchless RoCE; pinned DeepSeek 1017 EXL3 mixed weights;
C16 / shared adaptive R48 / **9M physical KV / 3.5 GiB checkpoints**;
8K compute chunks, 401,408 context. Temperature **0.6**, seed **42**, MAX
thinking ON, max output **65,536**. Same frozen 78 initial request bodies,
tools and evaluator as the retained workflow campaign. Actual generated
reasoning/tool turns can differ. Benchmark waves remain **C1-C12**; C16 is
separately smoke-tested.

**78/78 correct final tasks - 327 requests - 145,668 completion tokens -
908.31 seconds summed wave time - 12 recoverable tool errors - 0 inference
retries.** Recoverable tool errors remain counted; they are not final failures.

| Concurrent workflows | Native decode median / request, tok/s | Aggregate completed-workflow tok/s | Wave time, s | Correct | Tool errors | Post-prefill first-token upper bound, ms median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 105.33 | 96.45 | 48.43 | 1/1 | 4 | 3.56 |
| C2 | 70.83 | 117.80 | 35.70 | 2/2 | 1 | 3.54 |
| C3 | 51.63 | 130.93 | 45.24 | 3/3 | 0 | 2.74 |
| C4 | 40.39 | 141.31 | 48.84 | 4/4 | 0 | 4.23 |
| C5 | 35.91 | 150.40 | 61.51 | 5/5 | 0 | 3.84 |
| C6 | 31.08 | 156.34 | 58.37 | 6/6 | 0 | 3.65 |
| C7 | 26.53 | 152.75 | 85.36 | 7/7 | 0 | 3.85 |
| C8 | 24.70 | 169.83 | 84.67 | 8/8 | 0 | 3.69 |
| C9 | 22.94 | 175.73 | 81.97 | 9/9 | 0 | 3.47 |
| C10 | 21.53 | 167.11 | 105.77 | 10/10 | 2 | 3.77 |
| C11 | 19.79 | 186.03 | 109.67 | 11/11 | 0 | 3.49 |
| C12 | 19.62 | 179.92 | 142.79 | 12/12 | 5 | 4.03 |

**Definitions:** native decode is an individual request's completion-token
rate during native decode, including reasoning. Workflow aggregate is all
completion tokens divided by complete wave wall time, including prefill,
queuing, tools and repairs. It is neither the sum of per-request medians nor
an instantaneous monitor peak. First-token values above exclude prefill;
they retain the existing host-event/preparation upper-bound definition,
not isolated GPU kernel time. Total TTFT remains in the JSON.

[All numeric cells, timings, output lengths and rank memory](../results/20261031-4.0.0/WORKFLOW.json).
The existing [frozen workflow runner](../benchmarks/workflow/) is retained.
This is an agent/tool workload, separate from stock SparkDash thinking-OFF
prose/code tests. No new SparkDash score is claimed for 4.0.0.

## Historical P2 cold prefill - same selected compute path

These 21 samples were measured before the final P3 capacity change, using
**C12 / 7M / 3 GiB**, the selected P2 source and 8K compute chunks. P3 retains
that compute path but has C16/9M/3.5 GiB; cold prefill was not rerun for P3.
The final decode AOT repair adds missing launch coverage and is bound in the
production parity receipt. Do not present these as a fresh P3 cold campaign.

C1, three samples per size; thinking OFF, temperature 0, top_p 1, seed 42,
max output 8; zero cached input tokens. Throughput is **computed prompt
tokens / native prefill duration**, excluding queue, HTTP and decode. Exact
same frozen cold fixtures and aggregation as the TP3 AF campaign.

| Prompt tokens | Samples | Cached tokens | Median computed prefill, tok/s | Median prefill, s | Min-max tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 3 | 0 | **2,180.9** | 0.470 | 2,136.8-2,232.1 |
| 2,048 | 3 | 0 | **2,799.3** | 0.732 | 2,791.9-2,814.2 |
| 4,096 | 3 | 0 | **3,259.4** | 1.257 | 3,241.3-3,273.7 |
| 8,192 | 3 | 0 | **3,640.8** | 2.250 | 3,629.0-3,652.3 |
| 32,768 | 3 | 0 | **4,205.3** | 7.792 | 3,948.4-4,212.4 |
| 65,536 | 3 | 0 | **4,289.3** | 15.279 | 4,287.8-4,292.0 |
| 131,072 | 3 | 0 | **4,254.6** | 30.807 | 4,250.9-4,256.0 |

[Summary](../results/20261031-4.0.0/P2-COLD-PREFILL.json) -
[21 allowlisted sample records](../results/20261031-4.0.0/P2-COLD-SAMPLES.json).
Sample records include hashes and numerical measurements; prompts, answers,
private session/job IDs, host addresses and credentials are not exported.

## Memory and operational checks

The same P3 heavy run sampled all four ranks 305 times each:

| Rank | Minimum MemAvailable, GiB | Model swap, KiB | Cgroup swap, B | Max memory PSI full avg10 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 8.054 | 0 | 0 | 0.00 |
| 1 | 9.764 | 0 | 0 | 0.00 |
| 2 | 17.166 | 0 | 0 | 0.00 |
| 3 | 14.233 | 0 | 0 | 0.00 |

Separate operational checks passed **16 simultaneous requests**, an additional
text answer, and a MAX/8K image request identifying a red square and blue
circle. The image returned a complete answer in 1.066 seconds with 68 output
tokens, including 52 reasoning tokens. This is a simple functional smoke,
not a visual-quality benchmark.

Across the longer qualification window the minimum available memory per rank
was 6.343 / 8.448 / 15.462 / 13.738 GiB, with model swap zero. The operating
systems recorded 13 / 5 / 1 / 3 global swap-out pages; global OS swap is not
identical to model/cgroup swapping. The configured 9M pool is not a claim
that this workflow fills every position.

[Version/source/binary/kernel parity](../results/20261031-4.0.0/PRODUCTION-PARITY.json) -
[TP3 1.2.8 prefill report](PREFILL-1.2.8.md) -
[preserved historical SparkDash 1.2.6](SPARKDASH-20261024-1.2.6.md).
