# Reproduce Hermes, Owen and Q20 on TensorFold

The frozen adapter is in `tensorfold/benchmark-kit/`. The second benchmark is **adapted from [Christopher Owen's GitHub recipe](https://github.com/christopherowen/spark-ds41f)**, `bin/spark` at `dd35633edeadd8864ec177890025b6e7c5df47ce`, source SHA `b130f172ec4e039532e9fb0b7d4d6256ddc5f16918c88d0689841146631ba46a`. Existing [Owen methodology](OWEN-BENCHMARK.md) explains the common-context subset. We retain his stream timing and decode-window definitions, and add backend isolation, native counter coverage and cold-cache receipts.

The separate [real Hermes production workload comparison](HERMES-REAL-WORKLOAD.md) is a passive operational sample, published with [exact observation aggregates](../results/20261014-hermes-real-workload/observations.json) and an [offline audit](../tools/audit_real_workload.py). Its different five-hour windows and active-time rates must not be blended into the frozen benchmark tables below.

## Frozen conditions and metric definitions

Use `TF351-3.3M` or `TF29`, the same **3.3M** kit on every rank, context 401,408, eight streams, MAX and the supplied cache/tool fixes. Run one variant at a time. Loading time is separate from the measurement budget. Do not label a new 10M run as a reproduction of the 3.3M campaign.

| Measurement | Definition |
|---|---|
| Owen decode C1/C2/C4/C8 | Aggregate output / shared first-to-last-token decode window; one discarded warmup and three measured repeats in the full fixture |
| Owen cold 32K / 256K | Actual rendered input tokens / HTTP TTFT; two repeats, unique verified cache salt per repeat |
| Hermes cold prefill | Newly computed tokens / engine prefill seconds; include only requests with verified zero cached tokens |
| Hermes hot request | Prime then exact-prefix replay; retain cached/computed token counts, latency and completion status |
| Workflow C1/C3 | Complete bounded tool cohort wall time and semantic fixture correctness; also record output length and newly computed input |
| Q20 | Twenty frozen cases, 24 attempts including selected sampling repeats; incomplete/empty/length-capped answers count as failures |

MAX chat reasoning is explicit. Some Owen endpoints use `/v1/completions`, which does not have the chat `reasoning_effort` field; retain the endpoint and payload for each cell. Do not call every Owen request a chat-reasoning comparison. C1 is one request; C8 throughput is the cohort total.

**Retained-run audit:** all measured Owen decode requests in full V3 and final V7 exhausted the 256-token budget in reasoning, with no final content. Code/prose label the prompt. These are short decode microbenchmarks, not completed-answer speed or quality measurements. The completed Hermes C1/C3 workflows are separately scored. [Correction and comparable results](BENCHMARK-CORRECTION-20261014.md).

To verify the published numerical correction **without calling a model**, run `python3 tools/audit_tensorfold_results.py` from the repository root. It checks receipt hashes, counter coverage, cold-token counts, completed-workflow correctness and the Owen timing formula, and writes `results/20261013-tensorfold/comparison-audit.json`. It does not connect to an API or change a runtime.

## Close ingress and prepare the native control lane

The native head must be started with `--benchmark-control`; `tensorfold/launch.py --benchmark` also forces a loopback bind. Close all gateways/agent ingress for the exclusive window before measurement. Merely supplying `--exclusive` does not close them. The runner does not deploy, stop services or reopen APIs.

Copy `tensorfold/benchmark.example.json` to untracked `tensorfold/benchmark.local.json`. Set a **new lease directory for each run**. A completed lease remains as evidence; an existing lease is never overwritten. For a C4 reference use `benchmark_engine: "vllm"` and its already documented control middleware instead of the native flag.

The native adapter checks `/benchmark/status`, the process generation, supported cache salts and `/metrics`. Cold/hot replay fails if support is unproved. All owned sockets must drain and the engine must be idle before the next reset; a disconnect alone is not cancellation acknowledgement. Exact rendered token IDs are required from `/tokenize` for a paired quality comparison.

## Hermes public replay, up to 20 minutes per arm

From the repository root, after the exclusive engine is ready:

```sh
python3 tensorfold/benchmark-kit/benchmarks/run.py \
  --config tensorfold/benchmark.local.json \
  --prepare-only --output /srv/ds41f-bench/public-surrogate
python3 tensorfold/benchmark-kit/benchmarks/run.py \
  --config tensorfold/benchmark.local.json \
  --legacy /srv/ds41f-bench/public-surrogate/corpus.json \
  --version-label TF351 --reasoning max --profile coverage20 \
  --exclusive --output /srv/ds41f-bench/tf351-hermes-new
```

Preparation calls `/tokenize` and makes public synthetic code/prose surrogates; it sends no generation requests. The captured private legacy corpus is not published. These surrogate runs exercise the same scheduler/workflow phases but cannot reproduce the original private prompts byte for byte. Public long-context/workflow fixtures, graders, timing logic and corpus hashes are included. Pass `--version-label` explicitly because this standalone kit has no C4 runtime profile.

## Owen: common-context code, prose and prefill

```sh
python3 tensorfold/benchmark-kit/benchmarks/owen/fetch_source.py
python3 tensorfold/benchmark-kit/benchmarks/owen/run.py \
  --config tensorfold/benchmark.local.json --exclusive \
  --output /srv/ds41f-bench/tf351-owen-new
```

The fetcher hashes the pinned client and stores it as data; the replay imports only reviewed benchmark functions, not its administration CLI. The frozen fixture contains C1/C2/C4/C8 code and prose plus nominal 32K/256K cold prefill. The historical final V7 diagnostic replay used only C1/C8 and 32K; its single measured wave is not substituted for the three-repeat full V3 fixture.

Use a fresh output and lease directory, change the installed profile consistently across all ranks, then repeat for TF29 or C4. Save request timing, token counts, warmup, repeats, engine/counter receipts and failed phases. Do not blend Hermes wall throughput with Owen decode-window throughput.

## Q20: six-minute quality capture per arm

The fixture combines six HumanEvalPlus-derived tasks, selected official Google IFEval checks, synthetic tool/fact tasks and long-context checks. Source pins, hashes, Apache notices and the exact case/attempt selection are in `tensorfold/benchmark-kit/quality-q20/`. It is a small regression screen, not the full HumanEvalPlus/IFEval suite or a blind quality leaderboard.

First capture **C4** with a new exclusive reference window to get token IDs, then capture TF351 and TF29 with the same fixture. Set the corresponding engine in each config:

```sh
python3 tensorfold/benchmark-kit/quality-q20/q20_run.py \
  --fixtures tensorfold/benchmark-kit/quality-q20/FIXTURES.json \
  --benchmark-kit tensorfold/benchmark-kit \
  --config tensorfold/benchmark.local.json --arm C4-reference \
  --output /srv/ds41f-bench/q20-c4-new --exclusive --execute
python3 tensorfold/benchmark-kit/quality-q20/q20_run.py \
  --fixtures tensorfold/benchmark-kit/quality-q20/FIXTURES.json \
  --benchmark-kit tensorfold/benchmark-kit \
  --config tensorfold/benchmark.local.json --arm TF351 \
  --token-reference /srv/ds41f-bench/q20-c4-new/TOKENIZATION.json \
  --output /srv/ds41f-bench/q20-tf351-new --exclusive --execute
```

Repeat the second command for `--arm TF29` and a fresh destination. The runner forces matching top-p/top-k/min-p, compares rendered token IDs and caps inference at 360 s per arm. Three arms took **14.80 minutes** in the full V3 quality capture; the final tool deltas brought total inference to about **15.34 minutes**, excluding model loads and CPU review. A worst-case new three-arm run is approximately 18 minutes plus startup/drain/review; do not promise a 20-minute total including three weight loads.

Code grading is a separate CPU-only sandbox step against the locally retained responses. Use an operator-chosen immutable Python container image ID:

```sh
python3 tensorfold/benchmark-kit/quality-q20/q20_sandbox.py \
  --fixtures tensorfold/benchmark-kit/quality-q20/FIXTURES.json \
  --responses /srv/ds41f-bench/q20-tf351-new/RESPONSES.json \
  --image sha256:REPLACE_WITH_VERIFIED_LOCAL_PYTHON_IMAGE_ID \
  --output /srv/ds41f-bench/q20-tf351-code-new --execute
```

The harness runs generated code only in a bounded CPU Docker container: no network/GPU, no host mounts, read-only, unprivileged, resource caps. Containers and receipts remain for explicit cleanup. Noncode graders separate fact correctness from exact formatting; editorial/tool confirmations still need review. The published historical evidence includes numerical grades and failures, not private conversation bodies, token dumps or generated private answers.
