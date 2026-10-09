# Benchmark methodology and interpretation

The published results preserve separate measurement campaigns. Do not combine their numbers into one paired experiment. Tests keep the EXL3 checkpoint and TP3 architecture, but output length, accepted draft tokens and cache state affect observed throughput.

## Metrics

- **Request decode tok/s:** generated tokens divided by measured decode time, weighted across requests. Cold concurrent requests may have interleaved prefill stalls during this interval.
- **Aggregate output tok/s:** total cohort output divided by cohort wall time, including time to first token. Reasoning tokens count as generated tokens.
- **Computed prefill tok/s:** tokens actually computed divided by measured prefill time. A hot prefix is not a fresh prefill benchmark.
- **Draft acceptance:** accepted draft tokens divided by proposed draft tokens for the specified measurement scope. A shorter maximum draft changes the denominator and cannot be interpreted as a direct quality improvement.
- **Workflow wall time and correctness:** duration of a scored multi-turn agent task, including its tool rounds. Faster decode and faster task completion need not move in the same direction.

Concurrency C1 means one submitted request or workflow; C3 means three. Which interpretation applies is recorded in each phase. C8 short-context aggregate throughput and C1/C3 agent request decode are not directly comparable metrics.

## Original cold and hot campaign

The October 2 campaign measured A1, A1+B1, B1 and 0.8.8 across code and prose, concurrency C1 to C4, cold and hot: 16 cells per version and 64 measured cells total. B2 and A1+B2 were rejected rather than assigned synthetic performance numbers.

Inputs were fixed at approximately 16K to 24K tokens, output cap 256 tokens, predominantly reasoning. Hot priming was excluded from reported rates but included in the active per-version time budget. Hot cells were accepted only with measured cache reuse. A1 reused valid cold cells on the same model instance after an initial hot-cache verification problem and diagnostic requests. These are throughput tests, not a general quality benchmark. Each cell is one measurement.

See `results/20261002-six-version/` for every measured cell and the B2 failure record. Historical 0.8.9 is based on the A1 implementation; it must not be represented as an independently measured fresh baseline whenever the source is a historical A1 run.

## Expanded workload suite

The X through X11 campaign and the X11 follow-ups used the same coverage20 allocation. The allocation has seven regimes and a hard maximum of 1,200 seconds per variant, with 50 seconds reserved for draining owned requests.

| Phase | Original budget, seconds |
|---|---:|
| Agent workflow C1 | 80 |
| Agent workflow C3 | 130 |
| Legacy screen, six cells | 300 |
| Mixed workload | 140 |
| 64K C1 cold and hot | 95 |
| 128K C3 cold and hot | 325 |
| 8K C8 cold and hot | 80 |

The legacy screen is a subset of the separately prepared full16 corpus. Running coverage20 does not prove every full16 C1 to C4 cell was rerun on every X version. Startup C1 to C8 checks are functional samples, not a substitute for a full performance matrix.

Agent sampling uses temperature 0.6, seed 42, up to 1,536 output tokens per turn and at most five turns. Legacy tests use temperature zero, seed 42 and an output cap of 256. Independently salted cold requests and verified hot prefixes distinguish cache behavior. A phase that exceeds its budget retains partial evidence and an incomplete status; missing measurements are not zero throughput or zero gain.

## Reasoning comparison on X11c

The order was baseline max, none, medium, and repeated max. All four runs used the same loaded X11c model and the same frozen suite. Prefix cache was reset independently before each pass and the original cold phases; cache salts separated profiles. KV was not dumped or imported.

The DeepSeek native encoder accepts `high` at strength 50, while the gateway's public medium setting maps to that native value. The test adapter must put native `high` in both the top-level reasoning field and chat-template kwargs: the top-level field overrides the template setting. None disables both thinking flags. Max uses native `max` at strength 100.

An adapter issue was corrected after the none run was durably saved and before any successful medium response. The coordinator resumed without reloading the model. This was a client mapping repair, not a different inference variant.

All scored agent tasks passed in each mode. All four 128K C3 phases exhausted the 325-second budget, so their hot measurements remain incomplete. None and medium each have one run; max has two. Outputs differed, including 627 versus 965 tokens in the two max C1 cohorts. Medium's higher observed C1 token rate does not establish superior speed or equal quality for arbitrary tasks.

Short throughput probes can exhaust their output cap during reasoning and return no visible final answer. Reasoning tokens remain generated tokens in the reported rates. Those probes do not require a complete visible answer and are distinct from the scored multi-turn workflows, which all passed. A MAX decode rate is therefore not a rate of visible answer tokens.

The serving policy after these tests remains main max and vision medium. The benchmark medium mode did not change the main agent's production reasoning.

## Run a new public comparison

The sixteen expanded workflow and long-context bodies are included. The older legacy screen used private captured inputs and private operational source packets, so the public runner prepares eight clearly labelled synthetic surrogates. It uses the actual backend tokenizer and records their real counts. Their measurements are new results, not an exact replay of our historical legacy cells.

On rank 0, after the model is healthy, prepare the surrogate inputs without sending generation requests:

```bash
python3 benchmarks/run.py --config config/cluster.local.json \
  --prepare-only --output local-results/public-legacy
```

For a measured run, close every other ingress and confirm no client is still sending requests. Set `benchmark_control.enabled` to true before starting the measurement deployment, keep its `peer_address` equal to `127.0.0.1`, and create the configured lease directory on each node before launching. The launcher mounts it read-only in the model container. The host-side runner writes the owned lease into the rank 0 directory. This optional control lane adds only bounded cache reset and owned cancellation operations. It is inactive without the closed-window lease and is not required for ordinary serving.

```bash
# Use this exact directory only if it matches your local configuration.
mkdir -p /srv/dsv41/benchmark-lease
python3 benchmarks/run.py --config config/cluster.local.json --exclusive \
  --legacy local-results/public-legacy/corpus.json \
  --reasoning max --profile coverage20 --output local-results/x11c-max-01
```

Repeat with `--reasoning none` or `medium` and a fresh output directory. The `coverage20` profile retains the seven original budgets and a 20-minute hard cap. `--profile legacy-full16` instead runs the full sixteen code/prose, cold/hot, C1 through C4 cells with a 720-second phase budget. These profiles answer different questions.

Every run retains local metric snapshots and a `RESULT.json`, or a failed-closed receipt and partial evidence. A completed lease is marked inactive and retained. Before another run, inspect that the engine is idle and move the inactive lease to an operator-chosen archival filename; the tool refuses to overwrite any existing lease. After failure, resolve outstanding owned requests before moving its lease or resuming ingress. The runner never stops services, reloads weights or opens the API. Restore your serving policy and resume ingress only after verifying the final selected runtime.

The public adapter binds measurements to a stable frontend process and rejects counter resets or unexplained request counts. It does not claim independent generation UUID checks on every TP worker. An operator must preserve the exclusive window and monitor rank health for the entire run. Keep new local results private until they have been reviewed for content.

## Limits of the evidence

Measured results show workload-specific behavior. They do not establish confidence intervals, universal 20 percent speedups, model quality equivalence across reasoning modes, or the capacity to serve eight near-limit contexts simultaneously. User observations during real Hermes work explain the deployment preference, while synthetic measurements provide the reproducible comparison. Both are useful and should remain distinguishable.
