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

The serving policy after these tests remains main max and vision medium. The benchmark medium mode did not change the main agent's production reasoning.

## Limits of the evidence

Measured results show workload-specific behavior. They do not establish confidence intervals, universal 20 percent speedups, model quality equivalence across reasoning modes, or the capacity to serve eight near-limit contexts simultaneously. User observations during real Hermes work explain the deployment preference, while synthetic measurements provide the reproducible comparison. Both are useful and should remain distinguishable.
