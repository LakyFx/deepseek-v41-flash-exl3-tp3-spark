# Short diverse code/prose decode screen

This is the portable replay of the **31 October 2026 TP4 P2 short screen**,
whose selected compute path is used by **4.0.0**. It is separate from both
stock SparkDash and the frozen agent/tool workflow. No GPU benchmark was
rerun when these historical results were published.

The exact authored inputs cover twelve software topics, code and prose,
at **C1 / C3 / C6 / C12**. Each request generates **512 tokens**, greedily,
temperature 0, top_p 1, seed 42, thinking OFF. Two passes run consecutively
without resetting the model, cache or controller; the headline table uses
**pass 2**. Warmup includes KV, CUDA graphs, Engram and the controller together.
It does not isolate controller learning or evaluate full task correctness.

The measured profile was TP4, 8K compute / 2K physical KV quantum,
**C12 / adaptive R48 / 7M KV / 3 GiB checkpoints**. Current 4.0.0 capacity is
C16 / 9M / 3.5 GiB with the same selected native binaries/compute path.
A replay at that newer capacity is a separate measurement.

## Replay, only in an exclusive backend window

Python 3, standard library only. The operator must close other ingress and
expose the native backend on loopback before starting. The runner checks
native health/idle before every cell. It does not stop services, drain work,
reset caches, restart a model or reopen API ingress. `--exclusive` confirms
the operator has established that window; it cannot prove no other client
will arrive later. No automatic retries are made.

```bash
# Optional authentication is read from DS41F_API_KEY, never a CLI argument.
python3 benchmarks/short_decode/run.py \
  --base-url http://127.0.0.1:8003 \
  --exclusive --output local-results/tp4-short-decode-new
```

The output directory must be new. Full replies remain local: do not publish
them by copying the output directory wholesale. The public historical export
contains allowlisted numeric counters and body/content hashes only. It includes
both passes and all **88 requests / 45,056 generated tokens**.

## Metrics

- **Native decode sum**: sum of `tensorfold.tokens_per_second` over requests
  in a cell. Excludes prefill. It is not a time-aligned instantaneous peak
  or the whole-wave wall-clock throughput.
- **Per-request median**: median of those native rates, without multiplying
  the median by concurrency.
- **Whole-wave throughput**: total completion tokens / elapsed time from
  starting the concurrent batch until all replies complete, including prefill
  and client/server preparation. Both metrics are published separately.
- **Speculative acceptance**: total accepted / total drafted tokens, weighted
  by token count. No unweighted average of percentages.

[Human-readable measured results](../../docs/SHORT-DECODE-4.0.0.md) -
[numeric summary](../../results/20261031-4.0.0/SHORT-DECODE.json) -
[88 request counter records](../../results/20261031-4.0.0/SHORT-DECODE-REQUESTS.json).

Recalculate the published numbers without accessing a backend:

```bash
python3 tools/audit_short_decode.py
```
