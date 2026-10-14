# Hermes Agent in production: X11c, C4 and TF351

These are **passive observations of real Hermes Agent work**, not a new controlled same-prompt benchmark. They supplement the existing exclusive Hermes workflow, Owen and quality campaigns; they do not replace them. No extra model generations were run to obtain this table.

## Observed aggregate decode

Each version uses a separate **five-hour window, 00:00–05:00 CEST**: X11c on 11 October, C4 / DS41F 1.0.0 on 13 October and TF351 / 10M on 14 October 2026. Parentheses contain the active decode sample duration, in minutes. C1–C8 denote concurrent requests; C4 in the version column is the production version 1.0.0.

| Concurrency | X11c aggregate tok/s (active min) | C4 / 1.0.0 aggregate tok/s (active min) | TF351 aggregate tok/s (active min) | TF351 vs C4 |
|---|---:|---:|---:|---:|
| C1 | 51.7 (221.6) | 50.6 (168.0) | **83.7 (52.9)** | **+65.4%** |
| C2 | 79.9 (29.7) | 69.1 (71.7) | **111.4 (33.5)** | **+61.2%** |
| C3 | 101.3 (9.6) | 103.6 (32.5) | **125.5 (21.5)** | **+21.2%** |
| C4 | 103.6 (5.7) | 100.3 (8.9) | **135.1 (22.2)** | **+34.6%** |
| C5 | 112.1 (3.0) | not observed | 142.0 (24.2) | — |
| C6 | not observed | not observed | 145.7 (12.8) | — |
| C7 | not observed | not observed | 145.6 (8.7) | — |
| C8 | not observed | not observed | **150.6 (75.5)** | — |

All rates are total output across concurrently decoding requests, **not per-stream rates**. Reasoning contributes to generated tokens. A missing historical concurrency cell is not zero throughput. X11c C5 has only three active minutes. No TF29 production window was available, so no real-workload TF29 comparison is invented.

## What the average measures

The mean is weighted by the duration of valid intervals with positive output at the observed concurrency. For the vLLM observations it is positive streaming generation-counter deltas divided by their interval durations. For TensorFold it is the integral of the native `/health` live decode rate divided by the corresponding valid active time. Idle periods, pure prefill and invalid gaps are not assigned zero throughput and averaged into the five-hour decode number.

This is different from the controlled benchmark's summed request decode time and different from end-to-end workflow output/wall throughput. In particular, integrating the TF live rate produces **estimated observed decode work**, not a retrospective exact billing-token count. Old completion-only counters would create spikes; the old observer's TensorFold zero deltas are not used.

Prompts, context lengths, cache warmth, concurrency distribution and task mix differ between windows. The rate ratio describes these observations and **does not isolate engine speedup**. Active-time weighting also excludes failed or stalled periods; it is not a measure of job reliability, completed useful work or quality. The TF window **precedes** the 14 October short-prefill kernel repair; its errors are described in [the repair note](TENSORFOLD-HC-PREFILL-FIX.md). These historical rates are not a post-repair rerun.

As an independent sampling check, the legacy observer gives C4 C1/C2/C3/C4 rates of 51.13 / 71.03 / 105.87 / 103.65 tok/s, close to the monitor-derived 50.61 / 69.09 / 103.58 / 100.34. Different sampling intervals account for some discrepancy; the comparison keeps one declared source per row.

## Published data and recalculation

- [Full saved observation aggregates, exact windows, sample counts and independent legacy check](../results/20261014-hermes-real-workload/observations.json).
- [Unrounded summary and gains, including C5–C8](../results/20261014-hermes-real-workload/summary.json).
- [Offline numerical audit](../tools/audit_real_workload.py).
- [New poster](../assets/tf351-real-workload.png).

```sh
python3 tools/audit_real_workload.py
```

This recalculates rates and gains from the published saved aggregates without contacting a model or stopping production. It does not recreate the private production tasks. The original monitor archive is identified by SHA256 in the observations but is not published; it may contain operator details. The public export contains no user prompts, generated private text, API credentials or cluster addresses.

To obtain a comparable **new** operational window, record the active version, shared pool, context/reasoning settings, timestamps, concurrency and a valid live decode rate or streaming token-counter deltas. Keep invalid intervals and restarts separate. Export only metric aggregates. Run the existing [exclusive Hermes/Owen comparison](TENSORFOLD-BENCHMARK.md) if a controlled comparison is required instead.

## Pool and benchmark distinction

TF351's production observation used the **10,000,000-token shared pool**. The earlier comparative speed/quality campaign used **3.3M**, while 10M capacity was separately stress-qualified. The poster retains the separately measured cold-prefill results and labels them as benchmarks; those rates are not derived from this production table.
