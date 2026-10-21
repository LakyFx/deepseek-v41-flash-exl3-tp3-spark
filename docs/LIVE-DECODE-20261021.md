# Live aggregate decode: 1.2.2TF vs 1.2.4

This is the evidence behind the current [comparison poster](../assets/tensorfold-1.2.4-comparison.png). It was recovered from archived monitor observations during the same benchmark passes; no new model run was made.

Rates are **aggregate tok/s across active model requests**, from TensorFold health `live.decode_tokens_per_second`: a native rolling gauge sampled approximately every two seconds. They are not per-request rates multiplied by concurrency, and TP rank rates are not added together.

| Active C | 1.2.2TF median | 1.2.2TF peak | 1.2.4 median | Median gain | 1.2.4 peak | Samples old / new |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 75.5 | 104.5 | 87.0 | +15.2% | 108.5 | 18 / 17 |
| C2 | 101.5 | 130.5 | 117.5 | +15.8% | 137.5 | 16 / 15 |
| C3 | 119.0 | 160.0 | 128.0 | +7.6% | 177.5 | 21 / 21 |
| C4 | 116.5 | 201.0 | 139.0 | +19.3% | 155.5 | 31 / 20 |
| C5 | 122.5 | 210.5 | 156.5 | +27.8% | 178.0 | 32 / 26 |
| C6 | 123.5 | 223.5 | 147.5 | +19.4% | 233.0 | 37 / 28 |
| C7 | 131.0 | 224.0 | 154.0 | +17.6% | 246.0 | 32 / 29 |
| C8 | 142.0 | 206.5 | 167.0 | +17.6% | 244.0 | 34 / 28 |
| C9 | 140.0 | 224.0 | 168.5 | +20.4% | 234.0 | 50 / 41 |
| C10 | 153.5 | 224.0 | 173.0 | +12.7% | 256.5 | 43 / 35 |
| C11 | 153.0 | 227.5 | 181.0 | +18.3% | 282.0 | 48 / 37 |
| C12 | 152.5 | 224.0 | 182.0 | +19.3% | 264.0 | 56 / 46 |

## How to read these numbers

**Median** is the duration-weighted inverse-CDF p50 of positive decode gauges while exactly C model requests are active inside that C-workflow wave. **Peak** is the largest observed native rolling gauge in that selection. The **282 tok/s at C11** is a sampled short-window peak; it is not sustained throughput or a guaranteed instantaneous hardware maximum.

Idle, prefill-only periods and lower-concurrency tails do not enter this live-decode median. Slow positive observations are retained. All time, including tools, prefill and waiting, remains in the separate [completed-workflow result](WORKFLOW-BENCHMARK-20261021.md): **22:27.87 → 19:14.48**, **16.8% more correct tasks/minute**, **78/78 tasks in each version**.

One pass per version. Short coverage (<60 seconds or <30 observations) is flagged in `summary.csv`: original C1–C3 and new C1–C8. Output lengths and repair paths differ. The fixture uses authored tasks with real tools, code execution, tests and repair rounds; it is not a replay of unmodified production sessions.

## Why the C6 percentages differ

| Measurement at C6 | 1.2.2TF | 1.2.4 | Change |
| --- | ---: | ---: | ---: |
| Live aggregate decode median, tok/s (current poster) | 123.5 | 147.5 | +19.4% |
| Weighted native decode per request, tok/s | 23.61 | 28.72 | +21.7% |
| Correct completed tasks per minute | 3.13 | 3.96 | +26.8% |

These describe the same benchmark cell with different denominators. They are not interchangeable speedup claims. The **16.8%** announcement figure covers all C1–C12 waves, not just C6.

## Evidence and selection

The [numeric archive](../results/20261021-live-decode/) contains `samples.csv`, `summary.csv`, `metadata.json` and `SHA256.json`. Samples are selected by exact backend/model generation (7 for original, 27 for 1.2.4), online status, positive native decode, valid duration (0 < dt ≤ 15 s), exact active concurrency, and the corresponding archived cell’s before/after request, prompt and completion counters. This excludes lower-C tails from other waves and unrelated live workloads.

The metadata publishes those counter bounds and the same 24 source-cell hashes as the [frozen workflow archive](../results/20261021-workflow/metadata.json). Full-generation final counters also matched: original **350 requests / 738,845 input / 164,239 output**, 1.2.4 **345 / 715,755 / 160,855**. Private full monitor rows are retained locally; only the curated numeric whitelist is public.

Recalculate from the repository root using only the Python standard library:

```python
import csv, json, hashlib
from pathlib import Path
p = Path("results/20261021-live-decode")
for name, expected in json.loads((p / "SHA256.json").read_text()).items():
    assert hashlib.sha256((p / name).read_bytes()).hexdigest() == expected
samples = list(csv.DictReader((p / "samples.csv").open(newline="")))
for arm in ("original", "cdef"):
    for c in range(1, 13):
        rows = [r for r in samples if r["arm"] == arm and int(r["C"]) == c]
        total = sum(float(r["duration_seconds"]) for r in rows)
        elapsed = 0.0
        for r in sorted(rows, key=lambda r: float(r["native_aggregate_decode_tps"])):
            elapsed += float(r["duration_seconds"])
            if elapsed >= total / 2:
                median = float(r["native_aggregate_decode_tps"])
                break
        peak = max(float(r["native_aggregate_decode_tps"]) for r in rows)
        print(arm, c, median, peak)
```
