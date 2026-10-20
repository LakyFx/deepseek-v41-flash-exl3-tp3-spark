# The 1.2.3 online learning layer for speculative verification

Version **1.2.3 is the release name of the configuration measured as 1.2.2xTF**.
It retains the same target model, draft weights, target verifier and sampling
path. It learns the runtime economics of candidate verification: which extra
rows are likely to emit tokens, and how much those rows cost at the current
load and context length. This is not model-weight training or learning from
whether an agent's tool command was correct.

## Inspiration and attribution

We were inspired by **Liquan Liu, Yifan Zhang and Bowei Xu's
[AdaSpark: Adaptive DSpark with Online Learning for Tree Verification and N-gram Fill](https://arxiv.org/abs/2610.05774)**
and its [imparo implementation](https://github.com/zeraix/imparo).
In particular, we adapted the ideas of online verification-cost feedback,
acceptance calibration and pricing verification time at the observed decode
rate. Our implementation remains in TensorFold's existing head/copy chain
lanes: it is **not a full reproduction of AdaSpark's tree construction,
learning models or imparo engine**. We retain pinned cost priors and use a
small rolling-median/bucket model. The paper's speedups are not our results.

The native engine and existing speculative machinery are
[TensorFold, by Ashley Hart and its contributors](https://github.com/ashhart/TensorFold).
Our addition is the integration and bounded online controller shown in
[`adaptive.zig`](../tensorfold/adaptive/adaptive.zig) and the
[exact integration patch generator](../tensorfold/adaptive/patch.py).
The [complete workflow measurements](WORKFLOW-BENCHMARK-20261020.md) evaluate
the combined serving recipe, not the learner in isolation.

## What C12 and R48 mean

- **C12** is the serving slot cap. In the benchmark, C is the number of parallel
  agent workflows, each making multiple sequential model requests with tool
  gaps. Actual active native streams vary during a wave.
- **R48** is the maximum number of verification rows in a shared round,
  including mandatory anchor rows. It is not 48 draft tokens per request.
- The controller does not divide R equally by C. It compares legal candidate
  prefixes, preserving parent-before-child dependencies.
- A larger cap permits a larger round when useful; it does not force that
  size. The sampled C1 native-stream range was only 2–6 rows in this run.

Original 1.2.2TF and Bert1 already used cost-based dynamic allocation. The new
part is learning corrections to cost and acceptance, with average-reward
pricing, rather than introducing dynamic speculation for the first time.

## One round, from candidates to feedback

```text
existing head/copy candidates
       |
       v
estimate conditional acceptance and legal-prefix survival
       |
       v
price verification widths for load + context + sampling mode
       |
       v
choose a useful prefix allocation, up to R48
       |
       v
existing target verifier accepts/rejects token candidates
       |
       +--> measured verification time --> cost fit
       +--> reached-prefix labels       --> acceptance fit
       +--> emitted tokens / elapsed time --> average-reward price
```

Integration hooks are `verified()` after verification, `accepted()` for
reached chain-prefix outcomes before output/EOS truncation, and `finished()`
for overhead/reward-rate accounting. The module changes selected rows; it
does not replace the target sampling/verification decision.

## Verification-cost model

The starting table is the original R1–R32 table. Its R25–R32 entries were
already extrapolated; R33–R48 extend the prior by **3.1 ms per row**. These
extensions are priors, not independently measured width sweeps.

Each cost bucket distinguishes:

| Feature | Buckets |
|---|---|
| Sampling | Greedy / sampled |
| Active native streams | 1, 2, 3–4, 5–6, 7–8, 9–16 |
| Context | ≤8K, ≤16K, ≤32K, ≤64K, ≤128K, ≤256K, larger |
| Verification width | Exact row count, up to 48 |

Context is the mandatory-row-weighted mean. Choosing more speculative rows
does not redefine that feature and accidentally make a width look cheaper.
For a bucket/width, we collect actual verification time divided by prior
time, skip the first two sightings to reduce cold shape/graph effects, and
retain the **latest 16 ratios**. Their median supplies a robust correction.

Sparse exact-width cells borrow a correction within the same load/context/mode
bucket, then refine it with trust `n / (n + 8)`. They do not borrow across
different context or load bands. The final multiplier is clipped to **0.5–3×**.
Invalid/nonfinite timing and measurements outside **0.25–8×** baseline are
rejected. Process and memory guards are separate operational mechanisms.

Overhead has EWMA smoothing **α=0.1**. Emitted-token and elapsed-time averages
use **α=0.025**. The reward price becomes available after 16 observations in
the bucket:

```text
rho = smoothed emitted tokens / smoothed elapsed time

cold objective = expected emitted tokens / predicted total round time
warm objective = expected emitted tokens - rho * predicted total round time
```

This is a ratio of smoothed totals, not a mean of per-round rates. It avoids
overweighting very short rounds. The output proxy includes mandatory rows
plus the expected survival of selected drafts.

The controller scans legal prefix widths up to R48. A 2% score margin keeps
the incumbent when changes are marginal. Undermeasured neighbouring widths
may receive a useful probe at most every 64 rounds, within a 4% utility
margin. A probe is an ordinary useful verification round, not an extra GPU
benchmark call.

## Acceptance calibration

Raw cumulative survival estimates are converted to conditional probabilities.
Calibration buckets separate sampling mode, head/copy source, 16 depth bins
(depths ≥15 share the final bin), and eight raw-confidence bins.

For a reached position, the target outcome gives a success/failure label.
After the first mismatch, later positions were not reached: they are
**censored**, not recorded as failures. The prediction blends the prior with
decaying observations:

```text
retain = 1 / (1 + age_in_rounds / 512)
q = (32 * prior + successes * retain) / (32 + trials * retain)
```

Counts halve at 128 trials to allow adaptation. Unobserved buckets gradually
return toward the current prior. A per-mode/source guard enables the learned
prediction only after at least 32 observations and while its EWMA log loss
is no worse than **1.01×** the raw prior. Otherwise it falls back to the
original confidence. Path survival is the product of the conditional
probabilities; parent prefixes remain intact.

## Lifetime, shape limits and fallback

The learner is process-local and shared across requests. Opening a new chat
does not reset it. Restarting the native process discards it and returns to
the pinned priors; **there is no disk persistence**. Learning can track a
changing workload, but longer runtime does not guarantee monotonic improvement.

The opt-in is `TF_DS_ADAPTIVE_LEARNING=1`, on all three ranks in the released
profile. The implementation rejects unsupported adaptive shapes: more than
48 rows, more than 16 streams, or the separate head-tree mode. Invalid
configuration is an error rather than an implicit promise of support.

## What was measured

| xTF telemetry in the retained workflow run | Value |
|---|---:|
| Controller rounds | 11,672 |
| Cost updates / timing prediction samples | 11,267 |
| Acceptance labels | 112,076 |
| Nearby-width probes | 2 |
| Rejected timing samples | 0 |
| Sum absolute prediction error / sum verification time | 4.7605% |
| Mean absolute prediction error | 4.10 ms |
| Summed measured verification time for those samples | 969.52 s |

The 4.76% statistic is `sum(abs(predicted - actual)) / sum(actual)`.
It is not MAPE, a convergence guarantee or “95.24% prediction accuracy”.
The [full table of sampled row ranges](WORKFLOW-BENCHMARK-20261020.md#learner-telemetry)
uses 3-second telemetry samples, not a complete per-round history.

## Acceptance is not semantic task accuracy

Standard speculative inference can preserve the target distribution under
its algorithmic assumptions; see
[Leviathan, Kalman and Matias (2023)](https://proceedings.mlr.press/v202/leviathan23a.html).
The verifier nevertheless does not know whether a tool command or accounting
answer is correct. A higher acceptance fraction is an efficiency statistic.

The measured xTF recipe had fewer rejected intermediate tool attempts, but
the experiment simultaneously changed row cap, KV budget, snapshot-cache
size and allocation policy. Tool trajectories and output lengths diverged.
It does not prove that this learner improves the target model's reasoning,
or that a larger prompt cache prevents long-session repetition loops.

## Inspect or rebuild it

[`manifest.json`](../tensorfold/adaptive/manifest.json) pins the pristine base,
the exact integrated source hashes from the measured build, and the overlay
hashes. The portable patcher verifies all proposed output bytes before writing
the extracted source. [`build.py`](../tensorfold/adaptive/build.py) then runs
CPU tests and the ARM64 build, recording the resulting binary hashes. The
[1.2.3 recipe](DS41F-1.2.3.md) uses a separate binary directory and requires
that receipt before `--run`.
