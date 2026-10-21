# Choose a recipe

**Default recommendation: [1.2.4 / CDEF](DS41F-1.2.4.md)** for the shortest
observed completed-work time in this campaign. **CF** is the simpler alternative
with the highest completed-wave aggregate peak. **C** has the highest pooled
native request decode rate. Choose by your workload; the archive supports all
three rather than declaring one winner for every metric.

[All C1–C12 speeds and latency definitions](WORKFLOW-BENCHMARK-20261021.md)
· [profiles](../tensorfold/profiles/) · [exact source recipes](../tensorfold/variants/recipes/).

## Matched-workflow versions

“C12” below is configured stream capacity. Benchmark C1–C12 is simultaneous
workflows, including tool steps. “R” is the maximum **shared verification-row
budget**, not that many drafts per request. The configured KV pool is shared;
these workload tests do not prove full-pool capacity. Snapshot cache is separate
from the live KV pool. All use prefill chunks of 2,048 tokens and the same pinned
1017 model, mixed EXL3 quantization and sampler.

| Selector | Measured name | Added mechanisms | Capacity / rows | KV pool | Snapshot cache | Recipe |
| --- | --- | --- | --- | ---: | ---: | --- |
| `1.2.2TF` | original | Static cost priors; historical reference | C12/R32 | 8M | 2 GiB | [source](../tensorfold/variants/recipes/original.json) |
| `Bert1` | bert1 | More streams/rows; same original native source | C16/R48 | 7M | 2 GiB | [source](../tensorfold/variants/recipes/bert1.json) |
| `1.2.3` | historical xTF / 1.2.2xTF | Online cost and acceptance feedback | C12/R48 | 7M | 3 GiB | [original release/build](DS41F-1.2.3.md) |
| `1.2.3-reference` | baseline | Same learner, common benchmark instrumentation | C12/R48 | 7M | 3 GiB | [source](../tensorfold/variants/recipes/baseline.json) |
| `A` | a | A | C12/R48 | 7M | 3 GiB | [source](../tensorfold/variants/recipes/a.json) |
| `B` | b | A + B | C12/R48 | 7M | 3 GiB | [source](../tensorfold/variants/recipes/b.json) |
| `C` | c | A + B + C | C12/R48 | 7M | 3 GiB | [source](../tensorfold/variants/recipes/c.json) |
| `D` | d | D | C12/R48 | 7M | 3 GiB | [source](../tensorfold/variants/recipes/d.json) |
| `E` | e | E | C12/R48 | 7M | 3 GiB | [source](../tensorfold/variants/recipes/e.json) |
| `F` | f | F | C12/R48 | 7M | 3 GiB | [source](../tensorfold/variants/recipes/f.json) |
| `G` | g | G | C12/R48 | 7M | 3 GiB | [source](../tensorfold/variants/recipes/g.json) |
| `CF` | cf | A + B + C + F | C12/R48 | 7M | 3 GiB | [source](../tensorfold/variants/recipes/cf.json) |
| `DEF` | def | D + E + F | C12/R48 | 7M | 3 GiB | [source](../tensorfold/variants/recipes/def.json) |
| **`1.2.4`** | **cdef** | **A + B + C + D + E + F** | **C12/R48** | **7M** | **3 GiB** | [source](../tensorfold/variants/recipes/cdef.json) |

A/B/C are cumulative experiments. D/E/F/G each start from the instrumented
1.2.3 baseline. **CDEF contains all six A–F mechanisms; it does not contain G.**
Every row has one complete 78/78 workflow pass. Source materialization and
compilation on another host still require local startup/functional qualification.

## What each mechanism actually does

| Mechanism | Plain-language purpose | Implementation boundary |
| --- | --- | --- |
| 1.2.3 learner | Learn how expensive verification is and how many draft tokens survive, then spend rows where expected useful tokens/second is best | Rolling cost timing and bucketed acceptance feedback, per process; no weight training and no persisted universal learned state |
| **A — actual verification shape** | Price the graph the engine will really execute, including the deepest active chain and actual active streams | Fixed and speculative rows, correction-row bounds and normalized median residual priors; avoids pricing a cheaper hypothetical shape |
| **B — reachable sampling rows** | Avoid sampling rows the current linear head/copy path cannot use | Retains correction rows, forced tokens and fallback/tree paths; does not change target sampling mathematics |
| **C — CPU/GPU overlap** | Prepare penalty scores on CPU while the GPU works | Four persistent workers, bounded owned seed/position data, cancel/drain before buffer reuse; verifier unchanged |
| **D — bounded exploration credit** | Keep timing refreshes from taking too much host time | Credit accrues from ordinary host-observed work at .002/.998, capped at 5 ms; a probe pays its full time, negative debt survives idle, positive credit clears on idle. Not a per-request 0.2% latency guarantee |
| **E — refresh stale cost banks** | Let old cost observations age out as the workload changes | 30 s TTL, reset on the next valid observation; combined with A, additive residual resets to zero and stale priors are excluded. Not continuous retraining |
| **F — decode between admissions** | Let existing streams decode after one new admission before taking another | Admission fairness; does not implement resumable/chunk-interleaved prefill |
| **G — coalesce Engram reads** | Share identical in-flight Engram AIO reads instead of issuing the same read twice | Identical physical reads only; no weight, disk-layout or KV-format change; separate optional experiment |

The learner changes the **amount of speculative work**, not which target tokens
are accepted. It observes actual runtime cost and draft acceptance on the fly.
A restart resets its transient state. More time running is not a promise of
monotonic speed improvement; E deliberately permits adaptation after workload
changes. [Original learner equations and implementation](ADAPTIVE-SPECULATION.md).

The campaign did not export D/E/F/G trigger counters, so it cannot establish
their individual contribution. A fast recipe-level result does not prove every
included mechanism helped in that particular short pass.

## Build and select any of these recipes

Follow [the common 1.2.4 build/install procedure](DS41F-1.2.4.md), replacing
`--recipe cdef` and `--variant 1.2.4` with the source recipe and selector above.
Each selector uses its own `bin-<selector>/BUILD.json` and verifies its exact
manifest and executable hashes. All three TP3 ranks must use the same recipe.
Historical `1.2.3` retains its separate [adaptive builder](../tensorfold/adaptive/build.py).
Do not combine an old executable with a new profile and call it the new release.

The manifests publish the full source-file hash map and minimal full-file
overlays. New-campaign manifests also retain measured executable hashes and
recorded CPU check status; original/Bert1 do not invent missing build receipts.
`source_tree_sha256` is SHA256 of compact sorted JSON of that complete map.
Objects are deduplicated by SHA256; they are source files, not binary downloads.
The archive/base/object/final-tree checks fail closed on source drift.

## Earlier engines and releases remain available

| Version / era | What it preserves | Where to start |
| --- | --- | --- |
| X11c / B0 | Historical vLLM baseline and tuning arms | [variants](VARIANTS.md), [18-arm campaign](CAMPAIGN-20261012.md) |
| C4 / 1.0.0 | Retained vLLM recipe and completed-work comparison | [recipe](DS41F-1.0.0.md), [results](TENSORFOLD-RESULTS.md) |
| TF351 / TF29 | TensorFold on earlier Pollard/Mia checkpoints, independent pool qualifications | [TF351](TF351.md), [TF29](TF29.md), [build](TENSORFOLD-INSTALL.md) |
| 1.2.0 | New pinned DeepSeek 1017 weights and preparation | [model/setup](DS41F-1.2.0.md) |
| 1.2.1TF / 1.2.2TF gateway era | Optional API/vision serving layer and two concurrent image jobs | [gateway](GATEWAY.md), [release history](HISTORY.md) |
| 1.2.3, originally 1.2.2xTF | Online learner and historical original/Bert1/xTF pass | [recipe](DS41F-1.2.3.md), [complete report](WORKFLOW-BENCHMARK-20261020.md) |
| **1.2.4 / CDEF** | **Combined A–F native runtime recipe** | **[setup](DS41F-1.2.4.md), [complete new report](WORKFLOW-BENCHMARK-20261021.md)** |

Older checkpoints, output lengths and measurement scopes differ. Their legacy
numbers are preserved in their own tables rather than mixed into the matched
1017 C1–C12 comparison. Gateway versions and native-engine versions have separate
responsibilities; the new native recipe does not replace vision behavior.

## Inspiration and attribution

The native engine is [Ashley Hart and contributors' TensorFold](https://github.com/ashhart/TensorFold).
The online learner was inspired by [AdaSpark](https://arxiv.org/abs/2610.05774).
A–C draw on ideas studied in [DeepSeek DeepInfer, pinned revision](https://github.com/deepseek-ai/DeepInfer/tree/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01);
D–G draw on [NVIDIA FLARE, pinned revision](https://github.com/NVIDIA/flare-local-inference/tree/47581ccac82660953a5b59cb04b2750d64edc97d).
These are adaptations inside our existing TensorFold Zig runtime, not complete
ports of those engines or claims to their published results. [All credits and licenses](SOURCES.md).
