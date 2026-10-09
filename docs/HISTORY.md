# Tuning history

Each series is a separate measurement campaign. Values below retain their original metric scope. C1/C3 are request decode; C8 is aggregate output per wall second. A historical label is not an independently rerun baseline.

| Campaign | Version | C1 decode | C3 decode | C8 aggregate | 64K computed prefill |
|---|---|---:|---:|---:|---:|
| expanded-original | 0.8.9X | 58.54 | 29.75 | 132.67 | 1494.30 |
| expanded-original | 0.8.9X1 | 68.36 | 32.00 | 144.66 | 1516.67 |
| expanded-original | 0.8.9X2 | 68.21 | 31.92 | 144.36 | 1525.12 |
| expanded-original | 0.8.9X3 | 69.74 | 33.44 | 146.08 | 1505.49 |
| expanded-original | 0.8.9X4 | 70.08 | 32.59 | 139.49 | 1490.75 |
| expanded-original | 0.8.9X5 | 69.36 | 33.09 | 141.26 | 1507.17 |
| expanded-original | 0.8.9X6 | 64.81 | 31.72 | 147.37 | 1492.60 |
| expanded-original | 0.8.9X7 | 62.78 | 29.79 | 141.21 | 1513.56 |
| expanded-original | 0.8.9X8 | 60.06 | 33.17 | 149.05 | 1484.90 |
| expanded-original | 0.8.9X9 | 56.42 | 31.50 | 155.33 | 1478.08 |
| expanded-original | 0.8.9X10 | 65.90 | 32.89 | 140.00 | 1481.54 |
| expanded-original | 0.8.9X11 | 59.82 | 33.57 | 158.27 | 1488.89 |
| followup-fresh | 0.8.9X11 | 66.79 | 32.69 | 146.52 | 1479.20 |
| followup-fresh | 0.8.9X11a | 56.10 | 32.33 | 148.37 | 1488.95 |
| followup-fresh | 0.8.9X11b | 61.76 | 28.97 | 149.92 | 1422.88 |
| followup-fresh | 0.8.9X11c | 64.17 | 31.91 | 153.52 | 1468.11 |
| followup-fresh | 0.8.9X11d | 63.71 | 32.73 | 148.74 | 1410.85 |
| followup-fresh | 0.8.9X11e | 56.84 | 28.87 | 143.71 | 1409.50 |

## Variant definitions

| Version | Difference |
|---|---|
| 0.8.8 | Established rollback baseline, 32 Engram readers |
| A | Native small-batch Engram processing with one reader; operational candidate, not part of the final paired campaign |
| A1 / 0.8.9 | Native conversion plus restored 32 parallel disk readers |
| B1 | Dense GEMV and fused MoE preparation experiment, based on the credited sfxnz sources |
| A1+B1 | Combined experiment; gains did not simply add |
| B2 and A1+B2 | Cooperative EXL3 expert path rejected; failed qualification is retained without an invented throughput score |
| X | 0.8.9 plus overlapping Engram reads, large-row projection/indexer partitioning and draft TP work |
| X1 | X plus R small TP communication |
| X2 | X1 plus K small-row EXL3 expert kernels |
| X3 | X2 plus D draft projection parallelism |
| X4 | X3 plus I1 native indexer scores |
| X5 | X4 plus I2 top-k selection |
| X6 | X3 plus H B12X projections |
| X7 | X3 plus M mHC |
| X8 | X3 plus P prefetch |
| X9 | X3 with fixed maximum draft K3 |
| X10 | X3 plus I1, I2, H, M and P; fixed K5 |
| X11 | X10 plus adaptive verification; generated maximum K5 |
| X11a | X11 plus adaptive startup profiling at 64K |
| X11b | X11 plus alternative graph capture sizes |
| X11c | X11 plus bounded prefetch, selected production |
| X11d | X11 plus generated maximum K4, adaptive verification retained |
| X11e | X11 plus genuinely dynamic generated draft length |

Read [ARCHITECTURE.md](ARCHITECTURE.md) for module scopes and the current settings. X11a through X11e are independent changes from X11.

## October 2 complete C1 to C4 comparison

This table reports weighted request decode in the original 256-token cold/hot tests. All individual aggregate, prefill, cache and acceptance measurements remain in the linked CSV.

| Version | Cache | Workload | C1 | C2 | C3 | C4 |
|---|---|---|---:|---:|---:|---:|
| 0.8.8 | cold | code | 53.52 | 22.15 | 13.36 | 10.11 |
| 0.8.8 | cold | prose | 46.73 | 20.35 | 11.95 | 7.49 |
| 0.8.8 | hot | code | 53.28 | 40.90 | 31.97 | 28.96 |
| 0.8.8 | hot | prose | 45.82 | 36.89 | 27.35 | 28.88 |
| A1 | cold | code | 48.45 | 22.65 | 13.07 | 10.13 |
| A1 | cold | prose | 53.18 | 20.41 | 12.12 | 8.08 |
| A1 | hot | code | 67.44 | 40.24 | 30.40 | 28.06 |
| A1 | hot | prose | 46.68 | 41.18 | 29.66 | 26.98 |
| B1 | cold | code | 52.01 | 22.84 | 13.52 | 9.68 |
| B1 | cold | prose | 51.65 | 19.16 | 11.82 | 8.59 |
| B1 | hot | code | 60.30 | 41.24 | 30.79 | 26.84 |
| B1 | hot | prose | 50.27 | 32.04 | 28.53 | 28.88 |
| A1+B1 | cold | code | 47.61 | 22.60 | 12.76 | 9.84 |
| A1+B1 | cold | prose | 53.72 | 19.55 | 11.77 | 8.49 |
| A1+B1 | hot | code | 64.97 | 41.20 | 28.49 | 27.72 |
| A1+B1 | hot | prose | 48.47 | 37.95 | 30.77 | 27.20 |

[Full numeric CSV](../results/20261002-six-version/MEASUREMENTS.csv). B2 rejection evidence is in the same directory.

B2 produced CUDA error 716, a misaligned address, on all three ranks. The stock B1 expert path passed the corresponding check, which localized the failure to the cooperative launch. An unaligned shared BF16 buffer passed to a `float4` load was a source-level hypothesis; the exact failing instruction was not established. After three corrections, B2 and A1+B2 were skipped without relaxing numerical qualification. See the retained failure analysis for the distinction between evidence and hypothesis.

## Earlier tests and rejected approaches

The September 14 0.8.1 versus 0.8.2 comparison changes prefill chunking from 1,536 to 4,096 tokens. It includes cold/warm C1 through C6 and task completion evidence, with output lengths that differ. Warm prefixes were explicitly primed. The complete numeric CSV and measurement audit are in `results/20260914-081-082/`. It is not the same benchmark as the later capped-output campaign.

The separate T02a exact MXFP8 tactic cache experiment against 0.8.3 showed a geometric mean decode change of -2.32% in one C1 through C6 pass. C1 was +1.82%; C2 through C6 were -3.42%, -3.21%, -3.97%, -2.59% and -2.45%. Isolated faster kernels did not establish a faster model. Detailed numeric evidence is in `results/20260914-T02a/`.

Operational versions 0.8.4 through 0.8.7 included cache, allocator, diagnostic and stability work. They are not assigned invented rows in this table. The later 0.8.8 baseline is the recorded stable comparison point. Early production A/A1 comparisons used observational traffic and cache cohorts; see `results/20261001-observational-A1-088/`, which is not a synthetic paired replay.

Changing trained Markov rank, weights, expert parallelism or installing TensorFold was not part of the selected tuning sequence. NVFP4 and other model recipes were studied as alternatives and are excluded from this EXL3 performance claim.

## Why current X11c

The fresh X11 to X11c comparison gave C8 aggregate +4.77%, C1 request decode -3.92%, C3 request decode -2.40%, and 64K computed prefill -0.75%. Real long-context Hermes work and observed behavior motivated retaining X11c. The preference is not an assertion that every individual metric improved. The subsequent reasoning comparison is a separate experiment in `results/20261009-X11c-reasoning/`.

Private captured bodies and generated private text are omitted from public evidence. Public metrics preserve durations, counts, cache state, scoring and incomplete phases; public surrogate inputs must be measured as a new benchmark.
