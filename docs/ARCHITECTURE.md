# What X11c changes

The trained model, tokenizer and hybrid checkpoint remain the same. This is vLLM on three DGX Sparks, with tensor parallelism and PP1. Expert parallelism and TensorFold are not enabled. Approximately 3.51 bpw describes routed EXL3 experts; the checkpoint also contains native dense, shared, draft, Engram and vision tensors at their own precisions.

The original 64 attention heads and eight output groups do not divide evenly across TP3. The starting recipe pads virtual attention heads to 72 and output groups to nine, retaining metadata describing the original shape. Loading and attention patches map these virtual slots to the original weights. This is a compatibility mechanism, not a newly trained model. The Markov rank 256 is a checkpoint dimension and remains unchanged.

## Base X behavior

- A1 preserves 32 Engram disk-reader threads and converts eligible small batches with native C code. It does not turn all disk I/O into one reader.
- X overlaps eligible Engram reads with GPU work, while keeping ownership and ordering checks. A batch still waits when it actually needs the prepared embedding rows.
- X distributes eligible large input-row work, attention indexer projection work, main draft projection work and Markov projection work across TP ranks. The large-row threshold is 2,048. This changes host/GPU scheduling and can make prefill and decode appear more interleaved.
- Prefix retention is 128 and prefill is chunked at 4,096 tokens. Decode can share scheduling intervals with arrivals and tool follow-up requests. Repeated prefill bars do not prove that an unchanged prefix is recomputed; use computed and cached token counters.

## X11 components

| Module | Change | Important scope |
|---|---|---|
| R | Sparknet one-shot RoCE for eligible small TP collectives | Bulk transfers retain NCCL; a failed attempted collective is fatal, not silently replayed through another backend |
| K | Small-row MUL1 K3/K4 EXL3 expert tile | Retains the EXL3 decoder, trellis and arithmetic contract for the qualified TP3 dimensions |
| D | Parallel draft projection work | Same trained draft weights and sampling policy |
| I1 | Native indexer score computation | Selective eligible geometry with reference checks |
| I2 | Optimized indexer candidate selection/top-k | Selection metadata and masking remain part of correctness checks |
| H | B12X dense, attention and shared projections | Does not replace EXL3 routed expert weights |
| M | mHC implementation optimization | Preserves model dimensions and numerical qualification requirements |
| P | Weight prefetch to L2 overlapped with TP communication | Cache bandwidth and launch overhead can compete with useful work |
| A | Adaptive verification with maximum generated draft K5 | Shortens selected verification work; it does not dynamically generate a shorter draft |

Sampling remains probabilistic for draft sampling and block rejection for verification. A speculative speedup depends on both acceptance and the costs of drafting, verifying, communicating and reading Engram. A higher acceptance fraction alone is not a complete performance result.

## X11c selection

X11c applies one change to X11: limit prefetch to 12 through 48 physical rows, with at most 8 MiB per FFN opportunity, 12 MiB per attention opportunity and one CTA. Physical rows include speculative/graph padding and are not equal to HTTP concurrency.

The separate X11a, X11b, X11d and X11e experiments respectively changed the adaptive profile context, CUDA graph buckets, maximum draft K4, or genuinely dynamic generated draft depth. They are alternatives built from X11, not accumulated changes in X11c.

We selected X11c because it retained a useful C1 and improved the short C8 aggregate result in the fresh comparison, while real Hermes work often has long prompts and overlapping tasks. The measured 64K prefill rate remained approximately 1.47K tok/s. The selection is a workload preference, not evidence that every concurrency or prompt becomes faster.

KV uses the native `fp8_ds_mla` layout, with an explicit 8,074,035,200-byte per-rank allocation. The resulting logical pool is approximately 3.30 million tokens in the observed build. This shared pool is not a per-request context limit. The maximum model context is 401,408 and `max_num_seqs` is eight, but memory accounting and attention work still constrain actual simultaneous long contexts.
