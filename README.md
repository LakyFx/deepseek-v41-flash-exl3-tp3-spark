# DeepSeek V4.1 Flash EXL3 on three DGX Sparks

**DS41F 1.0.0 is campaign C4**, our selected production recipe for Hermes agent work: TP3 on three DGX Sparks, the same approximately 3.51 bpw Pollard EXL3 experts, MAX reasoning, long prompts and a measured **3,299,563-token KV pool**.

The model, quantization, engine and upstream kernels are other people's work. Our contribution is integration, TP3 tuning, bounded correctness checks and workload measurements. Read [sources and credits](docs/SOURCES.md). The second benchmark is explicitly adapted from [Christopher Owen's spark-ds41f recipe](docs/OWEN-BENCHMARK.md).

## Start here

- [Final 1.0.0 recipe and technical explanation](docs/DS41F-1.0.0.md): exactly what C4 contains and why it was selected.
- [Complete latest campaign](docs/CAMPAIGN-20261012.md): all 18 measured arms, isolated changes, combinations, Hermes and Owen tables, C1?C8 curve, incomplete phases and tradeoffs.
- [Earlier tuning history](docs/HISTORY.md): 0.8.x, A/B, X1?X11, X11a?e and reasoning tests, including rejected changes.
- [Install](docs/INSTALL.md), [host environment](docs/HOST-ENVIRONMENT.md), [verification scope](docs/VERIFICATION.md), [architecture](docs/ARCHITECTURE.md).
- [Run our benchmark](docs/BENCHMARKS.md), [run the Owen subset](docs/OWEN-BENCHMARK.md), [reconstruct experimental variants](docs/EXPERIMENT-REPLAY.md).

Agents should read [AGENTS.md](AGENTS.md). The launcher takes your topology and paths; private production addresses are not part of the recipe.

## Selected settings

| Setting | DS41F 1.0.0 / C4 |
|---|---|
| Engine / parallelism | Pinned vLLM + CUDA EXL3, TP3, PP1, EP disabled |
| Checkpoint | [Pollard EXL3 3.5 bpw](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard), pinned revision; no weight change during tuning |
| Host | Kernel `7.0.0-1019-nvidia-64k`, 64K pages, driver `580.178.04`, signed memory-saver packing |
| KV | 8,074,035,200 bytes per rank, native `fp8_ds_mla`, 3.30M observed logical tokens |
| Context / scheduler | Maximum 401,408 tokens, eight sequences, 4,096-token prefill chunks |
| Engram | Local disk, 32 readers, native conversion and overlapped reads |
| Draft | DSpark K5, probabilistic draft/block rejection, adaptive verification with eligible C1 full-proposal path |
| Reasoning | Main MAX; vision medium mapped to native `high` |
| C4 additions | N1 + N2 + N3 + N4 + N6 over the qualified X11c host64k baseline |
| Long prefill | Bounded decoder replay with cache, indexer, vision and row-partitioning integration |

The KV pool is shared capacity, not a per-request context limit or a guarantee that eight maximum-length requests always fit. The 3.51 bpw figure describes routed experts, not every tensor in the hybrid checkpoint.

## Measured tradeoff

| Same-campaign metric | B0: X11c host64k | C4: 1.0.0 |
|---|---:|---:|
| Hermes C1 request decode tok/s | 62.3 | 63.6 |
| Hermes C3 request decode tok/s | 31.2 | 34.1 |
| Hermes C8 hot output/wall tok/s | 159.0 | 146.1 |
| Hermes cold 64K computed prefill tok/s | 1,318.0 | **1,734.1** |
| Owen code C8 decode-window tok/s | 192.9 | **194.5** |
| Owen cold nominal 256K input/TTFT tok/s | 1,230.2 | **1,687.2** |

C4 improved the measured long prefill by about 32?37% in these two distinct tests. It was one of only two arms to complete the bounded 128K-C3 cold/hot phase. **It did not win every decode cell:** Hermes C8 declined, while other candidates were stronger there. We chose the long-context compromise for our workload. The original Hermes 70+ C1 and 180+ C8 targets were not both achieved.

These are different metric definitions, not interchangeable speeds. Hermes has one campaign pass per arm; Owen averages three measured decode repeats. We publish output lengths, acceptance, cache evidence and incomplete phases, not confidence intervals or a universal speedup claim.

## Source and evidence

| Path | Contents |
|---|---|
| `config/ds41f-1.0.0.json` | Current exact profile; `config/x11c.json` is historical |
| `runtime/rootfs/`, `native/` | Selected C4 sources and rebuild inputs |
| `runtime/source-provenance.json` | Measured source and portable export hashes |
| `experiments/20261012/` | Baseline and per-arm source/settings archive |
| `benchmarks/` | Public Hermes workloads and frozen Owen common-context adapter |
| `results/` | All retained historical campaigns and current numerical envelopes |
| `licenses/` | Third-party notices |

Private captured legacy prompts and generated private text are excluded. The public legacy surrogates can be measured as a new comparison; they are not identical historical inputs. Production C4 is tested; a newly built portable image still needs qualification on your cluster. This repository update did not reload the running model.
