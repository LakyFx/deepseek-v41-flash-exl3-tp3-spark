# DeepSeek V4.1 Flash EXL3 on three DGX Sparks

**A big thank you to [Christopher Owen](https://github.com/christopherowen), [@christopherowen on X](https://x.com/christopherowen), for sharing his recipes, networking work and benchmarks. His [spark-ds41f project](https://github.com/christopherowen/spark-ds41f) was a major source of inspiration for this recipe and our tuning work.**

**DS41F 1.0.0 is campaign C4**, our selected production recipe for Hermes agent work: TP3 on three DGX Sparks, the same approximately 3.51 bpw Pollard EXL3 experts, MAX reasoning, long prompts and a measured **3,299,563-token KV pool**.

The model, quantization, engine and upstream kernels are other people's work. Our contribution is integration, TP3 tuning, bounded correctness checks and workload measurements. Read [sources and credits](docs/SOURCES.md). The second benchmark is explicitly adapted from [Christopher Owen's spark-ds41f recipe](docs/OWEN-BENCHMARK.md).

## Start here

- [Final 1.0.0 recipe and technical explanation](docs/DS41F-1.0.0.md): exactly what C4 contains and why it was selected.
- [Complete latest campaign](docs/CAMPAIGN-20261012.md): all 18 measured arms, isolated changes, combinations, Hermes and Owen tables, C1?C8 curve, incomplete phases and tradeoffs.
- [Earlier tuning history](docs/HISTORY.md): 0.8.x, A/B, X1?X11, X11a?e and reasoning tests, including rejected changes.
- [Install](docs/INSTALL.md), [host environment](docs/HOST-ENVIRONMENT.md), [verification scope](docs/VERIFICATION.md), [architecture](docs/ARCHITECTURE.md).
- [Run our benchmark](docs/BENCHMARKS.md), [run the Owen subset](docs/OWEN-BENCHMARK.md), [reconstruct experimental variants](docs/EXPERIMENT-REPLAY.md).

Agents should read [AGENTS.md](AGENTS.md). The launcher takes your topology and paths; private production addresses are not part of the recipe.

<!-- variant-catalog:start -->
## Choose your variant

**18 measured recipes. Same EXL3 weights. Different workload tradeoffs.**

| Your priority | Start by comparing | Why |
|---|---|---|
| Long-context agent work | **C4 / DS41F 1.0.0**, N6 | The only two arms to finish the full bounded 128K-C3 cold/hot phase |
| Hermes single-request decode | C3, C5-indexer | Highest and second-highest measured Hermes C1 |
| Hermes eight-request throughput | C5-indexer, N8b, B0 | Highest measured Hermes C8 wall throughput |
| Code generation at eight requests | N3, N1, N4 | Highest measured Owen code C8 decode-window throughput |
| Reference comparison | B0 | Fresh X11c baseline on the same upgraded hosts |

These are candidates to evaluate, not universal winners. **C4 is our production choice; the others are experimental alternatives.**
Click any variant below for its changes, tradeoffs, build commands and verification scope.
[All variant recipes](docs/VARIANTS.md) · [Reproduction guide](docs/EXPERIMENT-REPLAY.md) · [Verification status](docs/VARIANT-VALIDATION.md)

### Hermes: our agent workload

All values are **tok/s**. C1/C3 are weighted **per-request engine decode**; C8 is **aggregate output / cohort wall time**, including first-token latency. Prefill counts newly computed tokens over native prefill time. This is not one concurrency scaling curve.

| Variant | C1 decode | C3 decode | C8 wall | 64K prefill | 128K prefill |
|---|---:|---:|---:|---:|---:|
| [B0](docs/VARIANTS.md#b0) | 62.3 | 31.2 | 159.0 | 1,318.0 | n/a |
| [N1](docs/VARIANTS.md#n1) | 58.5 | 29.7 | 156.0 | 1,283.4 | 1,209.8* |
| [N2](docs/VARIANTS.md#n2) | 62.1 | 31.9 | 158.5 | 1,405.0 | 1,291.3* |
| [N3](docs/VARIANTS.md#n3) | 55.1 | 32.7 | 151.2 | 1,284.8 | n/a |
| [N4](docs/VARIANTS.md#n4) | 63.5 | 32.7 | 150.7 | 1,400.2 | 1,242.8* |
| [N5](docs/VARIANTS.md#n5) | 65.4 | 33.0 | 154.6 | 1,337.4 | 1,264.8* |
| [N6](docs/VARIANTS.md#n6) | 59.4 | 31.1 | 145.7 | **1,770.3** | **1,694.4** |
| [N7a](docs/VARIANTS.md#n7a) | 65.3 | 33.1 | 149.8 | 1,286.6 | n/a |
| [N7b](docs/VARIANTS.md#n7b) | 63.9 | 29.7 | 155.2 | 1,285.7 | n/a |
| [N8a](docs/VARIANTS.md#n8a) | 65.9 | 28.9 | 157.0 | 1,280.0 | n/a |
| [N8b](docs/VARIANTS.md#n8b) | 60.6 | 30.5 | 159.6 | 1,351.1 | 1,226.3* |
| [C1](docs/VARIANTS.md#c1) | 59.2 | 33.1 | 148.6 | 1,400.0 | 1,292.0* |
| [C2](docs/VARIANTS.md#c2) | 61.0 | 30.9 | 155.9 | 1,342.1 | 1,186.0* |
| [C3](docs/VARIANTS.md#c3) | **68.1** | 31.3 | 149.1 | 1,279.6 | n/a |
| [C4 / 1.0.0](docs/VARIANTS.md#c4) | 63.6 | 34.1 | 146.1 | 1,734.1 | 1,643.7 |
| [C5-indexer](docs/VARIANTS.md#c5-indexer) | 66.7 | 29.8 | **162.1** | 1,284.9 | n/a |
| [C5-sp512](docs/VARIANTS.md#c5-sp512) | 60.7 | 31.5 | 153.8 | 1,402.8 | 1,282.4* |
| [C5-sp256](docs/VARIANTS.md#c5-sp256) | 65.4 | **34.3** | 158.9 | 1,347.6 | 1,211.3* |

`*` Cold prime measured, but the full 128K-C3 phase did not finish. `n/a` is missing, not zero. Only N6 and C4 completed that full phase.

### Christopher Owen benchmark: code

Adapted from [Christopher Owen's spark-ds41f](https://github.com/christopherowen/spark-ds41f), with [frozen source, inputs and methodology](docs/OWEN-BENCHMARK.md). Values are aggregate **decode-window tok/s**, averaged over three measured repeats after a discarded warmup.

| Variant | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| [B0](docs/VARIANTS.md#b0) | **59.8** | 84.9 | 127.3 | 192.9 |
| [N1](docs/VARIANTS.md#n1) | 55.7 | 85.6 | 132.9 | 205.6 |
| [N2](docs/VARIANTS.md#n2) | 56.9 | **90.3** | 132.8 | 196.5 |
| [N3](docs/VARIANTS.md#n3) | 57.5 | 83.5 | 131.2 | **208.5** |
| [N4](docs/VARIANTS.md#n4) | 55.5 | 85.2 | 119.7 | 204.2 |
| [N5](docs/VARIANTS.md#n5) | 58.5 | 82.0 | 130.6 | 191.0 |
| [N6](docs/VARIANTS.md#n6) | 56.8 | 84.5 | 130.4 | 187.8 |
| [N7a](docs/VARIANTS.md#n7a) | 56.8 | 81.8 | 128.4 | 196.7 |
| [N7b](docs/VARIANTS.md#n7b) | 58.9 | 87.3 | 128.3 | 201.2 |
| [N8a](docs/VARIANTS.md#n8a) | 56.9 | 79.8 | 130.4 | 195.1 |
| [N8b](docs/VARIANTS.md#n8b) | 57.9 | 84.2 | 127.8 | 192.8 |
| [C1](docs/VARIANTS.md#c1) | 56.0 | 84.7 | 126.9 | 202.1 |
| [C2](docs/VARIANTS.md#c2) | 56.7 | 82.3 | 129.0 | 195.9 |
| [C3](docs/VARIANTS.md#c3) | 56.3 | 78.0 | 128.0 | 190.9 |
| [C4 / 1.0.0](docs/VARIANTS.md#c4) | 57.3 | 83.7 | 129.0 | 194.5 |
| [C5-indexer](docs/VARIANTS.md#c5-indexer) | 56.8 | 78.5 | 127.2 | 194.0 |
| [C5-sp512](docs/VARIANTS.md#c5-sp512) | 57.2 | 82.9 | 131.1 | 198.6 |
| [C5-sp256](docs/VARIANTS.md#c5-sp256) | 55.9 | 86.3 | **133.4** | 201.3 |

### Christopher Owen benchmark: prose and cold prefill

Prose uses the same decode-window metric. Prefill is **input tokens / TTFT**, including first-token overhead; it differs from the Hermes native computed-prefill metric.

| Variant | Prose C1 | Prose C2 | Prose C4 | Prose C8 | 32K prefill | 256K prefill |
|---|---:|---:|---:|---:|---:|---:|
| [B0](docs/VARIANTS.md#b0) | 40.9 | 65.2 | 97.9 | 153.3 | 1,254.1 | 1,230.2 |
| [N1](docs/VARIANTS.md#n1) | 43.3 | 64.3 | 97.3 | 150.7 | 1,188.0 | 1,237.4 |
| [N2](docs/VARIANTS.md#n2) | 40.8 | 65.8 | 96.8 | **156.3** | 1,202.8 | 1,270.3 |
| [N3](docs/VARIANTS.md#n3) | 40.9 | 66.8 | 94.3 | 150.1 | 1,116.0 | 1,215.4 |
| [N4](docs/VARIANTS.md#n4) | 39.4 | 63.9 | 99.5 | 149.6 | 1,255.6 | 1,223.0 |
| [N5](docs/VARIANTS.md#n5) | 40.4 | 63.1 | **103.6** | 149.1 | 1,196.4 | 1,252.1 |
| [N6](docs/VARIANTS.md#n6) | **45.2** | 65.4 | 97.9 | 154.7 | **1,629.7** | 1,593.7 |
| [N7a](docs/VARIANTS.md#n7a) | 39.1 | 66.3 | 94.2 | 151.7 | 1,238.9 | 1,212.5 |
| [N7b](docs/VARIANTS.md#n7b) | 44.2 | **67.6** | 98.1 | 154.4 | 1,176.8 | 1,221.9 |
| [N8a](docs/VARIANTS.md#n8a) | 38.3 | 64.3 | 98.1 | 150.0 | 1,183.7 | 1,232.9 |
| [N8b](docs/VARIANTS.md#n8b) | 39.8 | 66.9 | 98.2 | 153.4 | 1,257.3 | 1,256.1 |
| [C1](docs/VARIANTS.md#c1) | 42.0 | 63.6 | 97.3 | 151.3 | 1,259.5 | 1,242.4 |
| [C2](docs/VARIANTS.md#c2) | 41.3 | 61.9 | 96.0 | 150.9 | 1,249.7 | 1,251.2 |
| [C3](docs/VARIANTS.md#c3) | 40.9 | 64.1 | 96.6 | 150.8 | 1,176.9 | 1,196.6 |
| [C4 / 1.0.0](docs/VARIANTS.md#c4) | 41.3 | 64.6 | 99.8 | 150.3 | 1,619.9 | **1,687.2** |
| [C5-indexer](docs/VARIANTS.md#c5-indexer) | 38.2 | 64.0 | 101.8 | 150.7 | 1,238.0 | 1,208.5 |
| [C5-sp512](docs/VARIANTS.md#c5-sp512) | 37.7 | 66.1 | 97.3 | 151.3 | 1,205.0 | 1,222.9 |
| [C5-sp256](docs/VARIANTS.md#c5-sp256) | 38.7 | 62.7 | 102.5 | 150.9 | 1,200.6 | 1,266.4 |

**Reading these tables:** bold marks the largest observed value in a column, not a statistically established winner. Hermes has one campaign pass per arm; Owen decode has three repeats. MAX reasoning was used for chat requests. Candidate names C1/C3/C4 are not concurrency levels. Results depend on prompt, acceptance, context and cache state. See [complete campaign limitations](docs/CAMPAIGN-20261012.md#completeness-correctness-and-interpretation).

### Reproduce a choice

Clone the current repository with tags, select an arm and build on Linux ARM64. For example:

```bash
git fetch origin --tags
python3 experiments/materialize.py --arm C5-indexer --output /srv/dsv41-build-c5-indexer
cd /srv/dsv41-build-c5-indexer
RECIPE_IMAGE=ds41f:experiment-c5-indexer bash build/build.sh
cp config/cluster.example.json config/cluster.local.json
```

Fill in your topology and paths, then follow [installation](docs/INSTALL.md#3-download-and-prepare-the-checkpoint) and [acceptance checks](docs/VERIFICATION.md#operator-acceptance). The commands above prepare/build only; they do not deploy. Source reconstruction is checked separately from public image builds and GPU qualification. Historical measurements do not certify a new image.

<!-- variant-catalog:end -->

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
