# DeepSeek V4.1 Flash EXL3 on three DGX Sparks

This recipe publishes our current **0.8.9X11c** configuration: three NVIDIA DGX Sparks using tensor parallelism, Pollard EXL3 routed experts at approximately 3.51 bits per weight, DSpark speculation, and a measured KV pool of **3,299,563 tokens**. We use it for long prompts and real work with the Hermes agent. The installation target is X11c; earlier variants are documented experiments.

The model, quantization format, inference engine, and upstream kernels are other people's work. Our contribution is a TP3 configuration, integration and tuning of existing ideas, numerical checks, and measurements. See [sources and credits](docs/SOURCES.md) before using or citing this recipe.

Start with the [installation guide](docs/INSTALL.md) and [technical changes](docs/ARCHITECTURE.md). Agents should also read [AGENTS.md](AGENTS.md). The included launcher accepts your topology instead of embedding our private network.

## What we selected

| Setting | X11c |
|---|---|
| Engine | Pinned vLLM with the EXL3 CUDA plugin and included overlays |
| Parallelism | TP3, pipeline parallelism 1, expert parallelism disabled |
| Weights | [bot-lab-21 Pollard EXL3 3.5 bpw](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard), revision `f129e31a81e1337aa33e129e2d847fc7e37c8733` |
| Weight changes during tuning | None |
| KV allocation | 8,074,035,200 bytes per rank |
| Observed logical KV pool | 3,299,563 tokens |
| KV format | CLI `fp8`, native `fp8_ds_mla`, block size 128 |
| Maximum model context | 401,408 tokens, including generated tokens |
| Scheduler | Up to 8 sequences, prefill chunk 4,096 tokens |
| Engram | Local disk rows, 32 reader threads, native conversion and overlapped staging |
| Draft | Maximum generated length K5, probabilistic draft sampling, block rejection sampling |
| Adaptive verification | Enabled; this does not dynamically shorten draft generation |
| Reasoning policy | Main agent max; vision medium, mapped to native `high` |
| Limited L2 prefetch | 12 to 48 input rows, FFN 8 MiB, attention 12 MiB, one CTA |

The pool is shared across requests. It does not mean eight independent 401K prompts and their outputs always fit simultaneously. Model context, logical pool size, admission policy and available host RAM are different limits. The observed pool comes from engine receipts, not a generic bytes-per-token extrapolation.

## Why X11c

We optimize for long context, reusable prefixes, tool calls and concurrent agent work. We retained the proven EXL3 checkpoint and approximately 3.3M pool instead of replacing them with another weight format or engine. X11c limits the prefetch work that the preceding X11 configuration performed on small batches.

In the fresh follow-up campaign, X11c reached **153.52 aggregate output tok/s at C8**, compared with X11's 146.52. Its C1 request decode was 64.17 versus 66.79 tok/s, and C3 was 31.91 versus 32.69. This is a workload tradeoff, not a win in every cell. X3 and fixed K3 were also competitive in particular regimes. We selected X11c after these tests and practical Hermes use, rather than selecting only the largest short-prompt decode number.

The benchmarks are single passes without confidence intervals. Acceptance and generated text vary, and X11 itself produced substantially different numbers in two campaigns. We do not claim a universal or statistically established 20 percent improvement.

## Reading the results

| X11c reasoning run | C1 request decode tok/s | C3 request decode tok/s | C8 aggregate output tok/s | 64K cold prefill tok/s |
|---|---:|---:|---:|---:|
| Baseline max | 67.50 | 32.12 | 154.38 | 1,467.88 |
| None | 69.97 | 24.98 | 146.25 | 1,456.36 |
| Medium, native high | 76.53 | 28.69 | 149.37 | 1,462.74 |
| Max repeated | 64.82 | 32.75 | 150.91 | 1,448.78 |

C1 and C3 measure weighted decode speed per request on scored agent workflows. C8 measures total output divided by cohort wall time, including first-token latency. These columns are **different metrics**, not one concurrency scaling curve. Non-reasoning and medium finished the scored workflows sooner mainly because they generated fewer tokens. All scored agent tasks passed in these four modes. That small test set does not establish equal general reasoning quality.

All four 128K C3 tests reached the original 325-second phase budget. Their incomplete hot measurements are not zero and are not a completed long-context throughput result. The other six phases completed. The four reasoning runs used one model load and the same running X11c generation.

Read [benchmark methodology](docs/BENCHMARKS.md), [version history](docs/HISTORY.md), and the machine-readable [results](results/) for the full scope, settings, failures and measured cells.

## Repository layout

| Path | Purpose |
|---|---|
| `config/x11c.json` | Selected engine, cache, draft, graph and prefetch settings |
| `config/dependency-versions.json` | Versions observed in the running production image |
| `runtime/rootfs/` | Mounted production Python and native source dependencies |
| `runtime/mounts.json` | Container destinations, without private host paths |
| `native/` | Sources needed to rebuild the modified EXL3 extension and fused transform |
| `build/base/` | Pinned base image construction and original recipe patches |
| `results/` | Historical measurements and incomplete or rejected outcomes |
| `licenses/` | Preserved third-party license notices |

The recipe uses configurable cluster inputs. Private IP addresses, SSH aliases, credentials, certificates, private conversations and the private runtime Git history do not belong in this repository.
