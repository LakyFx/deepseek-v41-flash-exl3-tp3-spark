# DeepSeek V4.1 Flash: TensorFold on three DGX Sparks

Thank you to **[@ashxhart](https://x.com/ashxhart)**, creator of [TensorFold](https://github.com/ashhart/TensorFold), [tensorfold.dev](https://tensorfold.dev/). This engine made the new serving path possible. Thank you also to [bertholomus](https://github.com/bertholomus/TensorFold) for the community DeepSeek implementation, [Mia AI Lab](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) for its checkpoint and contributions, and [Christopher Owen](https://github.com/christopherowen/spark-ds41f), [@christopherowen](https://x.com/christopherowen), for networking, tuning inspiration and the benchmark methodology. All earlier [sources and credits](SOURCES.md) remain applicable to the historical vLLM recipes.

## Choose a recipe

| Recipe | Routed experts | Other rank tensors | Logical KV pool | Status |
|---|---|---|---:|---|
| [TF351](TF351.md) | Pollard EXL3 approximately 3.51 bpw | Mia EXL3/ancillary export | **10,000,000 tokens** | Current selected production profile; separate short pool stress and API qualification |
| [TF351-3.3M](TF351.md#the-frozen-benchmark-profile) | Same experts | Same export | 3,300,000 | Frozen comparative speed, prefix-cache and scoped quality profile |
| [TF29](TF29.md) | Mia EXL3 approximately 2.9 bpw | Mia export | 3,300,000 | Tested alternative; 10M on TF29 has not been tested |
| [DS41F 1.0.0 / C4](DS41F-1.0.0.md) | Pollard EXL3 approximately 3.51 bpw | Original hybrid checkpoint | 3,299,563 observed | Preserved vLLM recipe and independent rollback option |

All TF profiles keep **TP3, no EP, 401,408 tokens per request, eight streams, MAX reasoning**, round rows 16 and a separate 1 GiB prompt snapshot budget on the head. The 10M shared pool is 3.03 times the configured 3.3M pool, not a 10M per-request context or 30M across the cluster. Vision tensors remain BF16.

This is our **custom DeepSeek TP3 port**, not a claim that the current upstream installer supports this exact cluster. Use the source and kernel snapshots shipped here. Updating upstream TensorFold independently can change model, cache and API behavior.

## What changed from the vLLM path

The native Zig CUDA engine reads independent rank files and uses captured EXL3 and Triton CUDA kernels. Runtime serving does not use the Python model executor. The TP3 port pads physical vocabulary 129,280 to 129,408 and physical attention layout 64 heads/eight output groups to 72/nine. Copies preserve complete EXL3 rotation blocks, extra output contributions are suppressed, and both target and draft sampling mask nonlogical token IDs. This padding is an implementation layout, not a retrained model.

The recipe includes three-rank communication, CUDA graphs, mHC side work, L2 prefetch, asynchronous local Engram reads and speculative rounds. The cache repairs protect neighboring retained extents, retain complete prompt/accepted snapshots, permit checked short continuations and preserve DSML separators in tool-call history. Both `TF_DS_CACHE_SHORT_CONTINUATION=1` and `TF_DS_TOOL_SEPARATOR_ROUNDTRIP=1` are part of the tested profile.

The native compressed KV/index path uses FP4, while C4 uses `fp8_ds_mla`. TF351 retains the original routed experts but changes ancillary tensors, cache format, sampler and engine. Its performance or quality comparison with C4 is **not an engine-only or expert-bits-only experiment**.

## Frozen software and data

| Input | Pin |
|---|---|
| Published native runtime snapshot | `61357debe730ac9ffe20e760edcddbc253b0e525` |
| Final cache/speed candidate | `c4844adfef8df07f2bd299a6a23e5d04a6ec43c0` |
| Earlier full V3 speed/quality candidate | `6b7c3ac6df05e8041a4a77c26152723c795fad42` |
| Community DeepSeek Zig base | `6a1af414`, `bertholomus/TensorFold` DeepSeek branch |
| Python RoPE arithmetic reference | `8b1fe210d5e0d68e0cdb3b14755c135465c9ffa8`; ops SHA in `tensorfold/make_rope.py` |
| Pollard checkpoint | `bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard`, `f129e31a81e1337aa33e129e2d847fc7e37c8733` |
| Mia checkpoint | `Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw`, `64ba41b6c916a587db06eae2e19b7845f7be6e6b` |
| Zig | 0.17.0 |
| Qualified hosts | GB10 / SM121; kernel `7.0.0-1019-nvidia-64k`, 64K pages and the existing signed packing module |

The monitor extension after `c4844ad` adds observations; it does not change weights, numerical kernels, speculative scheduling or cache policy. The snapshot includes runtime source, build inputs, required export tools and license notices. Private operator controllers, source Git history and recordings are excluded. Runtime source file bytes are retained and individually hashed in `tensorfold/artifacts/source-files.json`. The `source_commit` records original provenance; it is not an instruction to fetch a private commit from upstream.

`tensorfold/inputs.json` binds the archives, frozen tensor-selection plans, native key order and public upstream manifests. `tensorfold/rope-hashes.json` binds every generated 3.3M/10M RoPE table. Kernel provenance builder paths were replaced with generic labels; kernel cubin bytes were preserved. The prebuilt kernels target SM121 and require a matching CUDA/driver environment; they are not portable kernels for arbitrary GPUs.

### 14 October short-prefill kit overlay

The kernel archive now includes six `_hc_mix_pf` row-4/5/6 additions and its 542-entry manifest. The published base runtime pin above is unchanged; only the expanded CPU checker source is overlaid from `5fa66fc`. Production serving binaries and weights were unchanged in that repair. [Exact inventory, provenance and tests](TENSORFOLD-HC-PREFILL-FIX.md). [Real Hermes production observations](HERMES-REAL-WORKLOAD.md) supplement the controlled results below and are labelled separately.

## Install and start

Follow [the complete preparation and launch procedure](TENSORFOLD-INSTALL.md). It includes source extraction, native head **and follower** build, pinned checkpoint verification, six rank exports, BF16 vision extraction, local Engram ownership, offline RoPE generation, topology and launch commands. Preparation is CPU/file work; it does not stop an existing deployment.

Print commands for all ranks before starting them. `tensorfold/launch.py` never replaces services or discovers private topology. The example has documentation addresses and deliberately incomplete NIC settings; supply your ordered four-rail devices, rank-specific peer masks, RoCE GID and NCCL interface/address range.

## Results and verification scope

Read [all TensorFold results and metric definitions](TENSORFOLD-RESULTS.md) and the [14 October measurement correction](BENCHMARK-CORRECTION-20261014.md). Final TF351 completed Hermes C1/C3 workflows in **15.207 / 35.601 s**, versus fresh C4 **22.965 / 39.003 s**. Its measured workflow request decode was **85.001 / 34.491 tok/s**, including reasoning and tools. Final Owen short code-prompt C1/C8 measured **84.470 / 396.030 tok/s**, prose-prompt **83.513 / 394.949 tok/s**; all those requests exhausted 256 tokens in reasoning and produced no final answer. Cold nominal 32K measured **2,147.643 input tok/s / HTTP TTFT**. These speed results used the frozen **3.3M** pool. The separately qualified 10M capacity is not a repeat of the full speed/quality campaign.

The recorded technical checks include all-rank communication, 22 linear probes per rank, 16 exact greedy serial/speculative pairs, sampled full-vocabulary requests, SSE/MAX and tool roundtrips. The short Q20 screen completed 20/24 attempts for C4 and TF351, 21/24 for TF29, with length/time caps accounting for incomplete attempts; it is not evidence that 2.9 bpw is generally better. Post-cache-fix tool deltas passed 3/3 on each TF arm and all four Hermes workflows completed correctly. Bounded partial-prefix replay can differ numerically from cold execution; historical strict mismatches remain failures.

The 10M stress test covered a 131,939-token cold request, eight concurrent roughly 48K inputs, short C1, direct vision, authenticated gateway chat/Responses/tool smoke and gateway vision. Minimum observed MemAvailable was 8.843/11.926/10.840 decimal GB per rank. It did not test eight 401K contexts, a cluster power cycle or unlimited snapshot retention. Newly rebuilt public artifacts still need operator acceptance on the destination cluster; publication itself does not constitute a new GPU run.
