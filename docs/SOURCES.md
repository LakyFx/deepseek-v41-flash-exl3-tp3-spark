# Sources and credits

## 1.2.0 / DeepSeek 1017 addition

[DS41F 1.2.0](DS41F-1.2.0.md) uses our independent [EXL3 3.5bpw TensorFold export](https://huggingface.co/LakyFx/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-TensorFold), pinned on the Hub to `8621801c1fc74aece1fa83e67c6ade3fde1e7894`, from [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) at `1bae0e1088993b9b82de08f59de5be5a34e42170`. DeepSeek created the base model. All model weight values in that release come from the new source revision; older Pollard/Mia bytes are not reused.

Thank you to [TensorFold](https://tensorfold.dev/) and [Ashley Hart / @ashxhart](https://x.com/ashxhart) for the persistent, open work that inspired this export. The Pollard precision map and quantization work by bot-lab-21/WestWaters, Mia-AiLab's earlier EXL3 export and turboderp's ExLlamaV3 remain credited methods, tools and earlier integration sources. The source and measurement attributions below retain their historical scope; older quality measurements are not new 1017 results.

This deployment is assembled on the work of the following projects and authors. A source used in the runtime is distinct from a recipe studied for inspiration. Weight quality, upstream kernel microbenchmarks and our model-level measurements are also distinct claims.

## TensorFold TP3 recipes: primary thanks

**Thank you to [Ashley Hart / @ashxhart](https://x.com/ashxhart), the creator of [TensorFold](https://github.com/ashhart/TensorFold), [tensorfold.dev](https://tensorfold.dev/). TensorFold is the engine used by both TF351 and TF29.** Existing vLLM, kernel, networking, model and benchmark credits below remain applicable to the retained C4 recipe and the work that led here.

| Source | Role / pin |
|---|---|
| [TensorFold, ashxhart](https://github.com/ashhart/TensorFold) | Native Zig CUDA engine, EXL3 infrastructure, DeepSeek kernels and speculation; frozen TP3 source snapshot `61357debe730ac9ffe20e760edcddbc253b0e525` is supplied here, not fetched as an upstream commit |
| [bertholomus/TensorFold](https://github.com/bertholomus/TensorFold) | Community DeepSeek Zig base at `6a1af414`; our TP3 layout/cache patches extend it |
| [Mia EXL3 2.9 bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) | TF29 experts and both TF variants' ancillary tensors; pinned `64ba41b6c916a587db06eae2e19b7845f7be6e6b`; BF16 vision remains intact |
| [Pollard EXL3 checkpoint](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard) | TF351 routed experts retain the approximately 3.51 bpw bundles, revision `f129e31a81e1337aa33e129e2d847fc7e37c8733` |
| [Christopher Owen / @christopherowen](https://x.com/christopherowen), [spark-ds41f](https://github.com/christopherowen/spark-ds41f) | Benchmark source and earlier communication/tuning inspiration, as detailed below; our results are not Owen's own hardware measurements |
| [Google IFEval](https://github.com/google-research/google-research/tree/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval), [EvalPlus](https://github.com/evalplus/evalplus) | Frozen small public quality subsets, retained notices and upstream SHA manifests; no full leaderboard claim |

Native source and kernel packages retain `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES` and `LICENSES/`. The separate Python RoPE reference also carries the upstream TensorFold license. Our integration license does not relicense model weights, CUDA components or copied upstream work. See [TF source pins](TENSORFOLD-TP3.md#frozen-software-and-data).

The launch poster is generated artwork incorporating the TensorFold mesh logo from its public repository, credited to @ashxhart. It is a community recipe illustration; NVIDIA and TensorFold branding remains with its respective owners. Values and measurement scopes come from the linked result receipts. The current poster adds [real Hermes Agent production observations](HERMES-REAL-WORKLOAD.md), which are our own separate operational sample, not Owen's measurements. The [short-prefill AOT repair](TENSORFOLD-HC-PREFILL-FIX.md) compiles missing row-4/5/6 variants of the existing credited TensorFold arithmetic, preserving earlier sources and notices.

## Model and quantization

| Source | Contribution |
|---|---|
| [DeepSeek V4.1 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | Original model, architecture, tokenizer, native attention, Engram, vision and DSpark components |
| [bot-lab-21 Pollard EXL3 checkpoint](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard) | Our exact checkpoint, with mixed K3/K4 MUL1 routed experts at approximately 3.51 bpw; pinned to `f129e31a81e1337aa33e129e2d847fc7e37c8733` |
| [WestWaters pollard-weights](https://github.com/WestWaters/pollard-weights) | Quantization methodology explicitly credited by the checkpoint publisher |
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) | EXL3 trellis format and original implementations; related code retains its original license notices |

The EXL3 number describes routed experts. It is not the precision of every tensor in the hybrid checkpoint. We did not train the model or produce the historical Pollard/Mia quantizations described in this section; our separate 1017 export is identified above.

## Original serving recipe and software actually used

| Source | Pinned revision or version | Role |
|---|---|---|
| [tonyd2wild DeepSeek V4.1 Flash vLLM DGX Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark) | `fc725ecf10869c184f4347dd73336536d395753c` | Starting recipe, SM121 build, EXL3 TP3 compatibility, Engram on local disk, loading and serving patches; the original repository also credits Kai and other contributors |
| [vLLM](https://github.com/vllm-project/vllm) | Source `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba`; base nightly `8a728663c1c3eeace834a95f5654fa653cc1998c` | Inference engine, scheduler, prefix cache, distributed model execution and APIs |
| [Zeuss5 cuda-exl3](https://github.com/Zeuss5/cuda-exl3) | `6a1ffc34866e23f484574ce1922a8bca93eb33b2` | CUDA EXL3 decoding and expert kernels, vLLM integration; our small-row K kernel retains the original decoder and arithmetic contract |
| [Christopher Owen DGX Spark networking](https://github.com/christopherowen/dgx-spark-networking) | `b61660f2f6ad98a1d056f214b573d3ead3596978` | Sparknet one-shot RoCE collectives, proxy lifecycle and graph support; included source has compatibility and AOT integration changes |
| [local-inference-lab b12x](https://github.com/local-inference-lab/b12x) | `ba0902861cc19bd1a1d895e967299ff5a6ab2b55`, version 1.3.0 | B12X dense MXFP8 projections and related kernels; CuTe DSL 4.6.2 compatibility |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | `07869c61ba581e6d6b8ad8d142f4a6c89b707cc1` | GPU inference kernels and SM121 build inputs |
| [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass), [CCCL](https://github.com/NVIDIA/cccl), [NCCL](https://github.com/NVIDIA/nccl) | Build pins in `build/base/Dockerfile`; runtime NCCL 2.30.7 | CUDA templates, dependencies and bulk distributed communication |
| [PyTorch](https://github.com/pytorch/pytorch), [Transformers](https://github.com/huggingface/transformers), [Triton](https://github.com/triton-lang/triton), [TileLang](https://github.com/tile-ai/tilelang) | Exact observed versions in `config/dependency-versions.json` | Tensor runtime, tokenization and kernel compilation |

License notices supplied with copied source remain in the tree. Licensing the integration does not transfer ownership of these projects or the model. The build also uses dependencies such as spdlog, CUDA bindings and CUDA toolkit components, whose own licensing applies.

## Tuning ideas and studied alternatives

| Source | Relationship to this work |
|---|---|
| [Christopher Owen spark-ds41f](https://github.com/christopherowen/spark-ds41f), formerly [spark3-vllm-ds41f](https://github.com/christopherowen/spark3-vllm-ds41f) | Major inspiration for TP3 communication, B12X projections, row partitioning, shared draft work and adaptive verification. The repository moved; retain both links to make the history clear. Our checkpoint remains EXL3 rather than adopting that recipe's weight configuration. |
| [sfxnz two-Spark DeepSeek EXL3 recipe](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark) | Dense GEMV and fused MoE preparation source at `3de914682d0880c8e478712a60ada930339c22e4`, ported to our TP3 dimensions in the earlier B experiments. Imported code retains its MIT notice. These historical experiments did not change our checkpoint to that recipe's 2-bit pack. |
| [0xSero two-Spark DeepSeek recipe](https://github.com/0xSero/DeepSeek-V4.1-Flash-Two-Sparks) | Studied for DeepSeek prefill and EXL3 serving ideas. Its expert partitioning, checkpoint and KV geometry differ from ours. Its TP2 measurements are not our TP3 results. |
| [nacyot vLLM DS4F GB10](https://github.com/nacyot/vllm-ds4f-gb10) | Studied as an alternative for GB10 serving, cold-start throughput and long-session cache handling. Its experimental disk KV offloading is a separate proposal. The retained C4 recipe uses native GPU FP8 KV; TF recipes use native FP4 compressed KV/index. Local disk-backed Engram is separate from KV offloading. |
| [jakejharris jspark3](https://github.com/jakejharris/jspark3) | Studied as a three-Spark recipe for another model and as inspiration for reproducible deployment and performance work. Its reported GLM results are not attributed to DeepSeek. |
| [TensorFold](https://github.com/ashhart/TensorFold) | Engine used by the published TF351/TF29 native TP3 recipes. The original vLLM C4 installation remains independently available; see the TensorFold credits above. |
| [kindlingai GLM GX10](https://github.com/kindlingai/glm-5.3-flash-gx10), [jayleaton GLM TensorFold Spark](https://github.com/jayleaton/glm53-tensorfold-spark), [MiaAI Lab GLM TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) | Research and comparisons of alternative GLM serving recipes. They are not installed by this DeepSeek recipe. |
| [NousResearch Hermes agent](https://github.com/NousResearch/hermes-agent) | Main real agent workload motivating the selection of long context, reusable prefixes, tool behavior and concurrent work. Private Hermes conversations are excluded. |

We credit upstream contributors rather than presenting imported kernels or quantization as our invention. Our benchmark results measure our configuration and workload, not a controlled ranking of all these projects.

## DS41F 1.0.0 additions and benchmark attribution

The final C4 port retains the preceding credits and adds bounded decoder replay integration. The supporting Owen recipe snapshot is [`459a2e1e40607fcc3de3490050a00fc18f66f62b`](https://github.com/christopherowen/spark-ds41f/tree/459a2e1e40607fcc3de3490050a00fc18f66f62b); upstream vLLM file license headers remain intact. Our changes adapt this work to the existing EXL3 TP3 runtime and its cache/vision/speculation contracts. They are not presented as a newly invented attention algorithm.

**The second benchmark is taken from Christopher Owen's GitHub recipe**, specifically [`bin/spark` at `dd35633edeadd8864ec177890025b6e7c5df47ce`](https://github.com/christopherowen/spark-ds41f/blob/dd35633edeadd8864ec177890025b6e7c5df47ce/bin/spark). [OWEN-BENCHMARK.md](OWEN-BENCHMARK.md) documents exactly which functions, workload subset, repetitions and definitions were retained, and which transport/context adaptations were made. The whole upstream administration client is fetched by hash for local use rather than republished here.

The frozen cold-prefill fixture includes public CPython 3.12.3 standard-library excerpts; its license is retained in `licenses/python-stdlib.txt`. The host packing module is [dgx-spark-memory-saver](https://github.com/christopherowen/dgx-spark-memory-saver/tree/6f97319e7c00e5422b0afe4cabbd348211bfcb34), pinned at `6f97319e7c00e5422b0afe4cabbd348211bfcb34`. Machine-specific signing material is not distributed.


## 1.2.3 online speculative allocation and measurement context

The online cost/acceptance feedback and average-reward pricing were inspired
by Liquan Liu, Yifan Zhang and Bowei Xu, [AdaSpark: Adaptive DSpark with Online
Learning for Tree Verification and N-gram Fill](https://arxiv.org/abs/2610.05774),
and [imparo](https://github.com/zeraix/imparo). Our implementation uses the
existing TensorFold head/copy chain lanes, pinned/extrapolated priors, a rolling
median timing fit and bucketed acceptance calibration. It is an adaptation of
ideas, not a full implementation of that paper or engine. The paper's speedup
figures are not our measurement results.

[Our design and exact source](ADAPTIVE-SPECULATION.md) and the
[complete original/Bert1/xTF comparison](WORKFLOW-BENCHMARK-20261020.md)
distinguish recipe-level observations from causal claims. Standard speculative
inference background: [Leviathan, Kalman and Matias (2023)](https://proceedings.mlr.press/v202/leviathan23a.html).

## 1.2.4 / CDEF: DeepInfer and NVIDIA FLARE inspiration

A–C were informed by the studied [DeepSeek DeepInfer revision
774ce244](https://github.com/deepseek-ai/DeepInfer/tree/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01):
price actual verification shape, avoid unused sampling work and overlap host work.
D–G were informed by [NVIDIA FLARE revision
47581cca](https://github.com/NVIDIA/flare-local-inference/tree/47581ccac82660953a5b59cb04b2750d64edc97d):
bounded timing exploration, stale cost refresh, admission ordering and identical
Engram-read sharing. Our implementations adapt these ideas to the existing
TensorFold Zig host/lanes, retaining verifier/sampling contracts. They are not full
engine ports or reproductions of upstream performance claims.

CDEF contains A–F, while G is separate. [Exact boundaries](RECIPES.md) and
[our measurements](WORKFLOW-BENCHMARK-20261021.md) separate inspiration from
tested behavior. Existing source notices remain. The [comparison artwork](../assets/tensorfold-1.2.4-comparison.png)
is a community illustration; its values and scopes also appear in text.


## 1.2.7 prefix retention and runtime work

The activity-limited NVMe hypothesis and cache/graph correctness reviews were informed by [knapcio’s TP4 recipe](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4), including its [changelog](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/main/CHANGELOG.md) and [SWA incident analysis](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/main/docs/issue-16-swa-crash.md). Their TP4 results are not our TP3 measurements. Our bounded CPU checkpoint/prefix policies were developed for the existing TensorFold source and mixed agent workloads. The weights, verifier and retained TensorFold/upstream license notices are unchanged. The earlier AdaSpark, DeepInfer and NVIDIA FLARE inspirations remain described above; the new KV-retention controller is separate from the speculative verification controller.
