# Sources and credits

This deployment is assembled on the work of the following projects and authors. A source used in the runtime is distinct from a recipe studied for inspiration. Weight quality, upstream kernel microbenchmarks and our model-level measurements are also distinct claims.

## Model and quantization

| Source | Contribution |
|---|---|
| [DeepSeek V4.1 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | Original model, architecture, tokenizer, native attention, Engram, vision and DSpark components |
| [bot-lab-21 Pollard EXL3 checkpoint](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard) | Our exact checkpoint, with mixed K3/K4 MUL1 routed experts at approximately 3.51 bpw; pinned to `f129e31a81e1337aa33e129e2d847fc7e37c8733` |
| [WestWaters pollard-weights](https://github.com/WestWaters/pollard-weights) | Quantization methodology explicitly credited by the checkpoint publisher |
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) | EXL3 trellis format and original implementations; related code retains its original license notices |

The EXL3 number describes routed experts. It is not the precision of every tensor in the hybrid checkpoint. We did not train the model or produce this quantization.

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
| [jakejharris jspark3](https://github.com/jakejharris/jspark3) | Studied as a three-Spark recipe for another model and as inspiration for reproducible deployment and performance work. Its reported GLM results are not attributed to DeepSeek. |
| [TensorFold](https://github.com/ashhart/TensorFold) | Investigated as an alternative engine and source of kernel and speculation ideas. Current production and this installation target use vLLM. |
| [kindlingai GLM GX10](https://github.com/kindlingai/glm-5.3-flash-gx10), [jayleaton GLM TensorFold Spark](https://github.com/jayleaton/glm53-tensorfold-spark), [MiaAI Lab GLM TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) | Research and comparisons of alternative GLM serving recipes. They are not installed by this DeepSeek recipe. |
| [NousResearch Hermes agent](https://github.com/NousResearch/hermes-agent) | Main real agent workload motivating the selection of long context, reusable prefixes, tool behavior and concurrent work. Private Hermes conversations are excluded. |

We credit upstream contributors rather than presenting imported kernels or quantization as our invention. Our benchmark results measure our configuration and workload, not a controlled ranking of all these projects.
