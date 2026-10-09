# Third-party code and attribution

Our integration license does not relicense upstream source, libraries or weights. Read [SOURCES.md](docs/SOURCES.md) for project links, pinned revisions, the original recipe, named contributors and studied alternatives.

| Included source | Origin and applicable notices |
|---|---|
| `build/base/patch`, base build helpers and Engram sparse-copy helper | tonyd2wild recipe and its Tech2Wild/Kai contributions; `licenses/tonyd2wild-MIT.txt`; underlying vLLM-derived files also retain vLLM licensing |
| Modified vLLM Python in `runtime/rootfs/.../vllm` | vLLM, Apache 2.0, `licenses/vllm-Apache-2.0.txt`; our and original recipe changes are identified in source and provenance |
| `native/cuda-exl3`, T01 decoder headers and CUDA integration | Zeuss5 cuda-exl3, MIT, `native/cuda-exl3/LICENSE`; contains ExLlamaV3 primitives credited to turboderp; `licenses/exllamav3-MIT.txt` |
| `runtime/rootfs/.../v089_runtime/cooperative_source` | ExLlamaV3, original MIT notice included beside the source; historical rejected cooperative path is not enabled by X11c |
| Historical dense and MoE preparation code in `v089_runtime` | sfxnz recipe, MIT, original `LICENSE-sfxnz` retained; TP3 integration is ours |
| B12X vendor package | local-inference-lab, Apache 2.0, `licenses/b12x.txt`; original source headers retained |
| Sparknet vendor package | Christopher Owen dgx-spark-networking; original vendor `LICENSE` and `NOTICE` retained, including its RoCEnante, GPUNetIO, NCCL patch, SparkRing and Polycom references |
| CUDA templates and runtime build dependencies | NVIDIA CUTLASS, CCCL, NCCL, CuTe DSL and CUDA bindings/toolkit; original dependency source and binary notices apply; CUTLASS notice copied to `licenses/cutlass-BSD-3-Clause.txt` |
| Other build and runtime dependencies | PyTorch, FlashInfer, Triton, TileLang, Transformers, safetensors, spdlog and their dependencies retain their own licenses |
| Model, tokenizer, quantization and checkpoint documents | DeepSeek and the bot-lab-21 Pollard checkpoint publisher; licenses and credits are downloaded with the exact checkpoint, not replaced by our integration license |

The vendored source is included for reproducible installation and review. Selective ports, source hashes, publication adjustments and measured settings are recorded under `runtime/` and `config/`. Research-only links do not imply that their engine or weight pack is installed.
