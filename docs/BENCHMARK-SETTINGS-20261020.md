# Exact model, quantization and sampling: 20 October 2026 workflow benchmark

These settings apply to **all three measured arms: original 1.2.2TF, Bert1
and 1.2.2xTF (now released as 1.2.3)**. This is a documentation addendum;
the benchmark was not rerun and no results were changed.

The [full Markdown report](WORKFLOW-BENCHMARK-20261020.md) contains all
measurements, limitations, the learner design and these exact settings
in one document. This page is the standalone settings reference;
the complete addendum is also included in the report's
[Appendix C](WORKFLOW-BENCHMARK-20261020.md#appendix-c--exact-model-quantization-and-sampling).

## Model and weight identity

| Item | Exact benchmark configuration |
|---|---|
| Base checkpoint | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) |
| Base revision | `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Quantized checkpoint | [`LakyFx/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-TensorFold`](https://huggingface.co/LakyFx/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-TensorFold/tree/8621801c1fc74aece1fa83e67c6ade3fde1e7894) |
| Weight revision | `8621801c1fc74aece1fa83e67c6ade3fde1e7894` |
| Main routed experts | EXL3 mixed allocation, approximately **3.5101 bpw** |
| Language-model head | EXL3 K6 |
| DSpark draft projections | EXL3 K4 |
| Engram tables | FP8 |
| Vision tower | BF16; no images were used in this benchmark |
| Runtime KV/index cache | TensorFold FP4 compressed KV/index |
| API model alias | `deepseek-v4.1-flash` |

**3.51 bpw describes the routed experts, not a whole-model average.** This
is a mixed-precision model. The pinned weight release and its
[SHA-256 manifest](https://huggingface.co/LakyFx/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-TensorFold/blob/8621801c1fc74aece1fa83e67c6ade3fde1e7894/RELEASE-MANIFEST.json)
identify the exact other tensors as well; matching only the nominal bpw
of a different quantization is insufficient for a like-for-like comparison.

Tokenizer/template SHA-256 values:

```text
tokenizer.json        c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b
tokenizer_config.json 6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547
chat_template.jinja   7d1356675e429ec4a55cbcc1a6fbeb40615dbeb257aca25b8c3705a42087f33e
```

## Sampling and request settings

| Parameter | Effective benchmark value | Where it came from |
|---|---|---|
| `temperature` | **0.6** | Explicit on every model call |
| `top_p` | **1.0** | Native HTTP default |
| `top_k` | **0**, disabled | Native HTTP default |
| `min_p` | **0.0**, disabled | Native HTTP default |
| `seed` | **42**, reused on every call | Explicit request value |
| Thinking / reasoning | Enabled, **MAX** | Request effort and template flags |
| `max_tokens` | **65,536 per call**, including reasoning | Request cap, not a fixed generation length |
| Request context limit | **401,408 tokens** | Native server configuration |
| `tool_choice` | `auto` | Explicit request value |
| Transport | Streaming SSE, `include_usage=true` | Benchmark client |
| Repetition/frequency/presence penalties | No overrides requested | Fields omitted |
| Custom stop / logit bias / response format / thinking budget | No overrides requested | Fields omitted |
| Workflow guard | 16 model turns; natural completion required | Benchmark protocol |

The following is the **sampling/control subset actually sent**, with the
task-specific messages, tool schemas and request ID omitted:

```json
{
  "model": "deepseek-v4.1-flash",
  "reasoning_effort": "max",
  "chat_template_kwargs": {
    "thinking": true,
    "enable_thinking": true,
    "reasoning_effort": "max"
  },
  "temperature": 0.6,
  "seed": 42,
  "max_tokens": 65536,
  "tool_choice": "auto",
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

`top_p`, `top_k` and `min_p` were omitted by the benchmark client. We
reconstructed their effective values from the frozen native HTTP
`resolveSampling` path, the three retained serving profiles (no sampler CLI
overrides), and the pinned wrapper inventory (no `generation_config.json`).
The HTTP resolver supplies **1.0 / 0 / 0.0**. Some internal sampling structs
have other initializer defaults; those do not determine these HTTP requests.
This is source/configuration evidence, not newly captured per-request sampler
telemetry. The [machine-readable settings](../results/20261020-adaptive/benchmark-settings.json)
include the relevant source hashes and distinguish sent fields from inherited
defaults.

For a comparison client, explicitly specify the inherited values as well:

```json
{"temperature": 0.6, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "seed": 42}
```

The upstream 1017 model card recommends temperature **1.0** and `top_p=1.0`.
**Our measurements used 0.6**; changing to 1.0 would be a different sampling
condition. [DeepSeek model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017#minimal-inference).

## Runtime and comparison boundaries

All arms used TensorFold native CUDA on **three NVIDIA DGX Sparks / GB10,
TP3, no expert parallelism**, GPU clock **2,200 MHz**, and DSpark speculation.
The benchmark called the native Chat Completions API directly, bypassing
the public gateway. Tasks and requested explanations were Czech. All settings
above stayed constant; [the per-arm table](WORKFLOW-BENCHMARK-20261020.md#configurations-and-what-changed)
lists the differing C/R limits, KV pool, snapshot cache and learner.

Compare the same concurrency, output-token scope and cache warmness as well
as model/sampler settings. C1–C12 counts simultaneous tool workflows; active
native requests vary while those workflows execute tools. The same seed does
not guarantee identical GPU outputs or later tool trajectories. This was one
matched-initial-input pass, not repeated trials or a single-knob causal test.
