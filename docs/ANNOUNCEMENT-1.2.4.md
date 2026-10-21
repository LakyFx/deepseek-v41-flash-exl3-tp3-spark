# DeepSeek V4.1 Flash: more work from three DGX Sparks

Our **1.2.4 TensorFold TP3 recipe** is published: DeepSeek-V4.1-Flash-1017,
EXL3 **~3.51 bpw routed experts**, **MAX reasoning**, on **3× DGX Spark**.

The focus is **smart adaptive speculation**. The controller learns actual
verification cost and draft acceptance, then adjusts how it spends the shared
speculative budget as load changes. Version 1.2.4 adds:

- **Better cost estimates:** price the verification graph actually executed and skip sampling rows the current path cannot use.
- **Less waiting:** prepare CPU penalty scores while the GPU works.
- **Controlled adaptation:** bound timing-probe overhead and refresh stale cost estimates.
- **Fairer scheduling:** give existing streams a decode turn between new admissions.

Against 1.2.2TF in our **heavy agent workflow**—real tools, code execution,
tests and repair rounds—we observed:

- **282 tok/s sampled aggregate decode peak at C11.**
- **16.8% more correct tasks/minute; 14.3% less total workflow time.**
- **78/78 tasks completed correctly in both versions.**

Same hardware, pinned weights and sampler; one pass per version. Tasks are
authored fixtures; output lengths and repair paths vary. The peak is a roughly
two-second rolling observation, not sustained workflow throughput.

All **14 measured recipes**, C1–C12 results, source/build instructions and
numeric evidence are public. Runtime feedback adjusts work allocation;
it does not retrain model weights.

Built on Ashley Hart and contributors’ **TensorFold**, with inspiration from
**AdaSpark, DeepInfer and NVIDIA FLARE**.

**Recipe, results and source:** https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark

**Weights:** https://huggingface.co/LakyFx/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-TensorFold

#DeepSeek #TensorFold #DGXSpark #LocalLLM #SpeculativeDecoding #LLMInference
