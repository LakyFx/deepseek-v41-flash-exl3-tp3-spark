# AF / 1.2.8 prefill kernels

The archive contains the exact measured merged AOT manifest, 19 new cubins
(24 masked-attention launch entries), and the EXL3 prompt MoE auxiliary cubin.
All original AOT entries are retained; original cubins come from the existing
pinned kernel archive. `MANIFEST.json` binds every byte. `prefill_assets.py`
combines them in an isolated directory at build time, never overwriting the
base runtime kit. The launcher revalidates the installed set at startup.

`kernels.py` preserves the attention source adapted from Ashhart's TensorFold
with explicit TP3 head-tail masks. `experts_ragged.cu` is the exact measured W
CUDA source, including its ExLlamaV3/Turboderp notice. The original licenses
remain in the source archive and repository notices. See [credits](../../docs/SOURCES.md#128-tp3-prefill-work).

The measured auxiliary build used:

```sh
nvcc --cubin -arch=sm_121 -O3 -std=c++17 --fmad=true --ftz=false \
  --prec-div=true --prec-sqrt=true -Xptxas=-v experts_ragged.cu -o experts_ragged.cubin
```

Attention used Triton 3.8.0, `GPUTarget('cuda',121,32)`, four warps, one stage,
no global scratch, HB16 with masked H24 tails and original PDL specialization.
`COMPILE.json` records source and compiled hashes. The shipped compiled cubins
are the reference assets; a different compiler must be qualified independently.
