# Exact TP4 4.0.0 payloads

`native-linux-arm64.tar.gz` contains the exact measured P2 executables used by
P3/4.0.0. Full source is reconstructed by `v400`; use `--source-build` for an
optional rebuild with Zig 0.17.0. Rebuilds need destination qualification.

`kernels.tar.gz` contains only the missing/changed TP4 modules, final AOT
manifest and P2 prompt MMA auxiliary cubin. `MANIFEST.json` binds these bytes
and all reused modules from the immutable base kernel archive. The final
manifest has 591 entries, including the required decode/short-tail coverage.
Historical module names may be Triton cache IDs; actual file SHA-256 values
are authoritative. `assets.py` checks the complete set, with no JIT fallback.

`tp4_mma_rot.cu` is the exact P2 CUDA auxiliary source. The native source
contains launch/layout code. `tp4_aot_kernels.py`, `AOT-SOURCE.json` and
`aot_tp4_main.zig` preserve upstream kernel body provenance and the initial
CPU launch-matrix helper. That helper alone is not the final decode coverage
check or a standalone generator. The shipped final AOT JSON records launch
specializations. Use the verified payload for the measured recipe; do not
substitute the TP3 AF manifest.

Upstream TensorFold, Triton, ExLlamaV3, CUDA and model licenses remain in the
base kit/source. Our source additions do not relicense upstream code.

[Complete recipe](../../../docs/DS41F-4.0.0.md).
