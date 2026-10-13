# Frozen TensorFold-compatible benchmark kit

This is a separate copy of the tested Hermes/Owen adapters and Q20 fixture. It leaves the original C4 benchmark code unchanged. Read [the reproduction guide](../../docs/TENSORFOLD-BENCHMARK.md) and [complete results](../../docs/TENSORFOLD-RESULTS.md).

`SOURCE-HASHES.json` binds copied source and fixtures. `quality-q20/FROZEN-MANIFEST.json` records the pre-generation freeze; its two omitted preparation scripts were private offline fetch/freeze helpers, not runtime dependencies. The public Q20 fixture and selected source files are included unchanged. Public legacy surrogates are not the private historical conversation corpus.
