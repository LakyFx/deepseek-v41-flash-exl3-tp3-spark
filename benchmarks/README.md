# Repeatable public workloads

`corpus/workloads` contains the sixteen reviewed source-reference and authored tool-task inputs used for the expanded workload suite. Decoded messages, tools and context text are unchanged. Files use ASCII JSON escaping so the repository prose has no em dash; the text passed to the model remains the original text. Published file hashes bind that encoding, while original request/context hashes remain available as provenance.

The original legacy corpus included captured private prose and synthetic code packets containing private operational source. Those complete bodies are excluded. Historical results, input token counts and opaque case fingerprints remain published. The public runner prepares safe legacy surrogate cases instead. A surrogate replay is not an exact paired comparison to the historical legacy results.

Read [the methodology](../docs/BENCHMARKS.md) before interpreting C1, C3, C8 or computed prefill. The public runner performs no service management and cannot establish an exclusive maintenance window by itself. Close ingress yourself, keep the backend on loopback, enable the narrow benchmark control lane, and supply its lease directory in local configuration. The lease is bound to the controller address and source port range, not an exposed vLLM developer API.

Preparation tokenizes the public surrogate inputs and records actual counts. It does not generate text. Measurement uses the same seven phase budgets as coverage20, with a 20-minute maximum, and records partial results. The separate full16 mode covers cold/hot code/prose at C1 through C4. None and medium modes are request-local and preserve production defaults.

```bash
# Backend ready, no generation: prepare the safe substitute legacy corpus.
python3 benchmarks/run.py --prepare-only --output local-results/public-legacy
# After manually closing all other ingress and enabling the control lane:
python3 benchmarks/run.py --exclusive --legacy local-results/public-legacy/corpus.json --reasoning max --output local-results/x11c-max
```

Use `--reasoning none` or `--reasoning medium` with a new output directory for another pass. Use `--profile legacy-full16` for the separate full16 matrix. After a successful run, the lease is retained with `window_closed: false`; move that completed receipt to your evidence archive before the next pass. On failure, the active lease and failed record remain, and the runner does not reopen ingress. Resolve pending requests and confirm idle before continuing.

Each preparation and measurement output is local and ignored by Git. Review a metric-only export before sharing results. Request bodies and generated text in local replay journals can contain your own private inputs if you supply a private corpus.
