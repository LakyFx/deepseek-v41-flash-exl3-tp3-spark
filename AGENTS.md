# Agent installation guide

Read README.md, docs/INSTALL.md, docs/VERIFICATION.md, docs/ARCHITECTURE.md and docs/BENCHMARKS.md before executing commands. Select the requested engine explicitly. For TensorFold read docs/TENSORFOLD-TP3.md, docs/TENSORFOLD-INSTALL.md, docs/TF351.md, docs/TF29.md and docs/TENSORFOLD-BENCHMARK.md. TF351 and TF29 are separate native profiles. The preserved vLLM target DS41F 1.0.0 is exactly C4; read docs/DS41F-1.0.0.md. Historical arms can be inspected/materialized as described in docs/EXPERIMENT-REPLAY.md; materialization is not a GPU qualification.

For C4 use config/ds41f-1.0.0.json and config/cluster.local.json. For TensorFold use tensorfold/profiles/{TF351,TF351-3.3M,TF29}.json and tensorfold/cluster.local.json. These engines do not share KV contents. Do not change weights, trained rank, EP policy, KV layout or graph sizes to bypass a build or startup error. Check the exact checkpoint revision and all three nodes' image identity. Missing AOT objects, incompatible ABI or wrong Engram row ownership must be fixed before accepting traffic.

Keep deployment separate from preparation. Build images, download and verify weights, prepare local Engram and print every rank command before touching a running service. Ask the operator for the actual topology and maintenance window if they are missing. Never invent management IPs, interface names, GID indexes or authentication secrets. The example configuration is deliberately incomplete.

The launcher manages only its explicitly named local container and never replaces an existing deployment. Preserve the previous image, commands, weight files and independent caches for rollback. No global Docker cleanup, no automatic driver update, no automatic reboot, and no cross-version KV import.

Run benchmarks only against an explicitly closed, exclusive backend window. Publish metric scopes, output lengths, correctness, partial results and missing measurements. The public legacy surrogate inputs differ from the private historical corpus. Do not label those surrogate measurements as a paired replay of the historical numbers.

Keep host addresses, API keys, user prompts, generated private content and TLS material out of commits and benchmark exports. Runtime state belongs in operator-owned local configuration and result directories, not in this instruction file.
