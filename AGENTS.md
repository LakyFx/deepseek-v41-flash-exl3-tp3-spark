# Agent installation guide

The integrated production release is **1.2.3**, the same configuration measured
as **1.2.2xTF**: DeepSeek 1017, C12/R48, 7M KV, 3 GiB prompt-state cache and
online cost/acceptance learning. Read `docs/DS41F-1.2.3.md`,
`docs/ADAPTIVE-SPECULATION.md` and `docs/WORKFLOW-BENCHMARK-20261020.md`.
Use `--variant 1.2.3` and its separately built `bin-1.2.3/` binaries/receipt;
the old engine plus a new environment flag is not the released adaptive build.
The gateway keeps two queued vision slots. Preserve older profiles and evidence.

For the optional production serving layer read `docs/GATEWAY.md`, install
`./gateway` in its own virtual environment and use `ds41f-gateway`. The package
matches the published production settings, requires an operator-supplied API key
and keeps durable vision jobs in operator-owned private state. Model preparation
does not install or restart this service. Preserve the legacy engine recipes.

For the additive **DS41F 1.2.0 / DeepSeek 1017** recipe, read `docs/DS41F-1.2.0.md` and use `tensorfold/profiles/1.2.0.json` with `--variant 1.2.0`. Download the pinned prebuilt rank/Engram/wrapper files from the linked LakyFx Hugging Face release; do not run the older Pollard/Mia weight preparation for this variant. Keep earlier profiles and historical test results intact. Historical benchmarks are not measurements of 1017.

Read README.md, docs/INSTALL.md, docs/VERIFICATION.md, docs/ARCHITECTURE.md and docs/BENCHMARKS.md before executing commands. Select the requested engine explicitly. For TensorFold read docs/TENSORFOLD-TP3.md, docs/TENSORFOLD-INSTALL.md, docs/TF351.md, docs/TF29.md and docs/TENSORFOLD-BENCHMARK.md. TF351 and TF29 are separate native profiles. The preserved vLLM target DS41F 1.0.0 is exactly C4; read docs/DS41F-1.0.0.md. Historical arms can be inspected/materialized as described in docs/EXPERIMENT-REPLAY.md; materialization is not a GPU qualification.

For C4 use config/ds41f-1.0.0.json and config/cluster.local.json. For TensorFold use tensorfold/profiles/{TF351,TF351-3.3M,TF29}.json and tensorfold/cluster.local.json. These engines do not share KV contents. Do not change weights, trained rank, EP policy, KV layout or graph sizes to bypass a build or startup error. Check the exact checkpoint revision and all three nodes' image identity. Missing AOT objects, incompatible ABI or wrong Engram row ownership must be fixed before accepting traffic.

Keep deployment separate from preparation. Build images, download and verify weights, prepare local Engram and print every rank command before touching a running service. Ask the operator for the actual topology and maintenance window if they are missing. Never invent management IPs, interface names, GID indexes or authentication secrets. The example configuration is deliberately incomplete.

The launcher manages only its explicitly named local container and never replaces an existing deployment. Preserve the previous image, commands, weight files and independent caches for rollback. No global Docker cleanup, no automatic driver update, no automatic reboot, and no cross-version KV import.

Run benchmarks only against an explicitly closed, exclusive backend window. Publish metric scopes, output lengths, correctness, partial results and missing measurements. The public legacy surrogate inputs differ from the private historical corpus. Do not label those surrogate measurements as a paired replay of the historical numbers.

Keep host addresses, API keys, user prompts, generated private content and TLS material out of commits and benchmark exports. Runtime state belongs in operator-owned local configuration and result directories, not in this instruction file.
