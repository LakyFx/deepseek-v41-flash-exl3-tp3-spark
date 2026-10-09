# Agent installation guide

Read README.md, docs/INSTALL.md, docs/VERIFICATION.md, docs/ARCHITECTURE.md and docs/BENCHMARKS.md before executing commands. The only install target is X11c. Historical names in docs/HISTORY.md describe experiments; they are not ready-to-install independent engines.

Use config/x11c.json as the inference profile and config/cluster.local.json as operator-supplied topology. Do not change weights, trained rank, EP policy, KV layout or graph sizes to bypass a build or startup error. Check the exact checkpoint revision and all three nodes' image identity. Missing AOT objects, incompatible ABI or wrong Engram row ownership must be fixed before accepting traffic.

Keep deployment separate from preparation. Build images, download and verify weights, prepare local Engram and print every rank command before touching a running service. Ask the operator for the actual topology and maintenance window if they are missing. Never invent management IPs, interface names, GID indexes or authentication secrets. The example configuration is deliberately incomplete.

The launcher manages only its explicitly named local container and never replaces an existing deployment. Preserve the previous image, commands, weight files and independent caches for rollback. No global Docker cleanup, no automatic driver update, no automatic reboot, and no cross-version KV import.

Run benchmarks only against an explicitly closed, exclusive backend window. Publish metric scopes, output lengths, correctness, partial results and missing measurements. The public legacy surrogate inputs differ from the private historical corpus. Do not label those surrogate measurements as a paired replay of the historical numbers.

Keep host addresses, API keys, user prompts, generated private content and TLS material out of commits and benchmark exports. Runtime state belongs in operator-owned local configuration and result directories, not in this instruction file.
