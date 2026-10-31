# Select TP3 or TP4 explicitly

TP3 **1.2.8 / AF-4K** and TP4 **4.0.0 / P3** are separate supported recipes.
For TP4 read `docs/DS41F-4.0.0.md`, `docs/BENCHMARK-4.0.0.md` and
`tensorfold/tp4/profile.json`. Source is `v400`; launch with
`python3 -m tensorfold.tp4.launch` on four explicitly ordered ranks. Preserve
C16/R48/9M/3.5 GiB, 8K compute, 2K KV quantum, 401408 context, the X4 retention
and speculation learners, 320 MiB credit, two-way owned Engram and final TP4
AOT/kernel assets. Use the separately patched pinned NCCL library. Never
select the TP3 launcher for TP4 or treat the published TP3 weight shards as
native TP4 caches. Use the byte-exact repartition tools and readback receipts.
Historical P2 cold prefill and final P3 workflow are separate scopes.
Publication does not authorize a live restart or a new benchmark.

All preserved TP3 and historical instructions follow.

# Current TP3 release: 1.2.8 / AF-4K

Read `docs/DS41F-1.2.8.md` and `docs/PREFILL-1.2.8.md`. Build `--recipe v128`,
install/launch `--variant 1.2.8` on all three ranks. Preserve C12/R48/7M/3 GiB,
adaptive speculation and X4 smart retention. Compute chunks are 4096, physical
KV quantum 2048, context 401408 and prompt workspace 403456. Followers require
`--prompt-context 401408`; use the launcher. Include the isolated pinned 1.2.8
AOT/auxiliary assets from the builder; never launch AF with the old AOT manifest.
Keep NVMe companion and MAX/8K vision policy. Reference qualification is cold
prefill and full-window correctness; do not relabel historical decode as 1.2.8.
Publication does not authorize changing a live installation. Preserve old
recipes, results, licenses and operator-owned runtime state.

## Preserved installation guidance

# Agent installation guide

Previous TP3 production is **1.2.7**, the unchanged deployed X4 payload. Read `docs/DS41F-1.2.7.md`, `docs/X1-X4-DEVELOPMENT.md` and `docs/LIVE-CACHE-1.2.7.md`. Build `--recipe v127`, launch `--variant 1.2.7` on all three ranks, and prepare the matching head-only template. Include the activity-limited NVMe companion on every rank. Preserve C12/R48/7M/3 GiB/chunk2048, the existing speculation learner, 320 MiB per-rank copy credit and MAX/8K vision. X1-X4 are historical selectors, not different weights. The new flyer uses live continuation results; +67% is modeled, not a measured decode gain. Never commit prompts, responses, job/session identifiers or credentials.

## Preserved 1.2.6 installation guidance

Previous production was **1.2.6**, measured as 5c. Read `docs/DS41F-1.2.6.md` and
`docs/SPARKDASH-20261024-1.2.6.md`. Build `--recipe v126`; launch `--variant 1.2.6`
on all three ranks. Prepare the head overlay with `--variant 1.2.6`. Preserve
7M KV/3 GiB snapshots/C12/R48/chunk2048 and the existing adaptive learner.
The native planner/copy credit is 320 MiB per rank. Standalone vision uses MAX,
8192 thinking and 32768 final tokens; other agent image histories keep their own
budgets. Private diagnostic state never belongs in commits. Both new posters
use only SparkDash run 2 (thinking OFF), not historical heavy workflow/RigMark.

The previous production recipe is **1.2.5 / K2a**. Read `docs/DS41F-1.2.5.md`,
build with `--recipe k2a` and launch with `--variant 1.2.5`. Prepare the exact
direct-output template overlay on the head only; followers retain the original
1017 template. All three native ranks require matching source/build receipts.
The optional updated gateway requires native vision-control receipts and keeps
two image requests in flight, finite deadlines and 32K final-answer headroom.
SparkDash thinking-OFF tests, MAX workflow and MAX RigMark are separate protocols.
Never relabel historical 1.2.4 gains as measurements of 1.2.5.

The previous main native recipe is **1.2.4 / CDEF = A+B+C+D+E+F**, DeepSeek 1017,
C12/R48, 7M configured KV and 3 GiB snapshots. Read `docs/DS41F-1.2.4.md`,
`docs/RECIPES.md` and `docs/WORKFLOW-BENCHMARK-20261021.md`. Use `--variant 1.2.4`
with the exact `cdef` source recipe and its independent `bin-1.2.4/BUILD.json`.
All three ranks must match; the launcher verifies source/receipt/binary identity.
Historical `1.2.3`, originally xTF, keeps its original separate adaptive builder.
Preserve older profiles, A–G/CF/DEF recipes and evidence. Native request decode,
wave aggregate and host-event latency have different scopes. The new public
`benchmarks/workflow` retains all 78 frozen initial body hashes; the older legacy
surrogate benchmark remains a separate measurement corpus.

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
