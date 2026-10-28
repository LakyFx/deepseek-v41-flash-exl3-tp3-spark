# Exact measured native recipes

**Current: [1.2.7 / X4](../../docs/DS41F-1.2.7.md)**, recipe `v127`, selector `1.2.7`. `x1` / `x2` / `x3` / `x4` preserve every deployed trial independently. X4 and v127 have identical source maps and reference executable hashes; only release metadata differs. [Development and commands](../../docs/X1-X4-DEVELOPMENT.md). The new results are live cache observations; previous decode benchmarks remain assigned to their original versions.

**Historical: [1.2.5 / K2a](../../docs/DS41F-1.2.5.md)**, recipe `k2a`, selector `1.2.5`. Its exact 1,137-file source map includes the native vision-control repair. Also prepare the separate head-only direct-output template. CDEF remains preserved as 1.2.4.

[Human/agent recipe chooser](../../docs/RECIPES.md) · [build/install 1.2.4](../../docs/DS41F-1.2.4.md).

`base.json` binds the full frozen source archive. Each `recipes/*.json` defines
its exact final source map and SHA256-addressed complete file overlays under
`objects/`. Files not overlaid come unchanged from the existing source archive.
These are published source, not opaque executable downloads. CDEF is **1.2.4**.

Fresh Linux extraction → exact recipe application → source verification → CPU
checks → native head/follower build → independent install → coordinated rank
qualification. [patch.py](patch.py) and [build.py](build.py) implement preparation;
neither manages services or submits inference. The existing launcher validates
the matching build receipt before execution.

New-campaign manifests include recorded CPU/build check status and measured
executable identities. Historical original/Bert1 use the frozen original native
source plus their own profiles; those manifests do not invent missing historical
build receipts. Rebuilding source on another host requires local qualification.
`source_tree_sha256` hashes compact sorted JSON of `source_files`, not archive
packing timestamps. Exact object bytes are preserved across Git checkouts.

The original source archive and its upstream license context remain intact.
[Attribution](../../docs/SOURCES.md) distinguishes TensorFold code, our runtime
adaptations and ideas studied in DeepInfer, NVIDIA FLARE and AdaSpark.
