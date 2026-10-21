# Exact measured native recipes

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
