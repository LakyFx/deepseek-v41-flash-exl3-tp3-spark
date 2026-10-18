# Prepare and run the two TensorFold TP3 recipes

For the new **DeepSeek 1017 / 1.2.0** variant, follow [its separate recipe](DS41F-1.2.0.md). It reuses the source/build/kit procedures below but downloads the prebuilt 1017 model/rank/Engram artifacts instead of performing the older Pollard/Mia checkpoint preparation. The original procedures below remain available for TF351 and TF29.

Read [the recipe and qualification scope](TENSORFOLD-TP3.md) first. This procedure prepares independent files and gives explicit per-rank commands. It does not modify the old C4 image or stop a production service. Substitute your storage paths and topology.

## 1. Extract the pinned source and kernels

On a Linux build machine, from this repository:

```sh
python3 tensorfold/extract.py --out /srv/ds41f-tf-build
```

The extractor checks `tensorfold/inputs.json` before creating a new directory. It produces `source/`, `kit/` and `reference/`. The kit contains the qualified SM121 cubins, Triton AOT manifest, frozen model metadata, compressed Engram token map and fatbins used when building the native executable. RoPE binaries are generated separately; model and Engram weights are not distributed in this Git repository.

**14 October kit update:** extraction now includes all 542 AOT variants, including the six short-prefill `_hc_mix_pf` additions. The source snapshot also includes the expanded CPU prefill selector check. [Patch hashes, recorded acceptance and existing-kit migration](TENSORFOLD-HC-PREFILL-FIX.md). Use the same updated kit on all ranks; the matching pool RoPE generation below remains required.

## 2. Build the head and followers

Use Zig **0.17.0**, as recorded by `.zig-version`. On an x86 Linux build host, cross compilation can use:

```sh
cd /srv/ds41f-tf-build/source
zig build native dsv41-tools -j2 \
  -Dtarget=aarch64-linux-gnu \
  -Dfatbins=/srv/ds41f-tf-build/kit/build-fatbins \
  --prefix /srv/ds41f-tf-build/built
```

The head is `built/native/bin/tensorfold-native`. The `dsv41-tools` step also builds `tf-dsv41-lanes` and `tf-dsv41-check` under the install prefix. Locate the exact paths with `find /srv/ds41f-tf-build/built -type f`; install the head and follower executable into each destination's `bin/`. Building only `native` leaves follower ranks unavailable.

The qualified environment used GB10/SM121, the updated 64K NVIDIA kernel and the existing memory packing module. Follow [host-environment notes](HOST-ENVIRONMENT.md) and retain its source/versions. Serving additionally needs the CUDA driver, NCCL 2.30.7, libibverbs/RDMA and compatible shared libraries; supply `library_path` if outside system paths. Check `ldd` on the aarch64 destination. Compilation on another host is not a numerical/cluster acceptance test.

## 3. Download the two pinned source checkpoints

Use the official Hugging Face CLI or your existing immutable snapshots:

```sh
hf download bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard \
  --revision f129e31a81e1337aa33e129e2d847fc7e37c8733 \
  --local-dir /srv/ds41f-input/pollard
hf download Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw \
  --revision 64ba41b6c916a587db06eae2e19b7845f7be6e6b \
  --local-dir /srv/ds41f-input/mia
```

The source checkpoints and resulting rank files require substantial preparation storage. TF351 rank exports total 257.716 decimal GB and TF29 215.249 GB, before source snapshots, Engram and extra assets. Shared source storage can be read for CPU exports; runtime rank weights and the rank's owned Engram ranges belong on each Spark's local SSD.

## 4. Export both TP3 variants and preserve vision

From the repository root, first print the CPU preparation plan:

```sh
python3 tensorfold/prepare.py \
  --source /srv/ds41f-tf-build/source --kit /srv/ds41f-tf-build/kit \
  --pollard /srv/ds41f-input/pollard --mia /srv/ds41f-input/mia \
  --work /srv/ds41f-tf-preparation --out /srv/ds41f-tf-weights
```

Add `--write` to actually verify the pinned source files and export all six rank artifacts sequentially. Outputs must be new directories. The exporter verifies all index entries, representative scalar mappings and the full output SHA, and rejects changed source fingerprints. This is CPU/file validation, not a GPU quality verdict.

It also extracts the 266 required Mia BF16 vision/aligner tensors, compares every selected byte to the source and places `vision.safetensors`, metadata and its index under `weights/models/{TF351,TF29}`. A metadata-only model folder cannot serve vision. `.partial` hardlinks retained by the exporter are provenance, not duplicate physical allocation.

The native key order and tensor plans are frozen in `tensorfold/preparation/`. TF351 selects Pollard bundles for all 40 layers, 384 experts and three expert projections; the rest comes from Mia. The export padding works on complete EXL3 blocks and requires the logical-vocabulary masks in the supplied runtime.

## 5. Generate the matching RoPE kit

Preparation uses the supplied Python reference with **CPU PyTorch 2.13.0** (qualified generator) and its NumPy dependency. In a separate preparation environment, install the CPU torch wheel and dependencies; runtime serving itself needs no torch installation. Changing the arithmetic or tensor block sizes can change RoPE bytes, so the helper verifies against the qualified table hashes.

For current TF351:

```sh
python3 tensorfold/make_rope.py --reference /srv/ds41f-tf-build/reference \
  --kit /srv/ds41f-tf-build/kit --pool 10000000
```

For TF29 or the frozen TF351 speed profile, extract a **separate new build directory** and use `--pool 3300000`. Do not overwrite a kit with existing RoPE binary files. The 10M four-table payload is about 5.12 decimal GB; the 3.3M payload is about 1.69 GB. The helper checks all four final SHA values, not only their length.

All ranks in one run must use the same complete kit and pool. A 10M env setting with a 3.3M RoPE kit is invalid. The shared pool and per-request context are separate limits.

## 6. Prepare local Engram for each rank

The original table weights and scales remain external to the rank cache. From the pinned Pollard source directory, print a rank's ownership plan:

```sh
python3 tensorfold/prepare_engram.py --source /srv/ds41f-input/pollard \
  --kit /srv/ds41f-tf-build/kit --rank 0 --out /srv/ds41f-engram-r0
```

Add `--write` to copy that rank's eight of 24 hash columns for layers 1 and 14. Repeat with ranks 1 and 2 and separate output directories. The helper derives row offsets from the frozen `engram.json`, retains the original safetensors offsets/index and samples copied rows byte for byte. Store the corresponding sparse copy as `engram/` on its owning node. Unowned holes are expected; these are not full-file copies. Existing correctly verified C4 Engram copies can be reused only when their rank ownership, source revision, byte offsets and selected row bytes match.

## 7. Distribute independent runtime trees

Each node's configured runtime root contains:

```text
/srv/ds41f-tensorfold/
  bin/tensorfold-native
  bin/tf-dsv41-lanes
  kit/                          # same complete matching pool kit on all nodes
  engram/                       # this node's rank-owned rows and index
  weights/TF351/                # this node's TF351 rank file and receipts
  weights/TF29/                 # this node's TF29 rank file and receipts
  weights/models/TF351/         # config, tokenizer and BF16 vision file/index
  weights/models/TF29/
```

Keep exact exported filenames. They include the rank, world, variant and plan digest expected by the native loader. Verify transferred file sizes and SHA against exporter receipts. Two variants can coexist on disk; changing the variant is a coordinated stop/start of all three ranks, not a live weight toggle.

## 8. Topology, print and launch

Copy `tensorfold/cluster.example.json` to your untracked `tensorfold/cluster.local.json`. Fill the actual head/master address, management interface, four NCCL HCAs, fabric CIDR, four **ordered** RDMA devices per rank, GID index and three peer masks per rank. The example mask rotation `0,12,3` / `3,0,12` / `12,3,0` describes our ring ordering; it must correspond to your cabling, not merely your management rank order. Never copy a private cluster's NIC/GID configuration blindly.

Print all commands with `tensorfold/launch.py --config ... --variant TF351 --rank 0` and ranks 1/2. `TF351` chooses 10M; `TF351-3.3M` and `TF29` choose 3.3M. Add `--run` on each corresponding node only when weights, kit, Engram, driver and network are ready. The launcher refuses example master addresses, wrong model metadata and undersized RoPE tables. The head waits for both followers after loading; followers retry within the configured `TF_DS_START_TIMEOUT=1800`. Control uses master ports 29660 and 29661, plus the configured RoCE fabric.

The direct API binds loopback by default. The optional [production gateway](GATEWAY.md) supplies the same serving defaults and durable vision jobs used in our deployment; install it separately with your own credentials and backend URL. Test `/health` and `/v1/models`, wait for `warming=false`, then test chat, MAX, tools and vision before accepting agent traffic. MAX is a head setting; the gateway's public MEDIUM mapping and image-job policy are described in its guide.

The optional `tensorfold/tp3@.service` template depends on `network-online.target` and uses the same rank launcher. Adjust the repository path, place operator topology in `/etc/ds41f-tensorfold.json`, set `TF_PROFILE=TF351` in `/etc/ds41f-tensorfold.env`, and enable only the corresponding rank instance on each node. All three profiles must agree. Preserve old service definitions and images. The template has not been tested through a new three-machine power cycle during publication.

## Benchmark and rollback

The historical speed campaign used an exclusive API window, MAX, a 3.3M pool and the frozen Hermes/Owen corpus. Existing public benchmark clients are documented in [BENCHMARKS.md](BENCHMARKS.md) and [OWEN-BENCHMARK.md](OWEN-BENCHMARK.md); their old control adapter assumes vLLM and should not be used unmodified against native TensorFold. Follow [the TF benchmark adapter](TENSORFOLD-BENCHMARK.md), which selects native control and metrics. Private legacy conversation inputs remain excluded; new surrogate runs are not an exact historical replay.

Keep C4 weights/image/commands and independent TensorFold directories intact. Before switching, close ingress, stop only the selected three-rank services, start the chosen profile consistently, complete health/API checks, then reopen ingress. On failure, stop that profile as a group and restore the retained old three-rank launch and gateway configuration. Do not import KV contents across engines or variants. A cold restart is the reproducible fallback. The public preparation/launcher scripts do not reboot hosts, change Secure Boot or delete rollback artifacts.
