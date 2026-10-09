# Install the selected X11c profile

This is a three-node, ARM64, one-GB10-GPU-per-node installation. It uses TP3 and disk-backed Engram. The build and model preparation can take much longer than the benchmark. Prepare them before replacing an existing service. The launcher never stops an existing model or changes a system service.

## 1. Host requirements

Use three DGX Sparks with working NVIDIA drivers, Docker, NVIDIA Container Toolkit, and working RoCE v2 between every pair. CUDA 13 and SM121 support come from the pinned container build. Keep management traffic and RoCE address selection explicit. This recipe does not upgrade a host driver or network firmware.

The measured transport has four logical HCA names per node and two HCA lanes per peer. Check your actual devices with `ibv_devices`, `ibv_devinfo`, `ip -br address`, and `/sys/class/infiniband`. Do not infer a GID index from a device name. Confirm that the selected index is the IPv4 RoCE v2 entry on each selected HCA. The example names describe the measured device layout; verify your machine before retaining them.

Download and build on a node with enough free disk and RAM. The body is approximately 257 GB of tensor bytes, plus approximately 203 GB for the original Engram tables, metadata, source and build caches. Each rank retains only its own Engram rows, approximately one third of those tables, in sparse files. `du -h` shows actual allocation; `ls -lh` shows the larger logical sparse size. Keep additional disk headroom for image layers and download cache. Model files are not in this Git repository.

The weights can be shared read-only from one Spark to the other two over NFS. Every rank must see the same prepared body at its configured `body_path`; Engram and writable caches remain on each rank's local disk. Alternatively, replicate the body without changing its bytes. Sharing the body is a startup I/O choice, not an expert-parallel deployment.

## 2. Clone and build

```bash
git clone https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark.git
cd deepseek-v41-flash-exl3-tp3-spark
bash build/build.sh
```

The first image rebuilds the original pinned vLLM, FlashInfer and EXL3 stack. The second embeds all selected runtime overlays and rebuilds the small-row EXL3 extension, T01 extension, native Engram conversion, RoCE proxy, TP3 CuTe AOT objects and NCCL 2.30.7. There is no hidden dependency on a previous deployment directory. Python, Torch and CuTe versions are checked before the native build. A changed upstream nightly image that no longer satisfies this contract fails rather than silently becoming a different recipe.

Transfer the completed image to the other nodes with `docker save` and `docker load`, or push it to your own registry by digest. Build once and use the same image on all three nodes. Use a named tag or digest in the local configuration; do not substitute `latest`.

## 3. Download and prepare the checkpoint

Install `huggingface_hub` in a host-side virtual environment. Download uses two workers and the exact revision recorded in `config/checkpoint-files.json`. Accept the checkpoint's own license terms; do not treat our integration license as a model license.

```bash
python3 -m venv .venv
.venv/bin/pip install huggingface_hub
.venv/bin/python tools/model.py download --destination /srv/dsv41/checkpoint
python3 tools/model.py verify --source /srv/dsv41/checkpoint
python3 tools/model.py body --source /srv/dsv41/checkpoint --destination /srv/dsv41/body
```

Verification reads all downloaded files and checks pinned LFS SHA256 or Git blob identities. This costs an additional full disk read. The body tool excludes the Engram table shards from the serving index and applies the observed TP3 configuration. On one filesystem it hard-links the unchanged body shards, so preparation does not duplicate the body. Across filesystems it copies them. It refuses to overwrite an existing destination. Virtual-head padding is described in [ARCHITECTURE.md](ARCHITECTURE.md).

On each node, prepare its own Engram rows from the original complete checkpoint directory, available locally or through a temporary read-only share:

```bash
# Node with TP rank 0
python3 tools/model.py engram --source /srv/dsv41/checkpoint --destination /srv/dsv41/engram-rank0 --rank 0
# Run the corresponding command on rank 1 and then rank 2, with --rank 1 or 2.
```

The copied ranges follow Engram head buckets, not a naive equal division of table rows. The helper preserves safetensors offsets and verifies boundary and randomly selected rows against the original bytes. A failed verification must be resolved before starting a rank. Do not move sparse files with a tool that expands holes into allocated zeros.

## 4. Configure the topology

```bash
cp config/cluster.example.json config/cluster.local.json
```

Edit every `REPLACE_...` value. `control_address` is the address used for Gloo and rendezvous; `control_interface` is the corresponding interface. `roce_cidr` selects the actual RoCE IPv4 addresses for NCCL. `peer_hcas` maps each other TP rank to the two local HCA lanes that reach that peer directly. The mapping differs by rank. Keep the selected `gid_index` consistent with the verified hardware entries.

Use identical configuration on all nodes, with node-specific paths in each row. Keep `benchmark_control.enabled` false for regular serving. Keep the backend on loopback and expose only an authenticated gateway or SSH tunnel to a trusted client. The distributed workers still use the non-loopback control addresses. Do not expose the backend's abort and benchmark endpoints to the Internet.

## 5. Start workers, then head

Each command runs in the foreground on its own host. Use separate terminals or your own service manager. Run rank 1 and rank 2 first, then rank 0:

```bash
python3 tools/launch.py --config config/cluster.local.json --rank 1 --run
python3 tools/launch.py --config config/cluster.local.json --rank 2 --run
python3 tools/launch.py --config config/cluster.local.json --rank 0 --run
```

Omitting `--run` prints the command. The launcher reads the built AOT manifest digest from the image, validates local Engram ownership and the TP3 model configuration, and creates only this recipe's writable directories. Missing local model paths, a wrong Engram rank, missing AOT objects or incompatible dependencies are errors. Do not work around such errors by enabling a different backend or changing the weight pack.

Startup includes kernel warmup and CUDA graph capture. Do not admit production requests before `/health` returns 200 and all rank logs confirm the selected version, Engram ranges, graph capture and KV capacity. The observed deployment had 3,299,563 KV tokens; that figure is a measured configuration result, not a guarantee under a different allocator or runtime. Verify your startup logs before advertising the pool size.

```bash
curl --fail http://127.0.0.1:8003/health
curl --fail http://127.0.0.1:8003/v1/models
```

## 6. Reasoning policy and clients

The model defaults to thinking enabled and native `max`. Our production gateway forces main agent requests to MAX and image requests to MEDIUM, mapped to native `high` strength 50. For a direct vision request set `reasoning_effort: "high"` at both the top level and in `chat_template_kwargs`, with thinking enabled. The native encoder rejects the literal `medium`. A top-level reasoning field overrides a conflicting template field.

The model-side serial renderer and one-image encoder batches are included. The separate private multi-protocol gateway, HTTPS certificates, user accounts and WhatsApp retry/cursor state are not copied. A direct model connection allows request-level effort overrides. If you require the same forced policy, use [gateway.py](../tools/gateway.py), which provides an OpenAI chat forwarding lane, before connecting Hermes. For Responses or other adapters, preserve the same policy in the adapter and verify the final model-bound body.

Run the forwarding lane on rank 0 in a host environment with `httpx`, `starlette` and `uvicorn`, or inside the built image with host networking and this script mounted read-only. Supply an API key through the environment; do not put it in the repository. It defaults to loopback port 8000. A trusted reverse proxy can terminate TLS and forward to that port. Its configurable upstream read timeout defaults to 1,200 seconds; this does not override a shorter Hermes client timeout or add queue priority.

```bash
# RECIPE_API_KEY must already be set in the process environment.
python3 tools/gateway.py
```

## Rollback and durable operation

Keep the old image, source, launch command, model directory and its independent caches intact until the new installation is accepted. The new recipe uses independent container names and state paths. Stop only these explicitly named X11c containers before restoring your previous commands; do not use global Docker cleanup or replace a running service in place. Keep the topology and original model byte checksums with your rollback record.

No KV cache dump/import is part of this recipe. A return to a different cache layout starts cold. Never import snapshots between versions merely because their advertised token capacity matches. Startup and routing checks must pass again before returning traffic to the restored system.
