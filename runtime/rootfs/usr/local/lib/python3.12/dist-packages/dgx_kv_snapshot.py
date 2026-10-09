"""Opt-in, quiescent prefix-pool checkpoints for the pinned vLLM runtime.

No pickle, no remote API, no background GPU access. Engine-thread coordinator;
bounded byte copies on workers. This is NOT a checkpoint of in-flight requests.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid

FORMAT = 1
CHUNK = 16 * 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    with open(temp, "x", encoding="utf-8") as f:
        os.chmod(temp, 0o600)
        json.dump(value, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    if os.name == "posix":
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def generation_dir(root, generation):
    require(bool(re.fullmatch(r"[a-f0-9]{32}", generation)), "Invalid snapshot id")
    root = Path(root).resolve()
    path = root / generation
    require(not path.is_symlink(), "Snapshot directory cannot be a symlink")
    return path


def pool_export(pool):
    require(pool.enable_caching, "Prefix caching disabled")
    require(all(b.ref_cnt == 0 for b in pool.blocks if not b.is_null), "Referenced blocks remain")
    free = [b.block_id for b in pool.free_block_queue.get_all_free_blocks()]
    records = []
    for b in pool.blocks:
        aliases = sorted(x.hex() for x in pool.cached_block_hashes_by_block.get(b.block_id, ()))
        if b.block_hash is not None:
            records.append([b.block_id, b.block_hash.hex(), b.block_hash_num_tokens, aliases])
        else:
            require(not aliases, "Aliases without primary hash")
    data = {"blocks": len(pool.blocks), "null": pool.null_block.block_id,
            "hash_block_size": pool.hash_block_size, "free": free, "cached": records}
    pool_validate(pool, data)
    # Check both representations, including duplicate hashes and partial aliases.
    expected = {(bytes.fromhex(h), bid) for bid, h, _, aliases in records for h in [h, *aliases]}
    actual = set()
    for h, value in pool.cached_block_hash_to_block._cache.items():
        for b in (value.values() if isinstance(value, dict) else [value]):
            actual.add((h, b.block_id))
    require(actual == expected, "Hash index inconsistent with block metadata")
    return data


def pool_validate(pool, data):
    require(data["blocks"] == len(pool.blocks), "Pool size changed")
    require(data["null"] == pool.null_block.block_id, "Null block changed")
    require(data["hash_block_size"] == pool.hash_block_size, "Hash block size changed")
    expected = set(range(len(pool.blocks))) - {pool.null_block.block_id}
    require(len(data["free"]) == len(expected) and set(data["free"]) == expected,
            "Free queue does not cover exactly all unused blocks")
    seen = set()
    for bid, h, count, aliases in data["cached"]:
        require(bid in expected and bid not in seen, "Duplicate/invalid cached block")
        seen.add(bid)
        require(count is None or (type(count) is int and count > 0), "Invalid token count")
        hashes = [h, *aliases]
        require(len(hashes) == len(set(hashes)), "Duplicate hash alias")
        for value in hashes:
            require(len(bytes.fromhex(value)) > 4, "Invalid block/group hash")


def pool_restore(pool, data):
    from vllm.v1.core.block_pool import BlockHashToBlockMap
    from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue
    pool_validate(pool, data)
    require(not pool.cached_block_hash_to_block._cache, "Restore requires an empty hash index")
    require(all(b.ref_cnt == 0 for b in pool.blocks if not b.is_null), "Pool not idle")
    pool.cached_block_hash_to_block = BlockHashToBlockMap()
    pool.cached_block_hashes_by_block.clear()
    for b in pool.blocks:
        b.reset_hash()
        b.prev_free_block = b.next_free_block = None
    pool.free_block_queue = FreeKVCacheBlockQueue([pool.blocks[i] for i in data["free"]])
    for bid, h, count, aliases in data["cached"]:
        b = pool.blocks[bid]
        pool._insert_block_hash(bytes.fromhex(h), b, num_tokens=count)
        for alias in aliases:
            pool._insert_block_hash(bytes.fromhex(alias), b, num_tokens=count)


def tensors_inventory(mapping):
    """Persist complete unique storages: packed views, strides and aliases survive."""
    import torch
    storages, ids, views = [], {}, []

    def visit(value, path):
        if isinstance(value, torch.Tensor):
            require(value.layout == torch.strided, "Unsupported tensor layout")
            s = value.untyped_storage()
            key = (str(value.device), s.data_ptr(), s.nbytes())
            if key not in ids:
                ids[key] = len(storages)
                raw = torch.empty(0, dtype=torch.uint8, device=value.device).set_(s, 0, (s.nbytes(),), (1,))
                storages.append(raw)
            views.append([path, ids[key], str(value.dtype), list(value.shape),
                          list(value.stride()), value.storage_offset()])
        elif isinstance(value, dict):
            for k in sorted(value):
                require(isinstance(k, str), "Non-string layer key")
                visit(value[k], [*path, k])
        elif isinstance(value, (tuple, list)):
            for i, v in enumerate(value):
                visit(v, [*path, i])
        else:
            raise ValueError(f"Unsupported KV member {type(value)} at {path}")
    visit(mapping, [])
    require(bool(storages), "No captured KV allocations")
    return {"views": views, "sizes": [s.numel() for s in storages]}, storages


def write_storages(path, descriptor, storages):
    import torch
    path.mkdir(mode=0o700, parents=True, exist_ok=False)
    require(shutil.disk_usage(path).free > sum(descriptor["sizes"]) + 1024**3,
            "Insufficient disk space (requires cache size plus 1 GiB)")
    files = []
    for i, raw in enumerate(storages):
        sha = hashlib.sha256()
        target = path / f"storage-{i:04d}.bin"
        with open(target, "xb") as f:
            os.chmod(target, 0o600)
            for offset in range(0, raw.numel(), CHUNK):
                # Blocking device-to-host copy; no full-pool CPU allocation.
                part = raw[offset:offset + CHUNK].to(device="cpu", copy=True)
                payload = memoryview(part.numpy())
                f.write(payload)
                sha.update(payload)
            f.flush()
            os.fsync(f.fileno())
        files.append({"size": raw.numel(), "sha256": sha.hexdigest()})
    result = {"descriptor": descriptor, "files": files}
    atomic_json(path / "rank.json", result)
    return result


def verify_storages(path, expected, descriptor):
    require(expected["descriptor"] == descriptor, "Tensor shape/stride/storage layout changed")
    require(json.loads((path / "rank.json").read_text()) == expected, "Rank manifest mismatch")
    require(len(expected["files"]) == len(descriptor["sizes"]), "Storage count mismatch")
    for i, info in enumerate(expected["files"]):
        fpath = path / f"storage-{i:04d}.bin"
        require(not fpath.is_symlink(), "Symlink storage not allowed")
        require(info["size"] == descriptor["sizes"][i] == fpath.stat().st_size, "Truncated storage")
        sha = hashlib.sha256()
        with open(fpath, "rb") as f:
            while chunk := f.read(CHUNK):
                sha.update(chunk)
        require(sha.hexdigest() == info["sha256"], "Storage checksum mismatch")


def read_storages(path, expected, storages):
    import torch
    for i, raw in enumerate(storages):
        sha = hashlib.sha256()
        with open(path / f"storage-{i:04d}.bin", "rb") as f:
            for offset in range(0, raw.numel(), CHUNK):
                payload = bytearray(f.read(min(CHUNK, raw.numel() - offset)))
                require(len(payload) == min(CHUNK, raw.numel() - offset), "Storage changed during restore")
                sha.update(payload)
                if payload:
                    raw[offset:offset + len(payload)].copy_(torch.frombuffer(payload, dtype=torch.uint8))
        require(sha.hexdigest() == expected["files"][i]["sha256"], "Storage changed after verification")


def worker_operation(worker, operation, root, generation, entries=None):
    """Called collectively on ALL ranks, on their normal worker execution thread."""
    import torch
    require(Path(root).is_dir() and Path(root).stat().st_mode & 0o077 == 0,
            "Every rank requires an existing private snapshot root (0700)")
    rank = worker.rank
    runner = worker.model_runner
    descriptor, storages = tensors_inventory(runner._dgx_snapshot_tensors)
    # Detect replacement of files at an unchanged model path as well as wrong rank.
    model_path = Path(runner.model_config.model)
    require(model_path.is_dir(), "Only local model artifacts supported")
    descriptor["artifact_files"] = [[str(f.relative_to(model_path)), f.stat().st_size, f.stat().st_mtime_ns]
                                    for f in sorted(model_path.rglob("*"))
                                    if f.is_file() and f.suffix in (".safetensors", ".json", ".bin", ".pt")]
    require(bool(descriptor["artifact_files"]), "No model artifacts found")
    descriptor["model_id"] = os.environ.get("DGX_KV_MODEL_ID")
    require(bool(re.fullmatch(r"[a-f0-9]{64}", descriptor["model_id"] or "")), "Missing worker model identity")
    descriptor["runtime_id"] = os.environ.get("DGX_KV_RUNTIME_ID")
    require(bool(re.fullmatch(r"sha256:[a-f0-9]{64}", descriptor["runtime_id"] or "")), "Missing image identity")
    torch.cuda.synchronize()
    path = generation_dir(root, generation) / f"rank-{rank}"
    if operation == "save":
        generation_dir(root, generation).mkdir(mode=0o700, parents=True, exist_ok=True)
        result = write_storages(path, descriptor, storages)
        return {"rank": rank, **result}
    require(entries is not None, "Missing rank manifests")
    expected = next(e for e in entries if e["rank"] == rank)
    local = {k: v for k, v in expected.items() if k != "rank"}
    if operation == "verify":
        verify_storages(path, local, descriptor)
    elif operation == "restore":
        read_storages(path, local, storages)
        torch.cuda.synchronize()
    else:
        raise ValueError("Unknown operation")
    return {"rank": rank, "ok": True}


def guard_idle(core):
    require(not core.scheduler.has_requests() and not core.batch_queue, "Engine has active or queued work")
    require(core.scheduler.connector is None, "KV transfer connector is unsupported")
    coordinator = core.scheduler.kv_cache_manager.coordinator
    for manager in coordinator.single_type_managers:
        for name in ("req_to_blocks", "num_cached_block", "_partial_hit_reqs", "_pending_cow_copies",
                     "_pending_boundary_state_offloads", "new_block_ids"):
            require(not getattr(manager, name, None), f"Unconsumed manager state: {name}")
    return coordinator.block_pool


def fingerprint(core):
    from vllm.v1.core import kv_cache_utils
    c = core.vllm_config
    require(c.model_config.hf_config.model_type == "deepseek_v41", "Only DeepSeek V4.1 adapter supported")
    require(getattr(c, "lora_config", None) is None, "LoRA unsupported")
    require(c.parallel_config.tensor_parallel_size == 3, "Only TP3 supported by this adapter")
    require(c.parallel_config.pipeline_parallel_size == 1 and c.parallel_config.data_parallel_size == 1,
            "PP/DP unsupported")
    identity = os.environ.get("DGX_KV_MODEL_ID", "")
    require(bool(re.fullmatch(r"[a-f0-9]{64}", identity)), "DGX_KV_MODEL_ID must be trusted model manifest SHA256")
    runtime_id = os.environ.get("DGX_KV_RUNTIME_ID", "")
    require(bool(re.fullmatch(r"sha256:[a-f0-9]{64}", runtime_id)), "Runtime image digest required")
    # Explicit configuration, hash seed, exact source ABI; no volatile pointers.
    config = {"model": c.model_config.hf_config.to_dict(), "dtype": str(c.model_config.dtype),
              "cache": str(c.cache_config), "parallel": [3, 1, 1],
              "speculative": str(c.speculative_config),
              "scheduler": [c.scheduler_config.max_num_batched_tokens, c.scheduler_config.max_num_seqs,
                            c.model_config.max_model_len],
              "none_hash": kv_cache_utils.NONE_HASH.hex(), "model_id": identity, "runtime_id": runtime_id}
    root = Path(kv_cache_utils.__file__).parents[2]
    sources = ["v1/core/block_pool.py", "v1/core/kv_cache_utils.py", "v1/core/kv_cache_coordinator.py",
               "v1/core/single_type_kv_cache_manager.py", "v1/kv_cache_interface.py",
               "v1/worker/gpu_model_runner.py", "v1/worker/utils.py", "v1/worker/gpu/model_runner.py", "v1/worker/gpu/attn_utils.py"]
    config["source_abi"] = {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in sources}
    config["adapter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return digest(config)


class Controller:
    def __init__(self, core):
        self.core = core
        self.root = Path(os.environ["DGX_KV_SNAPSHOT_DIR"]).resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        require(self.root.stat().st_mode & 0o077 == 0, "Snapshot root must be private (0700)")
        self.fingerprint = fingerprint(core)
        self.last_poll = 0.0
        self.hold = False
        self.restore_at_start()

    def rpc(self, operation, generation, entries=None):
        result = self.core.model_executor.collective_rpc(
            worker_operation, args=(operation, str(self.root), generation, entries))
        require(isinstance(result, list) and sorted(e["rank"] for e in result) == [0, 1, 2],
                "Missing or duplicate TP rank response")
        return sorted(result, key=lambda e: e["rank"])

    def status(self, **values):
        atomic_json(self.root / "status.json", {"time": time.time(), **values})

    def restore_at_start(self):
        generation = os.environ.get("DGX_KV_RESTORE_ID", "")
        if not generation:
            self.status(state="cold", reason="No restore requested")
            return
        pool = guard_idle(self.core)
        require(not pool.cached_block_hash_to_block._cache, "Startup cache already populated")
        try:
            manifest = json.loads((generation_dir(self.root, generation) / "complete.json").read_text())
            require(manifest["format"] == FORMAT and manifest["fingerprint"] == self.fingerprint,
                    "Snapshot incompatible with this model/configuration")
            require(manifest["generation"] == generation, "Wrong generation")
            require(manifest["metadata_sha256"] == digest(manifest["pool"]), "Metadata checksum mismatch")
            pool_validate(pool, manifest["pool"])
            self.rpc("verify", generation, manifest["ranks"])
        except Exception as exc:
            # No GPU bytes touched, so a genuine cold start is safe here.
            self.status(state="cold", reason=str(exc), rejected=generation)
            return
        # Any failure after writes start is fatal: never serve a partial restoration.
        self.rpc("restore", generation, manifest["ranks"])
        pool_restore(pool, manifest["pool"])
        require(pool_export(pool) == manifest["pool"], "Restored metadata differs")
        self.status(state="restored", generation=generation, cached_blocks=len(manifest["pool"]["cached"]))

    def poll(self):
        now = time.monotonic()
        if now - self.last_poll < 1:
            return
        self.last_poll = now
        command = self.root / "command.json"
        if not command.exists():
            return
        # Rename to claim exactly once. Requests are kept as an audit trail.
        generation = "invalid"
        try:
            require(not command.is_symlink(), "Command symlink rejected")
            cmd = json.loads(command.read_text())
            generation = cmd["id"]
            generation_dir(self.root, generation)
            claimed = self.root / (generation + ".command.json")
            require(not claimed.exists(), "Command ID already used")
            os.rename(command, claimed)
            if cmd["action"] == "resume":
                self.hold = False
                self.status(state="serving", command=generation)
                return
            require(cmd["action"] == "save", "Unknown command")
            pool = guard_idle(self.core)
            data = pool_export(pool)
            folder = generation_dir(self.root, generation)
            folder.mkdir(mode=0o700, exist_ok=False)
            self.status(state="saving", generation=generation)
            ranks = self.rpc("save", generation)
            atomic_json(folder / "complete.json", {"format": FORMAT, "generation": generation,
                        "fingerprint": self.fingerprint, "metadata_sha256": digest(data),
                        "pool": data, "ranks": ranks})
            self.hold = True
            self.status(state="saved-held", generation=generation, cached_blocks=len(data["cached"]),
                        bytes=sum(sum(r["descriptor"]["sizes"]) for r in ranks))
        except Exception as exc:
            # Save only reads tensors: no serving state changed by failed snapshot.
            self.status(state="save-failed", command=generation, reason=str(exc))


def tick(core):
    if not os.environ.get("DGX_KV_SNAPSHOT_DIR"):
        return
    if not hasattr(core, "_dgx_snapshot_controller"):
        core._dgx_snapshot_controller = Controller(core)
    controller = core._dgx_snapshot_controller
    controller.poll()
    while controller.hold and core.is_running():
        time.sleep(0.25)
        controller.poll()


def queue_timeout(block):
    return 1.0 if block and os.environ.get("DGX_KV_SNAPSHOT_DIR") else None
