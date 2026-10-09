#!/usr/bin/env python3
"""Make a node-local sparse copy of this rank's Engram rows, for DSV41_ENGRAM_DIR.

For each LAYER:START:END it copies, from the (NFS) model dir, the shard's safetensors header plus
rows [START, END) of layers.LAYER.engram.embed.weight and .scale, at the SAME byte offsets, into a
sparse local file of the same name and size. Then it copies the index json, writes engram-local.json
(row range per layer) and checks random rows byte for byte against the source.
Reads are rate-limited and dropped from the page cache as it goes (unified-memory box, live vLLM).

usage: engram_local.py SRC_DIR DST_DIR LAYER:START:END [LAYER:START:END ...] [--mbps=600]
"""
import json
import os
import random
import struct
import sys
import time

src, dst = sys.argv[1], sys.argv[2]
specs = [a for a in sys.argv[3:] if not a.startswith("--")]
mbps = float(next((a.split("=", 1)[1] for a in sys.argv[3:] if a.startswith("--mbps=")), "600"))
os.makedirs(dst, exist_ok=True)
wm = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
CH = 32 << 20


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


jobs, fds, layers = [], {}, {}
for spec in specs:
    layer, start, end = (int(x) for x in spec.split(":"))
    layers[str(layer)] = [start, end]
    for kind, row_bytes in (("weight", None), ("scale", None)):
        name = f"layers.{layer}.engram.embed.{kind}"
        shard = wm[name]
        hlen, h = header(f"{src}/{shard}")
        a, _ = h[name]["data_offsets"]
        row_bytes = h[name]["shape"][1]  # fp8 / e8m0: one byte per element
        jobs.append((shard, hlen, hlen + a + start * row_bytes, (end - start) * row_bytes, name, row_bytes, start, end))
        if shard not in fds:
            sfd = os.open(f"{src}/{shard}", os.O_RDONLY)
            dfd = os.open(f"{dst}/{shard}", os.O_RDWR | os.O_CREAT, 0o644)
            os.ftruncate(dfd, os.fstat(sfd).st_size)  # sparse: holes cost no disk
            os.pwrite(dfd, os.pread(sfd, hlen, 0), 0)  # safetensors header, verbatim
            fds[shard] = (sfd, dfd)

total = sum(j[3] for j in jobs)
print(f"copying {total / 2**30:.1f} GiB in {len(jobs)} ranges, limit {mbps:.0f} MB/s", flush=True)
done, t0, next_report = 0, time.time(), 2 << 30
for shard, hlen, off, length, name, rb, s, e in jobs:
    sfd, dfd = fds[shard]
    end = off + length
    while off < end:
        n = min(CH, end - off)
        buf = os.pread(sfd, n, off)
        if len(buf) != n:
            raise OSError(f"short read {shard} at {off}")
        os.pwrite(dfd, buf, off)
        os.posix_fadvise(sfd, off, n, os.POSIX_FADV_DONTNEED)
        off += n
        done += n
        if done >= next_report:
            next_report += 2 << 30
            os.fdatasync(dfd)
            os.posix_fadvise(dfd, 0, 0, os.POSIX_FADV_DONTNEED)
            el = time.time() - t0
            print(f"  {done / 2**30:6.1f} / {total / 2**30:.1f} GiB  {done / el / 1e6:5.0f} MB/s  {el:5.0f}s", flush=True)
        lag = done / (mbps * 1e6) - (time.time() - t0)
        if lag > 0:
            time.sleep(lag)
for sfd, dfd in fds.values():
    os.fdatasync(dfd)
    os.posix_fadvise(dfd, 0, 0, os.POSIX_FADV_DONTNEED)

with open(f"{src}/model.safetensors.index.json", "rb") as f, open(f"{dst}/model.safetensors.index.json", "wb") as g:
    g.write(f.read())
json.dump({"source": src, "layers": layers, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "note": "sparse copy: only these rows exist; tools/engram_local.py"}, open(f"{dst}/engram-local.json", "w"), indent=1)

bad, checked = 0, 0
rng = random.Random(0)
for shard, hlen, off0, length, name, rb, s, e in jobs:
    sfd, dfd = fds[shard]
    if os.pread(sfd, hlen, 0) != os.pread(dfd, hlen, 0):
        bad += 1
        print(f"HEADER MISMATCH {shard}")
    for r in [s, e - 1] + [rng.randrange(s, e) for _ in range(2000)]:
        o = off0 + (r - s) * rb
        checked += 1
        if os.pread(sfd, rb, o) != os.pread(dfd, rb, o):
            bad += 1
print(f"verify: {checked} rows checked, {bad} mismatches; elapsed {time.time() - t0:.0f}s", flush=True)
sys.exit(1 if bad else 0)
