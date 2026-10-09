"""Download the pinned checkpoint and prepare TP3 body and local Engram copies."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROFILE = json.loads((ROOT / 'config/x11c.json').read_text())


def prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def ranges(config, rank):
    """Match EngramLayout and ParallelEngramEmbedding head bucket boundaries."""
    if rank not in (0, 1, 2):
        raise ValueError('this installation target is TP3')
    c = config.get('text_config', config)
    seen, rows = set(), {}
    columns = (c['engram_max_ngram_size'] - 1) * c['engram_n_heads']
    width = math.ceil(columns / 3)
    for layer, expected in zip(c['engram_layer_ids'], c['engram_num_embeddings']):
        sizes = []
        for _ in range(c['engram_max_ngram_size'] - 1):
            current = c['engram_vocab_size'] - 1
            for _ in range(c['engram_n_heads']):
                current += 1
                while not prime(current) or current in seen:
                    current += 1
                seen.add(current)
                sizes.append(current)
        if sum(sizes) != expected:
            raise ValueError('checkpoint Engram bucket geometry differs')
        rows[str(layer)] = [sum(sizes[:rank * width]), sum(sizes[:(rank + 1) * width])]
    return rows


def fresh(destination, source):
    destination = destination.resolve()
    source = source.resolve()
    if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('source and destination must be separate directories')
    if destination.exists():
        raise FileExistsError('use a new destination; existing data is never replaced')
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def download(destination):
    from huggingface_hub import snapshot_download
    checkpoint = PROFILE['checkpoint']
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=checkpoint['repository'], revision=checkpoint['revision'],
                      local_dir=str(destination), max_workers=2)
    (destination / 'recipe-download-receipt.json').write_text(json.dumps(checkpoint, indent=2) + '\n')


def verify(source):
    inventory = json.loads((ROOT / 'config/checkpoint-files.json').read_text())
    for row in inventory['files']:
        path = source / row['rfilename']
        if not path.is_file() or path.stat().st_size != row['size']:
            raise ValueError('checkpoint file size differs: ' + row['rfilename'])
        lfs = row.get('lfs')
        h = hashlib.sha256() if lfs else hashlib.sha1()
        if not lfs:
            h.update(f"blob {row['size']}\0".encode())
        with path.open('rb') as stream:
            while raw := stream.read(32 * 1024**2):
                h.update(raw)
        expected = lfs['sha256'] if lfs else row['blobId']
        if h.hexdigest() != expected:
            raise ValueError('checkpoint hash differs: ' + row['rfilename'])
        print('verified ' + row['rfilename'], flush=True)


def body(source, destination):
    destination = fresh(destination, source)
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    wm = index['weight_map']
    tables = {k: v for k, v in wm.items() if '.engram.embed.' in k}
    remaining = {k: v for k, v in wm.items() if k not in tables}
    if not tables or set(tables.values()) & set(remaining.values()):
        raise ValueError('expected separate Engram shards in the original checkpoint')
    destination.mkdir()
    # Hard links avoid a second 257 GB body copy on the download node. On NFS
    # or another filesystem, use a normal verified byte copy.
    for name in sorted(set(remaining.values())):
        src, dst = source / name, destination / name
        if not src.is_file():
            raise FileNotFoundError(src)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    for p in source.iterdir():
        if p.is_file() and p.suffix != '.safetensors' and p.name not in {
            'config.json', 'model.safetensors.index.json', 'recipe-download-receipt.json'
        }:
            shutil.copy2(p, destination / p.name)
    # This is the observed TP3 virtual-head configuration. Weight bytes stay
    # unchanged. See docs/ARCHITECTURE.md for the 64 to 72 head padding.
    shutil.copy2(ROOT / 'config/model-config.reference.json', destination / 'config.json')
    total = sum((source / name).stat().st_size for name in set(remaining.values()))
    index['weight_map'] = remaining
    # Safetensors metadata total_size describes tensor bytes, not file headers.
    # Keep the checkpoint's body figure instead of substituting a file sum.
    expected = PROFILE.get('body_tensor_bytes', 257251577160)
    index['metadata'] = {**index.get('metadata', {}), 'total_size': expected}
    (destination / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2) + '\n')
    receipt = {'checkpoint': PROFILE['checkpoint'], 'tp': 3, 'weight_bytes_modified': False,
               'body_shards': len(set(remaining.values())), 'body_file_bytes': total,
               'body_tensor_count': len(remaining), 'engram_excluded_tensors': sorted(tables),
               'config_sha256': hashlib.sha256((destination / 'config.json').read_bytes()).hexdigest()}
    (destination / 'recipe-body-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


def engram(source, destination, rank, mbps):
    destination = fresh(destination, source)
    config = json.loads((ROOT / 'config/model-config.reference.json').read_text())
    rows = ranges(config, rank)
    command = [sys.executable, str(ROOT / 'tools/engram_local.py'), str(source), str(destination)]
    command += [f'{layer}:{lo}:{hi}' for layer, (lo, hi) in rows.items()]
    command += [f'--mbps={mbps}']
    subprocess.run(command, check=True)
    print(json.dumps({'tp': 3, 'rank': rank, 'ranges': rows}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['download', 'verify', 'body', 'engram', 'ranges'])
    parser.add_argument('--source', type=Path)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--rank', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--mbps', type=float, default=600)
    a = parser.parse_args()
    if a.action == 'ranges':
        print(json.dumps(ranges(json.loads((ROOT / 'config/model-config.reference.json').read_text()), a.rank), indent=2))
    elif a.action == 'download':
        if not a.destination:
            parser.error('--destination required')
        download(a.destination)
    elif a.action == 'verify':
        if not a.source:
            parser.error('--source required')
        verify(a.source)
    else:
        if not a.source or not a.destination:
            parser.error('--source and --destination required')
        if a.action == 'body':
            body(a.source, a.destination)
        else:
            if a.mbps <= 0:
                parser.error('--mbps must be positive')
            engram(a.source, a.destination, a.rank, a.mbps)


if __name__ == '__main__':
    main()
