"""Fetch and verify the retained ARM64 base, then optionally import it into Docker."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 4 * 1024**2


def verify(path, item):
    if path.stat().st_size != item['bytes']:
        raise ValueError(f"size mismatch: {path.name}; retain this file for inspection")
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(CHUNK):
            digest.update(data)
    if digest.hexdigest() != item['sha256']:
        raise ValueError(f"SHA256 mismatch: {path.name}; retain this file for inspection")


def fetch(url, destination, item, attempts=5):
    if destination.exists():
        verify(destination, item)
        return
    partial = destination.with_name(destination.name + '.partial')
    for attempt in range(attempts):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > item['bytes']:
            raise ValueError(f"oversized partial file: {partial}")
        if offset == item['bytes']:
            verify(partial, item)
            os.replace(partial, destination)
            return
        request = urllib.request.Request(url, headers={
            'User-Agent': 'dsv41-x11c-recipe', 'Range': f'bytes={offset}-'})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status = response.status
                if status == 206:
                    expected = f"bytes {offset}-{item['bytes']-1}/{item['bytes']}"
                    if response.headers.get('Content-Range') != expected:
                        raise ValueError('unexpected Content-Range; refusing to append')
                    mode = 'ab'
                elif status == 200 and offset == 0:
                    mode = 'xb'
                elif status == 200:
                    # A server may ignore Range. Keep the existing partial intact.
                    raise ValueError('server ignored resume Range; existing partial retained')
                else:
                    raise ValueError(f'unexpected download status {status}')
                with partial.open(mode) as stream:
                    current = offset
                    while data := response.read(CHUNK):
                        current += len(data)
                        if current > item['bytes']:
                            raise ValueError('download exceeded manifest size')
                        stream.write(data)
            verify(partial, item)
            os.replace(partial, destination)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            if attempt + 1 == attempts:
                raise
            time.sleep(min(2 ** attempt, 16))
    raise RuntimeError('download attempts exhausted')


def image_identity(tag):
    result = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', tag],
                            text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def load(parts, image_id, local_tag):
    previous = image_identity(local_tag)
    if previous and previous != image_id:
        raise ValueError('the base tag already identifies a different image; retain it and resolve explicitly')
    if previous == image_id:
        print('Exact base image is already loaded.')
        return
    process = subprocess.Popen(['docker', 'load'], stdin=subprocess.PIPE)
    try:
        for path in parts:
            with path.open('rb') as stream:
                while data := stream.read(CHUNK):
                    process.stdin.write(data)
        process.stdin.close()
    except BaseException:
        process.stdin.close()
        process.wait()
        raise
    if process.wait() != 0:
        raise RuntimeError('docker load failed; downloaded files retained')
    if image_identity(local_tag) != image_id:
        raise ValueError('loaded image identity does not match the pinned base')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'config/base-image.json')
    parser.add_argument('--cache', type=Path, default=ROOT / '.cache/base-image')
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest['schema'] != 'dsv41.public-base-image.v1' or manifest['architecture'] != 'arm64':
        raise ValueError('unsupported base manifest')
    if args.load:
        previous = image_identity(manifest['local_tag'])
        if previous == manifest['image_id']:
            print('Exact base image is already loaded.')
            return
        if previous:
            raise ValueError('existing base tag differs; no image was changed')
    args.cache.mkdir(parents=True, exist_ok=True)
    paths = []
    for item in manifest['parts']:
        if not re.fullmatch(r'x11c-base-image\.tar\.gz\.part[0-9]{3}', item['file']):
            raise ValueError('unexpected archive part name')
        if not re.fullmatch(r'[0-9a-f]{64}', item['sha256']) or not 0 < item['bytes'] <= 1024**3:
            raise ValueError('invalid archive part checksum or size')
        path = args.cache / item['file']
        print(f"Checking {item['file']}", flush=True)
        fetch(manifest['download_base'].rstrip('/') + '/' + item['file'], path, item)
        paths.append(path)
    if args.load:
        load(paths, manifest['image_id'], manifest['local_tag'])


if __name__ == '__main__':
    main()
