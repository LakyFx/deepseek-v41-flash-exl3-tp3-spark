"""Build a separate, credited NCCL ring runtime. Never replace the TP3 library."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.base = a.base.resolve(strict=True)
    a.output = a.output.resolve()
    here = Path(__file__).resolve().parent / 'nccl-ring'
    manifest = json.loads((here / 'dual-pci-domain.json').read_bytes())
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=a.base, text=True).strip()
    changed = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=a.base, text=True)
    if revision != manifest['revision'] or changed.strip():
        raise ValueError('Require an unchanged checkout of the pinned NCCL revision')
    assert not a.output.exists()
    a.output.mkdir()
    assert digest(here / 'dual-pci-domain.patch') == manifest['patch_sha256']
    original = {}
    for root in ['src', 'makefiles']:
        for f in sorted((a.base / root).rglob('*')):
            if f.is_file():
                original[str(f.relative_to(a.base))] = digest(f)
    source = a.output / 'source'
    shutil.copytree(a.base, source, ignore=shutil.ignore_patterns('.git'))
    patches = ['dual-pci-domain.patch', 'cts-acquire-fence.patch']
    for name in patches:
        patch = here / name
        subprocess.run(['git', 'apply', '--check', str(patch)], cwd=source, check=True)
        subprocess.run(['git', 'apply', str(patch)], cwd=source, check=True)
    # Upstream CPU compatibility checks; no CUDA work is submitted.
    subprocess.run(['g++', '-std=c++11', '-O2', '-Wall', '-Wextra', '-Werror',
                    'tests/routing_handle/compat.cc', '-o', str(a.output / 'routing-handle-test')], cwd=source, check=True)
    subprocess.run([str(a.output / 'routing-handle-test')], check=True)
    cuda = Path('/usr/local/cuda').resolve(strict=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    start = time.time()
    subprocess.run(['make', '-j8', 'src.build', 'CUDA_HOME=' + str(cuda),
                    'NVCC_GENCODE=-gencode=arch=compute_121,code=sm_121'], cwd=source, env=env, check=True)
    runtime = a.output / 'lib'
    runtime.mkdir()
    lib = source / 'build/lib/libnccl.so.2.30.7'
    markers = lib.read_bytes()
    for marker in [b'NCCL_SWITCHLESS_RING_ONLY', b'NCCL_IB_EXTENDED_IPV4_GIDS', b'NCCL_IB_PRESERVE_PCI_DOMAIN']:
        assert marker in markers, marker
    shutil.copy2(lib, runtime / lib.name)
    for name in ['libnccl.so', 'libnccl.so.2']:
        (runtime / name).symlink_to(lib.name)
    for name in ['LICENSE.txt', 'ThirdPartyNotices.txt']:
        shutil.copy2(here / name, runtime / name)
    changes = {}
    for f in sorted((source / 'src').rglob('*')):
        if f.is_file():
            key = str(f.relative_to(source))
            if original.get(key) != digest(f):
                changes[key] = dict(before=original.get(key), after=digest(f))
    receipt = dict(status='COMPLETED', start=start, end=time.time(), source_revision=manifest['revision'],
                   base_source_sha256=hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest(),
                   patches={name: digest(here/name) for name in patches}, changed_files=changes,
                   cuda=str(cuda), nvcc=subprocess.check_output([str(cuda/'bin/nvcc'), '--version'], text=True),
                   library=str(runtime/lib.name), library_sha256=digest(runtime/lib.name),
                   cpu_compatibility='PASS', gpu_qualified=False, base_untouched=True)
    (a.output / 'BUILD.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
