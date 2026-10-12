"""Create a separate build context for a recorded host64k arm. No GPU actions."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'experiments/20261012'


def target(relative):
    for prefix, root in [('vllm/', 'usr/local/lib/python3.12/dist-packages/'),
                         ('dgx_campaign_runtime/', 'opt/dgx_campaign/runtime/'),
                         ('dgx_next_runtime/', 'opt/dgx_next/runtime/'),
                         ('b12x/', 'opt/dgx_campaign/vendor/b12x-package/')]:
        if relative.startswith(prefix):
            return 'runtime/rootfs/' + root + relative
    if relative == 'native/gemm.cu':
        matches = list((ROOT / 'native/cuda-exl3').rglob('gemm.cu'))
        if len(matches) != 1:
            raise ValueError('native source layout changed')
        return matches[0].relative_to(ROOT).as_posix()
    raise ValueError('unknown archive path: ' + relative)


def read_entry(entry):
    raw = (ARCHIVE / entry['file']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError('archival source hash differs')
    return raw


def materialize(arm, output):
    variants = json.loads((ARCHIVE / 'variants.json').read_text())['arms']
    selected = next((row for row in variants if row['arm'] == arm), None)
    if selected is None or not selected['measured']:
        raise ValueError('choose a measured arm; N9 was a no-op, not a measured variant')
    output = output.resolve()
    if output == ROOT or ROOT in output.parents or output.exists():
        raise ValueError('use a new directory outside this checkout')
    # Copy the build/tool scaffold; the full runtime comes from the baseline map.
    output.mkdir(parents=True)
    for name in ['build', 'tools', 'config', 'native', 'licenses']:
        shutil.copytree(ROOT / name, output / name, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['.dockerignore', '.gitignore', 'LICENSE', 'THIRD_PARTY_NOTICES.md']:
        shutil.copy2(ROOT / name, output / name)
    def put(name, raw):
        p = output / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    baseline = json.loads((ARCHIVE / 'baseline.json').read_text())['files']
    for path, entry in baseline.items():
        put(path, read_entry(entry))
    for relative, entry in selected['files'].items():
        put(target(relative), read_entry(entry))
    for source in (ROOT / 'runtime/rootfs').rglob('*'):
        if source.is_file() and source.name.startswith(('LICENSE', 'NOTICE', 'COPYING')):
            put(source.relative_to(ROOT).as_posix(), source.read_bytes())
    manifest = {'candidate': arm, 'parts': selected['parts'],
                'files': {k: v['sha256'] for k, v in selected['files'].items()},
                'publication_note': 'Portable source archive; new build needs its own qualification.'}
    put('runtime/rootfs/opt/dgx_next/CANDIDATE.json',
        (json.dumps(manifest, indent=2) + '\n').encode())
    # The default tree contains N4. Non-N4 arms must rebuild the pre-N4 kernel.
    if 'N4' not in selected['parts']:
        native_path = target('native/gemm.cu')
        raw = subprocess.check_output(['git', 'show', 'v0.8.9X11c:' + native_path], cwd=ROOT)
        put(native_path, raw)
    profile = json.loads((ROOT / 'config/ds41f-1.0.0.json').read_text())
    profile.update(version='experiment-' + arm, display_name='Archived ' + arm,
                   candidate=arm, parts=selected['parts'], internal_version='0.8.9X11c-' + arm,
                   environment=selected['environment'],
                   compilation=json.loads(selected['flags']['--compilation-config']))
    put('config/ds41f-1.0.0.json', (json.dumps(profile, indent=2) + '\n').encode())
    cluster = json.loads((output / 'config/cluster.example.json').read_text())
    cluster['image'] = 'ds41f:experiment-' + arm.lower()
    put('config/cluster.example.json', (json.dumps(cluster, indent=2) + '\n').encode())
    dockerfile = (output / 'build/Dockerfile').read_text()
    put('build/Dockerfile', dockerfile.replace('dgx.recipe.version="1.0.0"',
         'dgx.recipe.version="experiment-' + arm + '"').encode())
    put('EXPERIMENT.json', (json.dumps({'arm': arm, 'parts': selected['parts'],
        'image': cluster['image'], 'new_build_requires_gpu_qualification': True,
        'source': '20261012 campaign source archive; portable paths/control adapter'}, indent=2) + '\n').encode())
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(materialize(args.arm, args.output))


if __name__ == '__main__':
    main()
