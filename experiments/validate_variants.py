"""Validate archived build contexts; optionally build on a dedicated ARM64 host.

Retains contexts, logs and receipts. Never deploys or loads model weights.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import materialize as archive


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(row, output):
    expected = {name: entry['sha256'] for name, entry in
                json.loads((archive.ARCHIVE / 'baseline.json').read_text())['files'].items()}
    expected.update({archive.target(name): entry['sha256'] for name, entry in row['files'].items()})
    if 'N4' not in row['parts']:
        name = archive.target('native/gemm.cu')
        raw = subprocess.check_output(['git', 'show', 'v0.8.9X11c:' + name], cwd=archive.ROOT)
        expected[name] = hashlib.sha256(raw).hexdigest()
    for name, sha in expected.items():
        if digest(output / name) != sha:
            raise ValueError('materialized source mismatch: ' + name)
    profile = json.loads((output / 'config/ds41f-1.0.0.json').read_text())
    if profile['environment'] != row['environment'] or profile['parts'] != row['parts']:
        raise ValueError('environment or parts differ')
    if profile['compilation'] != json.loads(row['flags']['--compilation-config']):
        raise ValueError('compilation configuration differs')
    if profile['candidate'] != row['arm']:
        raise ValueError('candidate identity differs')
    # Exercise each generated launcher, not just the stored profile.
    probe = """import json
from tools.launch import build
from pathlib import Path
c=json.loads(Path('config/cluster.example.json').read_text())
for i,n in enumerate(c['nodes']):
 n['control_address']='192.0.2.'+str(i+1)
 n['control_interface']='management'
c['roce_cidr']='192.0.2.0/24'
print(json.dumps([build(c,r,'a'*64) for r in range(3)]))
"""
    commands = json.loads(subprocess.check_output([sys.executable, '-c', probe], cwd=output, text=True))
    for command in commands:
        for flag, value in row['flags'].items():
            actual = command[command.index(flag) + 1]
            if flag in ('--speculative-config', '--compilation-config'):
                if json.loads(actual) != json.loads(value):
                    raise ValueError('launch JSON differs: ' + flag)
            elif actual != value:
                raise ValueError('launch flag differs: ' + flag)
    subprocess.run([sys.executable, '-m', 'compileall', '-q', str(output / 'tools')], check=True)
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory outside checkout; retained')
    parser.add_argument('--arms', nargs='+', help='Default: all measured arms')
    parser.add_argument('--build', action='store_true', help='Explicit ARM64 Docker build and CPU ABI check; no GPU')
    args = parser.parse_args()
    rows = [r for r in json.loads((archive.ARCHIVE / 'variants.json').read_text())['arms'] if r['measured']]
    names = {r['arm'] for r in rows}
    if args.arms and (set(args.arms) - names):
        parser.error('unknown/unmeasured arm; N9 has no build')
    if args.arms:
        rows = [r for r in rows if r['arm'] in args.arms]
    if args.build and (sys.platform != 'linux' or platform.machine() not in ('aarch64', 'arm64')):
        parser.error('--build requires a dedicated Linux ARM64 Docker host')
    output = args.output.resolve()
    if output.exists() or output == archive.ROOT or archive.ROOT in output.parents:
        parser.error('output must be new and outside checkout')
    output.mkdir(parents=True)
    receipt = {'schema': 'ds41f.variant-validation.v1',
               'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=archive.ROOT, text=True).strip(),
               'tracked_tree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=archive.ROOT)),
               'scope': 'source reconstruction; optional CPU-only image build; no GPU qualification', 'arms': []}
    for row in rows:
        record = {'arm': row['arm'], 'materialization': 'not_run', 'public_build': 'not_run', 'cpu_abi': 'not_run', 'gpu_requalification': 'not_run'}
        receipt['arms'].append(record)
        try:
            context = archive.materialize(row['arm'], output / row['arm'])
            record['verified_source_files'] = validate(row, context)
            record['materialization'] = 'passed'
            if args.build:
                import os
                image = 'ds41f:check-' + row['arm'].lower() + '-' + receipt['source_commit'][:12]
                record['image_tag'] = image
                record['public_build'] = 'failed'
                with (output / (row['arm'] + '-build.log')).open('w') as log:
                    subprocess.run(['bash', 'build/build.sh'], cwd=context, env={**os.environ, 'RECIPE_IMAGE': image}, stdout=log, stderr=subprocess.STDOUT, check=True)
                record['public_build'] = 'passed'
                record['image_id'] = subprocess.check_output(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'], text=True).strip()
                record['cpu_abi'] = 'failed'
                with (output / (row['arm'] + '-abi.log')).open('w') as log:
                    subprocess.run(['docker', 'run', '--cidfile', str(output / (row['arm'] + '-container-id.txt')), '--runtime', 'runc', '--network', 'none',
                        '-e', 'NVIDIA_VISIBLE_DEVICES=void', '--memory=2500m',
                        '--mount', 'type=bind,src=' + str(context / 'build/verify_image.py') + ',dst=/qualification.py,readonly',
                        '--entrypoint', 'python3', image, '/qualification.py'], stdout=log, stderr=subprocess.STDOUT, check=True)
                record['cpu_abi'] = 'passed'
        except Exception as exc:
            if record['materialization'] == 'not_run':
                record['materialization'] = 'failed'
            record['error_type'] = type(exc).__name__
            print(row['arm'] + ': ' + str(exc), file=sys.stderr)
        cid = output / (row['arm'] + '-container-id.txt')
        if cid.exists():
            record['retained_container_id'] = cid.read_text().strip()
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(row['arm'] + ': ' + record['materialization'], flush=True)
    return 1 if any('error_type' in r for r in receipt['arms']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
