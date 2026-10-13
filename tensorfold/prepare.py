"""CPU-only preparation of the pinned TF351/TF29 rank artifacts. Never starts services."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--kit', required=True, type=Path)
    parser.add_argument('--pollard', required=True, type=Path)
    parser.add_argument('--mia', required=True, type=Path)
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--read-mib-s', type=float, default=128)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'inputs.json').read_text())
    for file in (ROOT / 'preparation').iterdir():
        expected = manifest['files']['preparation/' + file.name]
        if file.stat().st_size != expected['bytes'] or hashlib.sha256(file.read_bytes()).hexdigest() != expected['sha256']:
            raise ValueError('Pinned preparation changed: ' + file.name)
    if not args.write:
        print(json.dumps({'status': 'DRY_RUN', 'arms': ['TF351', 'TF29'], 'rank_artifacts': 6,
                          'rank_disk_bytes': 472964528463, 'vision_extra_bytes': 970536960,
                          'source': str(args.source), 'work': str(args.work), 'out': str(args.out),
                          'gpu_or_services_touched': False}, indent=2))
        return
    args.work.mkdir(parents=True, exist_ok=False)
    for file in (ROOT / 'preparation').iterdir():
        shutil.copyfile(file, args.work / file.name)
    spec = {'schema': 'ds41f.cpu-weight-preparation.v1', 'source_commit': manifest['runtime_commit'],
            'production_readonly_source': str(args.pollard.resolve()), 'mia_source': str(args.mia.resolve()),
            'files': {p.name: {'size': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p in args.work.iterdir()}}
    (args.work / 'PREPARATION.json').write_text(json.dumps(spec, indent=2) + '\n')
    tools = args.source / 'tools/dsv41-tp3'
    for name, directory in [('pollard', args.pollard), ('mia', args.mia)]:
        command = [sys.executable, str(tools / 'source_verify.py'), '--manifest',
                   str(args.work / (name + '-upstream-manifest.json')), '--root', str(directory),
                   '--read-mib-s', str(args.read_mib_s), '--out', str(args.work / (name + '-receipt.json'))]
        if name == 'pollard':
            command += ['--shards', str(args.work / 'pollard-selected.json')]
        else:
            command += ['--weights-only']
        subprocess.run(command, check=True)
    subprocess.run([sys.executable, str(tools / 'export_campaign.py'), '--preparation', str(args.work),
                    '--serving-kit', str(args.kit), '--out', str(args.out),
                    '--read-mib-s', str(args.read_mib_s), '--write'], check=True)


if __name__ == '__main__':
    main()
