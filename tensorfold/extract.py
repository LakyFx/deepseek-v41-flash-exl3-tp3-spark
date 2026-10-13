"""Verify every shipped input and extract three bounded archives into a NEW build directory."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    inputs = json.loads((ROOT / 'inputs.json').read_text())
    for name, expected in inputs['files'].items():
        path = ROOT / name
        if path.stat().st_size != expected['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
            raise ValueError('Shipped input hash mismatch: ' + name)
    args.out.mkdir(parents=True, exist_ok=False)
    for name, directory in [('tensorfold-source.tar.gz', 'source'), ('tensorfold-kernels.tar.gz', 'kit'),
                            ('tensorfold-rope-reference.tar.gz', 'reference')]:
        target = args.out / directory
        target.mkdir()
        with tarfile.open(ROOT / 'artifacts' / name) as archive:
            for member in archive.getmembers():
                rel = PurePosixPath(member.name)
                if rel.is_absolute() or '..' in rel.parts or not member.isfile():
                    raise ValueError('Unsafe or unexpected archive member')
                destination = target.joinpath(*rel.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open('xb') as output, archive.extractfile(member) as stream:
                    while block := stream.read(8 << 20):
                        output.write(block)
    print(json.dumps({'status': 'VERIFIED_EXTRACTED', 'runtime_commit': inputs['runtime_commit'],
                      'out': str(args.out), 'services_touched': False}, indent=2))


if __name__ == '__main__':
    main()
