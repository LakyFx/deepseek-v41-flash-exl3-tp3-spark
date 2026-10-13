"""Prepare this rank's eight hash-table ranges per Engram layer in a NEW local directory."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--kit', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--rank', choices=range(3), type=int, required=True)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    spec = json.loads((args.kit / 'engram.json').read_text())
    cols = (spec['ngram'] - 1) * spec['heads']
    if cols != 24:
        raise ValueError('Expected the qualified 24-column Engram geometry')
    ranges = []
    for index, layer in enumerate(spec['layers']):
        start = args.rank * 8
        stop = start + 8
        lo = spec['offsets'][index][start]
        hi = spec['offsets'][index][stop] if stop < cols else sum(spec['primes'][index])
        ranges.append(f'{layer}:{lo}:{hi}')
    helper = Path(__file__).resolve().parents[1] / 'tools/engram_local.py'
    argv = [sys.executable, str(helper), str(args.source), str(args.out), *ranges, '--mbps=128']
    if args.write:
        if args.out.exists():
            raise FileExistsError('Do not overwrite an existing Engram copy')
        subprocess.run(argv, check=True)
    else:
        print(json.dumps({'rank': args.rank, 'owned_rows': ranges, 'argv': argv, 'written': False}, indent=2))


if __name__ == '__main__':
    main()
