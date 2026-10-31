"""Resolve one local TP4 native unit and run the unchanged NVMe companion."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from tensorfold.tp4.launch import HERE, command


def arguments(config, rank, unit):
    _, env = command(config, rank, validate_files=True)
    pid = int(subprocess.check_output(['systemctl', 'show', unit,
              '--property=MainPID', '--value'], text=True).strip())
    if pid <= 1:
        raise RuntimeError('Native unit is not running')
    receipt = json.loads((Path(config['ranks'][rank]['root'])/'BUILD.json').read_bytes())
    name = 'tensorfold-native' if rank == 0 else 'tf-dsv41-lanes'
    return [sys.executable, str(HERE.parent/'nvme_keepalive.py'), '--pid', str(pid),
        '--exe-sha256', receipt['binaries'][name]['sha256'], '--engram-root', env['TF_DS_ENGRAM'],
        '--period-ms', '25', '--active-ms', '2000']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--rank', type=int, choices=range(4), required=True)
    p.add_argument('--unit', required=True)
    a = p.parse_args()
    argv = arguments(json.loads(a.config.read_bytes()), a.rank, a.unit)
    os.execv(argv[0], argv)
