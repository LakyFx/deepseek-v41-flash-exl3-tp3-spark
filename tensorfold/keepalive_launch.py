"""Run the production read-only NVMe helper against an explicitly selected rank.

Resolve the local native PID and build identity; no service or GPU management.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tensorfold import launch


def arguments(config, variant, rank, unit):
    profile = json.loads((launch.HERE/'profiles'/f'{variant}.json').read_bytes())
    if 'nvme_keepalive' not in profile.get('companion_services', {}):
        raise ValueError('Selected profile has no NVMe keepalive companion')
    argv, env = launch.command(config, variant, rank, validate_files=True)
    pid = int(subprocess.check_output(
        ['systemctl', 'show', unit, '--property=MainPID', '--value'], text=True).strip())
    if pid <= 1:
        raise RuntimeError('Native service has no running process')
    root = Path(config['ranks'][rank]['root'])
    receipt = json.loads((root/profile['binary_subdir']/'BUILD.json').read_bytes())
    binary = 'tensorfold-native' if rank == 0 else 'tf-dsv41-lanes'
    settings = profile['companion_services']['nvme_keepalive']
    return [sys.executable, str(launch.HERE/'nvme_keepalive.py'), '--pid', str(pid),
            '--exe-sha256', receipt['binaries'][binary]['sha256'],
            '--engram-root', env['TF_DS_ENGRAM'],
            '--period-ms', str(settings['period_ms']), '--active-ms', str(settings['active_ms'])]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--rank', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--unit', required=True)
    args = parser.parse_args()
    command = arguments(json.loads(args.config.read_bytes()), args.variant, args.rank, args.unit)
    os.execv(command[0], command)
