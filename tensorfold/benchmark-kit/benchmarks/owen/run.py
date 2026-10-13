"""Run the frozen Owen common-context subset in an exclusive local window."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks'))
from run import Network, ProfileClient, Window, dump
from owen.replay import run as replay
from runner.monitor import NativeMonitor
from owen.fetch_source import DEST as OWEN_SOURCE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/cluster.local.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclusive', action='store_true')
    parser.add_argument('--source', type=Path, default=OWEN_SOURCE)
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error('first run python3 benchmarks/owen/fetch_source.py')
    config = json.loads(args.config.read_text())
    control = config.get('benchmark_control', {})
    if (not args.exclusive or config.get('backend_bind') != '127.0.0.1'
            or not control.get('enabled') or control.get('peer_address') != '127.0.0.1'):
        parser.error('close other ingress and enable the loopback benchmark control lane')
    if args.output.exists():
        raise FileExistsError('use a fresh result directory')
    args.output.mkdir(parents=True)
    lease = Path(control['lease_path']) / 'lease.json'
    lease.parent.mkdir(parents=True, exist_ok=True)
    fixture = ROOT / 'benchmarks/owen/owen-fixture.json'
    owner = {'campaign_id': 'public-owen-' + uuid.uuid4().hex, 'window_closed': True,
             'plan_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest()}
    dump(lease, owner)  # Exclusive creation, never replace another run's lease.
    def save(kind, value):
        dump(args.output / (kind + '-' + uuid.uuid4().hex + '.json'), value)
    net = Network(config.get('backend_port', 8003), engine=config.get('benchmark_engine', 'vllm'))
    # A syntactically valid URL is required by the pinned urllib Request helper.
    net.BASE = 'http://127.0.0.1:' + str(net.port)
    client = ProfileClient(net, profile='max', armed=True)
    window = Window(net, client, lease, owner, save)
    window.net = net
    window.monitor = NativeMonitor(window.fetch, window.guard,
                                   save=lambda value: save('native', value), armed=True)
    try:
        window.reset()
        # Record counters and reject unexplained request completions per wave.
        result = replay(window, fixture, args.source)
        window.drain(deadline=time.monotonic() + 15)
        result['public_adapter'] = 'loopback control; same pinned timing functions and fixture'
        dump(args.output / 'RESULT.json', result)
        lease.write_text(json.dumps({**owner, 'window_closed': False, 'finished': True}, indent=2) + '\n')
    except BaseException as exc:
        client.abort_all_owned()
        save('FAILED-CLOSED', {'error': type(exc).__name__, 'ingress_must_remain_closed': True})
        raise


if __name__ == '__main__':
    main()
