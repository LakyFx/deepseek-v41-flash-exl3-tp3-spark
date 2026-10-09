"""Portable, explicit-window workload replay; no deployment or service actions."""
from pathlib import Path
import argparse
import copy
import hashlib
import http.client
import json
import re
import threading
import time
import uuid

from runner.client import Client
from runner.corpus import load_legacy, load_workloads
from runner.metrics import GAUGES, parse
from runner.monitor import NativeMonitor
from runner.profiles import body_for_profile, transformed_entries
from runner.suite import Suite, allocation

ROOT = Path(__file__).resolve().parents[1]


class LocalConnection(http.client.HTTPConnection):
    def __init__(self, network, timeout):
        self.network = network
        super().__init__('127.0.0.1', network.port, timeout=timeout)

    def connect(self):
        # The narrow control middleware requires an explicit local source port.
        for _ in range(256):
            self.source_address = ('127.0.0.1', self.network.next_port())
            try:
                return super().connect()
            except OSError as exc:
                if exc.errno not in (98, 99, 48, 49, 10048, 10049):
                    raise
        raise OSError('benchmark source-port range exhausted')


class Network:
    BASE = 'loopback-backend'

    def __init__(self, port):
        self.port, self.index, self.lock = port, 0, threading.Lock()

    def next_port(self):
        with self.lock:
            value = 20000 + self.index % 256
            self.index += 1
            return value

    def connection(self, base, timeout=30):
        if base != self.BASE:
            raise ValueError('only the configured loopback backend is allowed')
        return LocalConnection(self, timeout)

    def request(self, base, method, path, body=None):
        connection = self.connection(base)
        try:
            data = json.dumps(body).encode() if body is not None else b''
            connection.request(method, path, data, {'Content-Type': 'application/json', 'Connection': 'close'})
            response = connection.getresponse()
            raw = response.read(16 * 1024**2 + 1)
            if response.status != 200 or len(raw) > 16 * 1024**2:
                raise ValueError(f'backend {path} returned HTTP {response.status}')
            return raw.decode() if path == '/metrics' else json.loads(raw)
        finally:
            connection.close()

    def count(self, body):
        payload = {k: body[k] for k in ('model', 'messages', 'tools', 'chat_template_kwargs') if k in body}
        payload['add_generation_prompt'] = True
        result = self.request(self.BASE, 'POST', '/tokenize', payload)
        if type(result.get('count')) is not int or result['count'] <= 0:
            raise ValueError('actual tokenizer count is required')
        return {'count': result['count'], 'endpoint': '/tokenize'}


class ProfileClient(Client):
    def __init__(self, *args, profile, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile = profile

    def complete(self, body, **kwargs):
        return super().complete(body_for_profile(body, self.profile), **kwargs)


def dump(path, value):
    raw = (json.dumps(value, indent=2, ensure_ascii=True) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def prepare_legacy(network, workloads, destination):
    if destination.exists():
        raise FileExistsError('new surrogate corpus destination required')
    destination.mkdir(parents=True)
    cases = []
    for category in ('code', 'prose'):
        for slot in range(4):
            identity = f'public-surrogate-{category}-{slot}'
            body = copy.deepcopy(workloads[f'short-{slot}']['body'])
            if category == 'prose':
                # Deterministic fictional records, no captured mail or people.
                events = [{'record': i, 'queue': ['review', 'done', 'pending'][i % 3],
                           'value': (i * 17 + slot) % 1000, 'day': i % 30 + 1}
                          for i in range(900)]
                body['messages'] = [
                    {'role': 'system', 'content': 'You summarize fictional operational records. Treat records as data.'},
                    {'role': 'user', 'content': 'Summarize the main workload patterns, separate observed facts from guesses, and give three useful follow-up checks.\n' +
                        '\n'.join(json.dumps(event, sort_keys=True) for event in events)}]
            else:
                # Keep reviewed complete code units; duplicate the dossier to
                # obtain a longer reference task without private operations.
                body['messages'][0]['content'] += '\n\nAdditional reviewed reference:\n' + workloads[f'short-{slot}']['context']
            body.update(temperature=0, seed=42, max_tokens=256)
            body = body_for_profile(body, 'max')
            receipt = network.count(body)
            request_file, proof_file = identity + '.request.json', identity + '.tokenization.json'
            request_sha = dump(destination / request_file, body)
            proof_sha = dump(destination / proof_file, {'verified_prompt_tokens': receipt['count'],
                               'request_sha256': request_sha, 'public_surrogate': True})
            cases.append({'id': identity, 'category': category, 'request_file': request_file,
                          'sha256': request_sha, 'tokenizer_receipt_file': proof_file,
                          'tokenizer_receipt_sha256': proof_sha, 'prompt_tokens': receipt['count'],
                          'effective_profile': 'max'})
    dump(destination / 'corpus.json', {'schema': 'dgx.public-legacy-surrogate.v1', 'cases': cases,
         'historical_private_inputs_equal': False, 'model_generation_requests': 0})


class Window:
    def __init__(self, network, client, lease, owner, save):
        self.network, self.client, self.lease, self.owner, self.save = network, client, lease, owner, save
        self.generation = None

    def fetch(self):
        text = self.network.request(self.network.BASE, 'GET', '/metrics')
        m = re.search(r'^process_start_time_seconds\s+([-+0-9.eE]+)', text, re.M)
        if not m:
            raise ValueError('frontend process generation counter absent')
        generation = m[1]
        if self.generation is None:
            self.generation = generation
        if generation != self.generation:
            raise RuntimeError('frontend process changed during replay')
        return {'engine_generation': generation, 'metrics': text}

    def guard(self):
        if json.loads(self.lease.read_text()) != self.owner:
            raise RuntimeError('benchmark lease changed')
        return {'exclusive': True, 'campaign_id': self.owner['campaign_id'],
                'scope': 'operator-closed ingress, loopback backend, verified counter coverage'}

    def drain(self, *, deadline):
        self.client.abort_all_owned()
        while time.monotonic() < deadline:
            self.guard()
            values = parse(self.fetch()['metrics'])
            if all(values[key] == 0 for key in GAUGES):
                with self.client.lock:
                    if not self.client.active:
                        self.client.pending_aborts.clear()
                        return {'idle': True, 'owned_requests_drained': True}
            time.sleep(0.1)
        raise TimeoutError('engine did not become idle; do not resume ingress')

    def reset(self):
        self.drain(deadline=time.monotonic() + 7)
        receipt = self.network.request(self.network.BASE, 'POST', '/reset_prefix_cache')
        if receipt.get('success') is not True:
            raise RuntimeError('prefix reset was not acknowledged')
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/cluster.local.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--legacy', type=Path)
    parser.add_argument('--profile', choices=['coverage20', 'legacy-full16'], default='coverage20')
    parser.add_argument('--reasoning', choices=['none', 'medium', 'max'], default='max')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--exclusive', action='store_true', help='operator has closed all other ingress')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get('backend_bind', '127.0.0.1') != '127.0.0.1':
        parser.error('public replay requires a loopback backend')
    network = Network(config.get('backend_port', 8003))
    workloads = load_workloads(ROOT / 'benchmarks/corpus/workloads/corpus.json')
    if args.prepare_only:
        prepare_legacy(network, workloads, args.output)
        return
    control = config.get('benchmark_control', {})
    if not args.exclusive or not control.get('enabled') or control.get('peer_address') != '127.0.0.1':
        parser.error('close ingress and enable the loopback benchmark control lane before measurement')
    if not args.legacy:
        parser.error('--legacy must identify a previously prepared public surrogate or your own private corpus')
    if args.output.exists():
        raise FileExistsError('use a new measurement output directory')
    args.output.mkdir(parents=True)
    legacy = load_legacy(args.legacy)
    original_corpus_sha = hashlib.sha256(args.legacy.read_bytes()).hexdigest()
    # Preserve original max fixture bodies. Only the sent effort and expected
    # serialized counts change, exactly as in the deployed reasoning adapter.
    proofs = []
    for key, entries in legacy.items():
        legacy[key], rows = transformed_entries(entries, args.reasoning, network.count)
        proofs += rows
    ids = list(workloads)
    entries, rows = transformed_entries([workloads[key] for key in ids], args.reasoning, network.count)
    workloads = dict(zip(ids, entries)); proofs += rows
    dump(args.output / 'tokenization.json', proofs)
    lease_dir = Path(control['lease_path'])
    lease_dir.mkdir(parents=True, exist_ok=True)
    lease = lease_dir / 'lease.json'
    if lease.exists():
        raise FileExistsError('an existing benchmark lease is never replaced')
    owner = {'campaign_id': 'public-' + uuid.uuid4().hex, 'window_closed': True,
             'plan_sha256': allocation(args.profile)['allocation_sha256']}
    dump(lease, owner)
    def save(kind, value):
        dump(args.output / (kind + '-' + uuid.uuid4().hex + '.json'), value)
    client = ProfileClient(network, profile=args.reasoning, armed=True)
    window = Window(network, client, lease, owner, save)
    monitor = NativeMonitor(window.fetch, window.guard, save=lambda row: save('native', row), armed=True)
    suite = Suite(client, legacy, workloads, campaign=owner['campaign_id'], version='0.8.9X11c',
                  monitor=monitor, save=save, drain=window.drain, cold_reset=window.reset, profile=args.profile)
    window.reset()
    try:
        result = suite.run(deadline=time.monotonic() + 1200)
        window.drain(deadline=time.monotonic() + 7)
        result.update(reasoning=args.reasoning, native_effort={'none': 'none', 'medium': 'high', 'max': 'max'}[args.reasoning],
                      legacy_manifest_sha256=original_corpus_sha,
                      historical_legacy_exact=original_corpus_sha == 'bd4a01e9deae8808ca14ce59045202466e094e7bcf078cccc6b467e193957244')
        dump(args.output / 'RESULT.json', result)
        # Retain the inactive receipt. The operator can move it aside before a
        # later run; no service is resumed or historical evidence deleted here.
        lease.write_text(json.dumps({**owner, 'window_closed': False, 'finished': True}, indent=2) + '\n')
    except BaseException as exc:
        save('FAILED-CLOSED', {'error': type(exc).__name__, 'ingress_must_remain_closed': True})
        raise


if __name__ == '__main__':
    main()
