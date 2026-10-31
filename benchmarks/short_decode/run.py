"""Portable replay of the 31 October TP4 short decode screen; stdlib only.

No service management. Call only during an operator-established exclusive window.
Responses stay in the chosen local output directory, never in public exports.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import urllib.parse
import urllib.request


TOPICS = [
    'bounded job queue', 'immutable event log', 'prefix cache', 'lease expiry',
    'retry budget', 'stream cancellation', 'incremental parser',
    'graph buffer lifetime', 'snapshot retention', 'fair scheduler',
    'content digest', 'network reconnect',
]
CONCURRENCIES = (1, 3, 6, 12)


def fixture(kind, index):
    """Exact original authored software prompt and sampling body."""
    topic = TOPICS[index]
    if kind == 'code':
        task = f'Write a complete Python module implementing a {topic}. Include type hints, detailed functions, edge cases and runnable usage examples. Aim for at least 180 lines of code. Continue the module until the output limit.'
    elif kind == 'prose':
        task = f'Write a detailed technical tutorial about a {topic} for a distributed inference service. Explain architecture, failure handling, concurrency and concrete examples. Aim for at least 1800 words and continue until the output limit.'
    else:
        raise ValueError('kind must be code or prose')
    return dict(model='deepseek-v4.1-flash', messages=[
        dict(role='system', content='You are a precise software engineer. Answer directly in English. Different requests are independent.'),
        dict(role='user', content=task)], temperature=0, top_p=1, seed=42,
        max_tokens=512, stream=False, enable_thinking=False,
        chat_template_kwargs=dict(enable_thinking=False))


def body_bytes(body):
    return json.dumps(body, separators=(',', ':')).encode()


def request_metrics(response):
    """Allowlist only numeric measurements; no answers or runtime identifiers."""
    timing, usage = response['tensorfold'], response['usage']
    spec = response['speculative']
    if timing['enable_thinking'] or timing['sampling'] != 'greedy':
        raise ValueError('screen requires thinking OFF and greedy sampling')
    if usage['completion_tokens'] != 512:
        raise ValueError('screen requires exactly 512 output tokens per request')
    rate = float(timing['tokens_per_second'])
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('invalid native decode rate')
    return dict(
        prompt_tokens=usage['prompt_tokens'],
        cached_tokens=usage.get('prompt_tokens_details', {}).get('cached_tokens', 0),
        completion_tokens=usage['completion_tokens'],
        native_decode_tps=rate,
        native_decode_seconds=timing['seconds'],
        native_prefill_seconds=timing['prefill_seconds'],
        post_prefill_first_token_seconds=timing.get('post_prefill_first_token_seconds'),
        accepted=spec['accepted'], drafted=spec['drafted'],
    )


def summarize(cell):
    requests = cell['requests']
    rows = [request_metrics(r['response']) for r in requests]
    if len(rows) != cell['concurrency'] or cell['elapsed_seconds'] <= 0:
        raise ValueError('incomplete cell')
    tokens = sum(r['completion_tokens'] for r in rows)
    drafted = sum(r['drafted'] for r in rows)
    accepted = sum(r['accepted'] for r in rows)
    return dict(
        run=cell['run'], concurrency=cell['concurrency'], kind=cell['kind'],
        requests=len(rows), completion_tokens=tokens,
        sum_request_decode_tps=sum(r['native_decode_tps'] for r in rows),
        median_request_decode_tps=statistics.median(r['native_decode_tps'] for r in rows),
        whole_wave_seconds=cell['elapsed_seconds'],
        whole_wave_tps=tokens / cell['elapsed_seconds'],
        median_prefill_seconds=statistics.median(r['native_prefill_seconds'] for r in rows),
        accepted=accepted, drafted=drafted,
        acceptance_percent=100 * accepted / drafted if drafted else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8003')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclusive', action='store_true',
                        help='Confirm other ingress is CLOSED and backend is idle')
    args = parser.parse_args()
    url = urllib.parse.urlsplit(args.base_url)
    if not args.exclusive or url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost', '::1'):
        parser.error('requires --exclusive and an operator-managed HTTP loopback backend')
    if url.username or url.password or url.query or url.fragment or url.path not in ('', '/'):
        parser.error('base URL must contain only scheme, loopback host and port')
    base = args.base_url.rstrip('/')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('DS41F_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['DS41F_API_KEY']

    def get_health():
        req = urllib.request.Request(base + '/health', headers=headers)
        with opener.open(req, timeout=20) as stream:
            health = json.load(stream)
        live = health['live']
        if health['status'] != 'ok' or health['warming'] or live['connections'] or live['waiting']:
            raise RuntimeError('native backend must be healthy and idle before each cell')

    def post(body):
        raw = body_bytes(body)
        req = urllib.request.Request(base + '/v1/chat/completions', data=raw, headers=headers)
        start = time.monotonic()
        with opener.open(req, timeout=240) as stream:
            response = json.load(stream)
        request_metrics(response)
        return dict(body_sha256=hashlib.sha256(raw).hexdigest(),
                    elapsed_seconds=time.monotonic() - start, response=response)

    # Directory creation is exclusive: never overwrite an earlier run.
    get_health()
    args.output.mkdir(parents=True, exist_ok=False)
    cells = []
    try:
        for run in (1, 2):
            for concurrency in CONCURRENCIES:
                for kind in ('code', 'prose'):
                    get_health()
                    start = time.monotonic()
                    with ThreadPoolExecutor(concurrency) as workers:
                        requests = list(workers.map(post, [fixture(kind, i) for i in range(concurrency)]))
                    cell = dict(run=run, concurrency=concurrency, kind=kind,
                                elapsed_seconds=time.monotonic() - start, requests=requests)
                    name = f'run{run}-C{concurrency:02}-{kind}.json'
                    (args.output / name).write_text(json.dumps(cell, indent=2) + '\n', encoding='utf-8')
                    cells.append(summarize(cell))
                    (args.output / 'PROGRESS.json').write_text(json.dumps(cells, indent=2) + '\n', encoding='utf-8')
                    print(json.dumps(cells[-1]), flush=True)
        result = dict(status='PASS', selected_run=2, cells=cells,
                      scope='Short native decode screen; not SparkDash or heavy workflow')
        (args.output / 'SUMMARY.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    except BaseException:
        (args.output / 'FAILED.txt').write_text('Run did not complete; retain existing local evidence.\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
