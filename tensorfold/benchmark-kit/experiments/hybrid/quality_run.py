"""Execute frozen Czech tasks only inside an explicitly exclusive benchmark lease.

No service stop/start, weight reload or external tool actions. Tools are fixtures.
Importing this module never contacts the running model.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks'))
from run import Network, ProfileClient, Window, dump
from quality_score import typed_equal, summarize

SAMPLED_IDS = {'translate-negation', 'translate-conditional', 'digest-correction',
               'tools-digest-0', 'code-weighted', 'cs-negation'}


def run_case(case, client, *, seed, temperature, deadline, campaign, cache_policy='salted-case'):
    body = {'model': 'deepseek-v4.1-flash', 'messages': copy.deepcopy(case['messages']),
            'max_tokens': case['max_tokens'], 'temperature': temperature, 'seed': seed,
            'cache_salt': hashlib.sha256((campaign + ':' + case['id']).encode()).hexdigest()}
    if cache_policy == 'stage-reset-shared':
        body.pop('cache_salt')
    elif cache_policy != 'salted-case':
        raise ValueError('unknown quality cache policy')
    if '{{ARCHIVE}}' in json.dumps(body):
        raise ValueError('long fixture was not frozen')
    if case.get('tools'):
        body['tools'] = case['tools']
        body['parallel_tool_calls'] = False
    result = {'case_id': case['id'], 'seed': seed, 'temperature': temperature,
              'message': {}, 'finish_reason': None, 'tool_trace': [], 'requests': [],
              'frozen_input': case.get('frozen_input'), 'category': case['category']}
    started = time.monotonic()
    try:
        for _ in range(4):
            call = client.complete(body, deadline=deadline)
            reply = call['reply']
            message = reply.message()
            result['requests'].append({'request_id': call['request_id'],
                'body_sha256': call['body_sha256'], 'usage': reply.usage_counts(),
                'message': message, 'finish_reason': reply.finish_reason,
                'started_monotonic': call['started_monotonic'], 'ended_monotonic': call['ended_monotonic']})
            result.update(message=message, finish_reason=reply.finish_reason)
            calls = message.get('tool_calls', [])
            if not calls:
                break
            if reply.finish_reason != 'tool_calls' or len(calls) != 1:
                raise ValueError('expected one completed fixture tool call')
            tool = calls[0]
            actual = {'name': tool['function']['name'], 'arguments': json.loads(tool['function']['arguments'])}
            position = len(result['tool_trace'])
            result['tool_trace'].append(actual)
            steps = case.get('tool_steps', [])
            if position >= len(steps):
                raise ValueError('extra or unauthorized tool call')
            expected = steps[position]
            if not typed_equal(actual, {'name': expected['name'], 'arguments': expected['arguments']}):
                raise ValueError('wrong tool name, order, arguments or types')
            body['messages'].append(message)
            body['messages'].append({'role': 'tool', 'tool_call_id': tool['id'],
                                     'content': json.dumps(expected['result'], ensure_ascii=False)})
        else:
            raise ValueError('tool-turn budget exhausted')
    except (OSError, TimeoutError, ValueError, KeyError, TypeError) as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
    result['elapsed_seconds'] = time.monotonic() - started
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=['C4-reference', 'H1', 'H2', 'H3', 'H4', 'TF351', 'TF29'], required=True)
    p.add_argument('--seed', type=int, choices=[42, 2026, 1708], default=42)
    p.add_argument('--sampled', action='store_true')
    p.add_argument('--exclusive', action='store_true')
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    fixtures = json.loads(args.fixtures.read_text(encoding='utf-8'))
    if args.sampled:
        fixtures['cases'] = [c for c in fixtures['cases'] if c['id'] in SAMPLED_IDS]
    config = json.loads(args.config.read_text(encoding='utf-8'))
    cache_policy = config.get('quality_cache_policy', 'salted-case')
    if cache_policy not in ('salted-case', 'stage-reset-shared'):
        p.error('unknown quality_cache_policy')
    control = config.get('benchmark_control', {})
    if not args.run:
        print(json.dumps({'action': 'PLAN_ONLY', 'arm': args.arm, 'cases': len(fixtures['cases']),
                          'seed': args.seed, 'reasoning': 'max', 'temperature': 0.6 if args.sampled else 0,
                          'generation_requests': 0}, indent=2))
        return
    if (not args.exclusive or config.get('backend_bind') != '127.0.0.1'
            or not control.get('enabled') or control.get('peer_address') != '127.0.0.1'):
        p.error('exclusive loopback control window required')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    lease = Path(control['lease_path']) / 'lease.json'
    lease.parent.mkdir(parents=True, exist_ok=True)
    owner = {'campaign_id': 'hybrid-quality-' + uuid.uuid4().hex, 'window_closed': True,
             'plan_sha256': hashlib.sha256(args.fixtures.read_bytes()).hexdigest()}
    dump(lease, owner)
    def save(kind, value):
        dump(args.output / (kind + '-' + uuid.uuid4().hex + '.json'), value)
    net = Network(config.get('backend_port', 8003), engine=config.get('benchmark_engine', 'vllm'))
    client = ProfileClient(net, profile='max', armed=True)
    window = Window(net, client, lease, owner, save)
    results = []
    try:
        window.reset()
        if cache_policy == 'salted-case':
            net.validate_body({'cache_salt': 'capability-check-only'})
        # Actual rendered chat counts, independent of offline content counts.
        counts = []
        for case in fixtures['cases']:
            from runner.profiles import body_for_profile
            request = body_for_profile({'model': 'deepseek-v4.1-flash', 'messages': case['messages'],
                                        **({'tools': case['tools']} if case.get('tools') else {})}, 'max')
            counts.append({'case_id': case['id'], **net.count(request)})
        dump(args.output / 'TOKENIZATION.json', counts)
        deadline = time.monotonic() + 1800
        with ThreadPoolExecutor(max_workers=3) as pool:
            tasks = {pool.submit(run_case, case, client, seed=args.seed,
                temperature=0.6 if args.sampled else 0, deadline=deadline,
                campaign='hybrid-quality-fixed-seed-' + str(args.seed), cache_policy=cache_policy): case['id'] for case in fixtures['cases']}
            for task in as_completed(tasks):
                window.guard()
                window.fetch()
                result = task.result()
                dump(args.output / (result['case_id'] + '.json'), result)
                results.append(result)
        window.drain(deadline=time.monotonic() + 15)
        results.sort(key=lambda r: r['case_id'])
        dump(args.output / 'RESPONSES.json', results)
        dump(args.output / 'RESULT.json', {'arm': args.arm, 'seed': args.seed,
             'fixture_sha256': owner['plan_sha256'], 'sampled': args.sampled,
             'score': summarize(fixtures, results), 'generation': window.generation,
             'cache_policy': cache_policy, 'metric_backend': net.engine,
             'quality_speed_is_not_Hermes_or_Owen': True})
        lease.write_text(json.dumps({**owner, 'window_closed': False, 'finished': True}, indent=2), encoding='utf-8')
    except BaseException as exc:
        client.abort_all_owned()
        save('FAILED-CLOSED', {'error': str(exc), 'ingress_must_remain_closed': True})
        raise


if __name__ == '__main__':
    main()
