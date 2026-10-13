"""Private, exact rendered-token evidence before a paired engine campaign.

Only calls /tokenize. No model generation, cache reset or service management.
Fixed tool-history probes check template seams; generated conversations still
require runtime quality evaluation and cannot all be anticipated here.
"""
import copy
import hashlib
import json
import time

from .corpus import load_legacy, load_workloads
from .profiles import body_for_profile

SCHEMA = 'dgx.paired-token-inputs.v1'
OWEN_FIXTURE_SHA = '40427145003f73d3db195de5e98810fcc6ed294d68872df8028f9a3181fcb367'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def collect(frozen_speed, fixtures_path, owen_path):
    legacy_path = frozen_speed / 'corpora/legacy/corpus.json'
    workloads_path = frozen_speed / 'corpora/workload/corpus.json'
    owen_raw = owen_path.read_bytes()
    if hashlib.sha256(owen_raw).hexdigest() != OWEN_FIXTURE_SHA:
        raise ValueError('frozen Owen fixture drift')
    cases = []
    def chat(identity, body):
        cases.append({'id': identity, 'path': '/v1/chat/completions', 'body': body_for_profile(body, 'max')})
    for group, rows in load_legacy(legacy_path).items():
        for row in rows:
            chat('hermes-legacy/' + group + '/' + row['id'], row['body'])
    for name, row in load_workloads(workloads_path).items():
        chat('hermes-workflow/' + name, row['body'])
    owen = json.loads(owen_raw)
    for index, cell in enumerate(owen['decode']):
        for slot, (path, body) in enumerate(cell['payloads']):
            if path != '/v1/chat/completions':
                raise ValueError('unexpected Owen decode endpoint')
            chat(f'owen-decode/{index}/{slot}', body)
    for index, cell in enumerate(owen['prefill']):
        if cell['path'] != '/v1/completions':
            raise ValueError('unexpected Owen prefill endpoint')
        cases.append({'id': f'owen-prefill/{index}', 'path': cell['path'], 'body': copy.deepcopy(cell['body'])})
    quality = json.loads(fixtures_path.read_bytes())
    for case in quality['cases']:
        body = {'model': 'deepseek-v4.1-flash', 'messages': copy.deepcopy(case['messages'])}
        if case.get('tools'):
            body['tools'] = case['tools']
            body['parallel_tool_calls'] = False
        if '{{ARCHIVE}}' in json.dumps(body):
            raise ValueError('quality fixture has not been frozen')
        chat('quality/' + case['id'], body)
        for step_index, step in enumerate(case.get('tool_steps', [])):
            call_id = 'fixture-call-' + str(step_index)
            body['messages'].append({'role': 'assistant', 'content': None,
                'reasoning_content': 'I will use the fixture tool to verify this.',
                'tool_calls': [{'id': call_id, 'type': 'function', 'function': {
                    'name': step['name'], 'arguments': json.dumps(step['arguments'], ensure_ascii=False)}}]})
            body['messages'].append({'role': 'tool', 'tool_call_id': call_id,
                                     'content': json.dumps(step['result'], ensure_ascii=False)})
            chat(f'quality-tool-history/{case["id"]}/{step_index}', body)
    if len({case['id'] for case in cases}) != len(cases):
        raise ValueError('duplicate token-equivalence case')
    sources = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (
        ('hermes-legacy', legacy_path), ('hermes-workloads', workloads_path),
        ('quality', fixtures_path), ('owen', owen_path))}
    return cases, sources


def capture(network, cases, sources, *, arm, generation, pairing_id):
    if not generation or not pairing_id:
        raise ValueError('engine generation and campaign pairing identity required')
    started = time.monotonic()
    rows, memo = [], {}
    for case in cases:
        body_hash = digest(case['body'])
        key = (case['path'], body_hash)
        if key not in memo:
            receipt = network.count(case['body'])
            ids = receipt.get('token_ids')
            if (not isinstance(ids, list) or not ids or type(receipt.get('count')) is not int
                    or len(ids) != receipt['count'] or any(type(x) is not int or x < 0 for x in ids)):
                raise ValueError('actual token IDs required for ' + case['id'])
            memo[key] = {'count': len(ids), 'token_ids': ids, 'tokens_sha256': digest(ids)}
        rows.append({'id': case['id'], 'path': case['path'], 'body_sha256': body_hash, **memo[key]})
    return {'schema': SCHEMA, 'arm': arm, 'engine': network.engine, 'engine_generation': generation,
            'pairing_id': pairing_id, 'captured_at': time.time(), 'sources': sources, 'rows': rows,
            'generation_requests': 0, 'unique_tokenize_requests': len(memo),
            'tokenization_wall_seconds': time.monotonic() - started,
            'scope': 'frozen first inputs plus fixed quality tool-history probes; not every generated continuation'}


def compare(reference, candidate):
    for value in (reference, candidate):
        if value.get('schema') != SCHEMA or value.get('generation_requests') != 0 or not value.get('engine_generation'):
            raise ValueError('invalid token equivalence receipt')
    if reference.get('arm') != 'C4-reference' or reference.get('engine') != 'vllm':
        raise ValueError('fresh C4 reference token receipt required')
    if candidate.get('arm') not in ('TF351', 'TF29') or candidate.get('engine') != 'tensorfold':
        raise ValueError('named TensorFold candidate required')
    if not reference.get('pairing_id') or reference['pairing_id'] != candidate.get('pairing_id'):
        raise ValueError('token receipts belong to different campaigns')
    if reference.get('sources') != candidate.get('sources'):
        raise ValueError('token receipts use different frozen corpora')
    def indexed(value):
        rows = value.get('rows', [])
        index = {r['id']: r for r in rows}
        if not rows or len(index) != len(rows):
            raise ValueError('missing or duplicate token cases')
        for r in rows:
            ids = r.get('token_ids')
            if (not isinstance(ids, list) or not ids or type(r.get('count')) is not int
                    or len(ids) != r['count'] or any(type(x) is not int or x < 0 for x in ids)
                    or digest(ids) != r.get('tokens_sha256')):
                raise ValueError('invalid token sequence evidence: ' + r['id'])
        return index
    left, right = indexed(reference), indexed(candidate)
    if left.keys() != right.keys():
        raise ValueError('different token case coverage')
    mismatches = []
    for identity, a in left.items():
        b = right[identity]
        if a['path'] != b['path'] or a['body_sha256'] != b['body_sha256']:
            raise ValueError('different request input: ' + identity)
        if a['token_ids'] != b['token_ids']:
            mismatches.append({'id': identity, 'reference_count': a['count'], 'candidate_count': b['count']})
    return {'schema': 'dgx.paired-token-comparison.v1', 'equal': not mismatches,
            'cases': len(left), 'mismatches': mismatches, 'pairing_id': reference['pairing_id'],
            'reference_sha256': digest(reference), 'candidate_sha256': digest(candidate)}
