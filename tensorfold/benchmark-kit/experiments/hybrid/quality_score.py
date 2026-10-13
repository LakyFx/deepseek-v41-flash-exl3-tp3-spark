"""Strict mechanical checks plus explicit pending independent language review."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re


def typed_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(typed_equal(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(typed_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def score(case, response):
    failures = []
    message = response.get('message', {})
    content = message.get('content') or ''
    if response.get('error'):
        failures.append('request_error')
    if response.get('finish_reason') != 'stop':
        failures.append('not_completed_stop')
    if not isinstance(content, str) or not content.strip():
        failures.append('empty_visible_answer')
        content = ''
    if message.get('tool_calls'):
        failures.append('unprocessed_final_tool_call')
    if 'exact_text' in case and content.strip() != case['exact_text']:
        failures.append('exact_instruction_answer')
    for pattern in case.get('required_patterns', []):
        if re.search(pattern, content, re.IGNORECASE) is None:
            failures.append('missing_required_value:' + pattern)
    if 'exact_json' in case:
        try:
            parsed = json.loads(content)
            if not isinstance(parsed, dict) or set(parsed) != set(case['require_json_keys']):
                raise ValueError('keys')
            if not isinstance(parsed.get('summary'), str) or not parsed['summary'].strip():
                raise ValueError('summary')
            for key, expected in case['exact_json'].items():
                if not typed_equal(parsed.get(key), expected):
                    failures.append('wrong_facts:' + key)
        except (ValueError, TypeError):
            failures.append('invalid_json_or_schema')
    if 'tool_steps' in case:
        observed = response.get('tool_trace', [])
        expected = [{'name': x['name'], 'arguments': x['arguments']} for x in case['tool_steps']]
        if not typed_equal(observed, expected):
            failures.append('tool_sequence_or_arguments')
    pending = []
    if 'code_checks' in case:
        # External sandbox receipts are validated by the runner before joining.
        # A model-provided success claim is never a test result.
        pending.append('isolated_code_execution')
    if case.get('editorial_required'):
        pending.append('blind_Czech_editorial_review')
    return {'case_id': case['id'], 'category': case['category'],
            'mechanical_pass': not failures, 'failures': failures, 'pending': pending,
            'quality_status': 'FAIL' if failures else 'PENDING' if pending else 'PASS'}


def summarize(fixtures, responses):
    indexed = {}
    for response in responses:
        name = response['case_id']
        if name in indexed:
            raise ValueError('duplicate case result')
        indexed[name] = response
    expected_ids = {case['id'] for case in fixtures['cases']}
    if set(indexed) - expected_ids:
        raise ValueError('unknown result case')
    rows = []
    for case in fixtures['cases']:
        if case['id'] not in indexed:
            rows.append({'case_id': case['id'], 'category': case['category'],
                         'mechanical_pass': False, 'quality_status': 'INCOMPLETE', 'failures': ['missing_result'], 'pending': []})
        else:
            rows.append(score(case, indexed[case['id']]))
    return {'cases': rows, 'mechanical_pass': all(x['mechanical_pass'] for x in rows),
            'quality_pass': all(x['quality_status'] == 'PASS' for x in rows),
            'pending_review': sum(bool(x['pending']) for x in rows),
            'note': 'A mechanical pass alone is not evidence of unchanged Czech quality.'}


def blind_pairs(fixtures, baseline, candidate, seed=741):
    """Return anonymous review packet and separate, private decoding key."""
    left = {x['case_id']: x for x in baseline}
    right = {x['case_id']: x for x in candidate}
    if len(left) != len(baseline) or len(right) != len(candidate):
        raise ValueError('duplicate pair')
    rng = random.Random(seed)
    packet, key = [], []
    for case in fixtures['cases']:
        if not case.get('editorial_required'):
            continue
        name = case['id']
        if name not in left or name not in right:
            raise ValueError('missing editorial pair: ' + name)
        for result in (left[name], right[name]):
            if result.get('finish_reason') != 'stop' or result.get('error'):
                raise ValueError('incomplete editorial pair: ' + name)
        order = ['baseline', 'candidate']
        rng.shuffle(order)
        texts = {'baseline': left[name]['message']['content'], 'candidate': right[name]['message']['content']}
        pair_id = hashlib.sha256((str(seed) + ':' + name).encode()).hexdigest()[:16]
        packet.append({'pair_id': pair_id, 'task': case['messages'],
                       'reference': case.get('reference', case.get('reference_facts', case.get('exact_json'))),
                       'rubric': case.get('rubric'), 'A': texts[order[0]], 'B': texts[order[1]],
                       'required_review': {'winner': 'A|B|tie', 'critical_error_A': 'boolean',
                                           'critical_error_B': 'boolean', 'evidence': 'quote and explain'}})
        key.append({'pair_id': pair_id, 'case_id': name, 'A': order[0], 'B': order[1]})
    return packet, key


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--responses', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.fixtures.read_text(encoding='utf-8')),
                       json.loads(args.responses.read_text(encoding='utf-8')))
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
