"""Recalculate published comparisons from retained receipts; never call a model."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/20261013-tensorfold'
INVENTORY = json.loads((RESULTS / 'evidence-manifest.json').read_bytes())
INPUTS = {}


def read(relative):
    path = ROOT / relative
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    evidence_prefix = 'results/20261013-tensorfold/evidence/'
    if relative.startswith(evidence_prefix):
        assert digest == INVENTORY[relative[len(evidence_prefix):]]['public_sha256'], relative
    INPUTS[relative] = digest
    return json.loads(raw)


def rate(value, denominator):
    assert denominator > 0
    return value / denominator


def workflow(result, identifier, published_wall):
    phase = next(p for p in result['phases'] if p['id'] == identifier)
    data = phase['result']
    metrics = data['native_metrics']
    deltas = metrics['native_deltas']
    assert metrics['coverage_verified'] and metrics['idle_before'] and metrics['idle_after']
    assert all(t['correct'] and t['finished'] for t in data['tasks'])
    usage = {k: sum(request['usage'][k] for t in data['tasks'] for request in t['requests'])
             for k in ('prompt_tokens', 'completion_tokens', 'cached_tokens', 'computed_prompt_tokens')}
    assert usage['completion_tokens'] == deltas['request_generation_tokens_sum']
    assert abs(published_wall - data['cohort_wall_seconds']) < 0.02
    decode = rate(usage['completion_tokens'], deltas['request_decode_time_seconds_sum'])
    assert abs(decode - metrics['weighted_request_decode_tokens_per_second']) < 1e-8
    return dict(wall_seconds=published_wall, cohort_wall_seconds=data['cohort_wall_seconds'],
                correct_tasks=len(data['tasks']), usage=usage, decode_tokens_per_second=decode,
                decode_engine_seconds=deltas['request_decode_time_seconds_sum'],
                decode_scope='output includes reasoning and tools; summed per-request engine time',
                draft_acceptance=metrics['draft_acceptance_fraction'], coverage_verified=True)


def load_probe(result, identifier):
    phase = next(p for p in result['phases'] if p['id'] == identifier)
    data = phase['result']
    assert phase['complete'] and data['cold_cache_verified'] and data['hot_cache_verified']
    cold, hot = data['prime'], data['hot']
    for part in (cold, hot):
        assert part['usage_complete'] and part['native_metrics']['coverage_verified']
        assert sum(r['usage']['completion_tokens'] for r in part['rows']) == part['native_metrics']['native_deltas']['request_generation_tokens_sum']
    counts = [r['usage'] for r in cold['rows']]
    assert all(u['cached_tokens'] == 0 and u['computed_prompt_tokens'] == u['prompt_tokens'] for u in counts)
    computed = sum(u['computed_prompt_tokens'] for u in counts)
    seconds = cold['native_metrics']['native_deltas']['request_prefill_time_seconds_sum']
    output = sum(r['usage']['completion_tokens'] for r in hot['rows'])
    hot_rate = rate(output, hot['wall_seconds'])
    assert abs(hot_rate - hot['aggregate_output_tokens_per_wall_second']) < 1e-8
    return dict(stage='full V3; not rerun in final V7',
                cold_prompt_token_counts=[u['prompt_tokens'] for u in counts],
                cold_computed_tokens=computed, cold_cached_tokens=0, cold_engine_prefill_seconds=seconds,
                cold_computed_tokens_per_second=rate(computed, seconds),
                hot_output_tokens=output, hot_wall_seconds=hot['wall_seconds'],
                hot_aggregate_tokens_per_second=hot_rate,
                hot_scope='bounded output probe, not a scored completed agent workflow',
                hot_output_caps=[r['usage']['completion_tokens'] for r in hot['rows']])


def owen_audit(path):
    data = read(path)
    cells = [r for r in data['rows'] if r.get('sample') in ('sample1', 'sample2', 'sample3')]
    rows = [r for c in cells for r in c['requests']]
    assert rows
    assert all(r['ok'] and r['finish_reason'] == 'length' and r['completion_tokens'] == 256
               and r['content_chars'] == 0 and r['reasoning_chars'] > 0 for r in rows)
    for cell in cells:
        requests = cell['requests']
        first = min(r['first_s'] for r in requests)
        last = max(r['last_s'] for r in requests)
        recalculated = sum(r['completion_tokens'] - 1 for r in requests) / (last - first)
        assert abs(recalculated - cell['aggregate_decode_tps']) < 0.001
    return dict(measured_requests=len(rows), reasoning_only_requests=len(rows),
                final_content_chars=0, max_output_tokens=256, all_finish_reason='length',
                timing_recalculated=True, measured_cells=[c['point'] for c in cells])


def legacy_screen(result):
    data = next(p['result'] for p in result['phases'] if p['id'] == 'legacy-screen6')
    assert data['complete'] and not data['full16'] and len(data['cells']) == 6
    cells = []
    for cell in data['cells']:
        metrics = cell['native_metrics']
        assert cell['cache_verified'] and cell['usage_complete'] and metrics['coverage_verified']
        output = sum(r['usage']['completion_tokens'] for r in cell['rows'])
        assert output == metrics['native_deltas']['request_generation_tokens_sum']
        cells.append(dict(phase=cell['phase'], category=cell['category'], concurrency=cell['concurrency'],
                          weighted_request_decode_tokens_per_second=metrics['weighted_request_decode_tokens_per_second'],
                          aggregate_output_tokens_per_wall_second=rate(output, cell['wall_seconds']),
                          output_tokens=output, wall_seconds=cell['wall_seconds'], cache_verified=True,
                          scope='256-token bounded throughput probe, not scored task completion'))
    return cells


def mixed_probe(result):
    data = next(p['result'] for p in result['phases'] if p['id'] == 'mixed')
    assert data['cache_regime_verified'] and data['leaders']['usage_complete'] and data['injected']['complete']
    assert data['native_metrics']['coverage_verified']
    return dict(stage='V3', wall_seconds=data['wall_seconds'],
                leader_aggregate_output_tokens_per_wall_second=data['leaders']['aggregate_output_tokens_per_wall_second'],
                injected_cold_ttft_seconds=data['injected']['ttft_seconds'],
                injected_cold_wall_seconds=data['injected']['wall_seconds'],
                weighted_request_decode_tokens_per_second=data['native_metrics']['weighted_request_decode_tokens_per_second'],
                scope='three hot leaders plus one cold32K arrival; bounded load probe, not scored task completion',
                cache_verified=True)


def main():
    summary = read('results/20261013-tensorfold/summary.json')
    v3, final, result = {}, {}, {}
    for arm in ('C4-reference', 'TF351', 'TF29'):
        base = f'results/20261013-tensorfold/evidence/v3/results/{arm}'
        v3[arm] = read(base + '/hermes/RESULT.json')
        saved = next(x for x in summary['v3'] if x['arm'] == arm)
        if arm == 'C4-reference':
            source, workflows = v3[arm], saved['workflows']
        else:
            stage = 'v7' if arm == 'TF351' else 'v7r2'
            source = read(f'results/20261013-tensorfold/evidence/{stage}/{arm}-workflow_delta_probe/RESULT.json')
            workflows = next(x for x in summary['final'] if x['arm'] == arm)['workflows']
        result[arm] = dict(
            workflows={w['id']: workflow(source, w['id'], w.get('wall_seconds', w.get('wall'))) for w in workflows},
            load_probes={name: load_probe(v3[arm], name) for name in ('64K-C1', '128K-C3', '8K-C8')},
            legacy_screen_v3=legacy_screen(v3[arm]), mixed_v3=mixed_probe(v3[arm]))
        files = sorted((ROOT / base / 'owen').glob('OWEN-RESULT-*.json'))
        assert len(files) == 1
        result[arm]['owen_v3_audit'] = owen_audit(files[0].relative_to(ROOT).as_posix())
        if arm != 'C4-reference':
            stage = 'v7r2'
            result[arm]['owen_final_audit'] = owen_audit(f'results/20261013-tensorfold/evidence/{stage}/{arm}-owen_delta_probe/RESULT.json')
    for name in ('64K-C1', '128K-C3', '8K-C8'):
        counts = [result[a]['load_probes'][name]['cold_prompt_token_counts'] for a in result]
        assert counts[0] == counts[1] == counts[2]
    c4 = result['C4-reference']
    changes = {}
    for arm in ('TF351', 'TF29'):
        candidate = result[arm]
        rows = {}
        for concurrency in ('C1', 'C3'):
            baseline = c4['workflows']['workflow-' + concurrency]
            value = candidate['workflows']['workflow-' + concurrency]
            rows['workflow_' + concurrency + '_wall_reduction_percent'] = (1 - value['wall_seconds'] / baseline['wall_seconds']) * 100
            rows['workflow_' + concurrency + '_decode_increase_percent'] = (value['decode_tokens_per_second'] / baseline['decode_tokens_per_second'] - 1) * 100
        for name in ('64K-C1', '128K-C3'):
            rows['cold_' + name + '_rate_increase_percent'] = (candidate['load_probes'][name]['cold_computed_tokens_per_second'] / c4['load_probes'][name]['cold_computed_tokens_per_second'] - 1) * 100
        rows['hot_8K-C8_rate_increase_percent'] = (candidate['load_probes']['8K-C8']['hot_aggregate_tokens_per_second'] / c4['load_probes']['8K-C8']['hot_aggregate_tokens_per_second'] - 1) * 100
        changes[arm] = rows
    envelope = dict(schema='ds41f.saved-receipt-comparison-audit.v1',
                    audit_date='2026-10-14', inference_requests_sent=0, model_or_production_changed=False,
                    baseline='fresh C4 / 1.0.0 reference from the TensorFold comparison',
                    speed_pool_tokens=3300000, reasoning='max',
                    scope='final workflow C1/C3 and V3 load probes; stages kept separate; no new benchmark',
                    arms=result, changes_vs_c4=changes, source_public_sha256=INPUTS)
    target = RESULTS / 'comparison-audit.json'
    target.write_text(json.dumps(envelope, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
    print(json.dumps(dict(audit_pass=True, inference_requests_sent=0,
                         owen_v3_requests=sum(x['owen_v3_audit']['measured_requests'] for x in result.values()),
                         owen_final_requests=sum(x.get('owen_final_audit', {}).get('measured_requests', 0) for x in result.values()),
                         changes_vs_c4=changes), ensure_ascii=True))


if __name__ == '__main__':
    main()
