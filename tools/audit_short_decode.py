"""Audit the published TP4 short-screen numeric export; no network/model calls."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]


def audit():
    spec = importlib.util.spec_from_file_location('short_decode', ROOT/'benchmarks/short_decode/run.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    data = ROOT/'results/20261031-4.0.0'
    summary = json.loads((data/'SHORT-DECODE.json').read_bytes())
    raw = (data/'SHORT-DECODE-REQUESTS.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == summary['request_records_sha256']
    rows = json.loads(raw)
    assert len(rows) == 88
    assert sum(r['completion_tokens'] for r in rows) == 45056
    cells = summary['cells']
    expected = {(run, c, kind) for run in (1, 2) for c in runner.CONCURRENCIES for kind in ('code', 'prose')}
    assert {(c['run'], c['concurrency'], c['kind']) for c in cells} == expected
    assert len(cells) == len(expected) == 16
    for cell in cells:
        key = (cell['run'], cell['concurrency'], cell['kind'])
        selected = [r for r in rows if (r['run'], r['concurrency'], r['kind']) == key]
        assert {r['request_index'] for r in selected} == set(range(cell['concurrency']))
        assert len(selected) == cell['concurrency']
        assert all(r['completion_tokens'] == 512 for r in selected)
        for row in selected:
            body = runner.body_bytes(runner.fixture(row['kind'], row['request_index']))
            assert hashlib.sha256(body).hexdigest() == row['body_sha256']
        values = dict(
            sum_request_decode_tps=sum(r['native_decode_tps'] for r in selected),
            median_request_decode_tps=statistics.median(r['native_decode_tps'] for r in selected),
            median_prefill_seconds=statistics.median(r['native_prefill_seconds'] for r in selected),
            whole_wave_tps=sum(r['completion_tokens'] for r in selected)/cell['whole_wave_seconds'],
            acceptance_percent=100*sum(r['accepted'] for r in selected)/sum(r['drafted'] for r in selected),
        )
        for name, value in values.items():
            assert math.isclose(value, cell[name], rel_tol=1e-12), (key, name)
        assert cell['accepted'] == sum(r['accepted'] for r in selected)
        assert cell['drafted'] == sum(r['drafted'] for r in selected)
    acceptance = 100*sum(r['accepted'] for r in rows if r['run'] == 2)/sum(r['drafted'] for r in rows if r['run'] == 2)
    assert math.isclose(acceptance, summary['run2_acceptance_percent'], rel_tol=1e-12)
    parity = json.loads((data/'PRODUCTION-PARITY.json').read_bytes())
    assert parity['native'] == summary['native_binaries_sha256']
    assert summary['measured_profile']['parallel'] == 12
    assert summary['current_profile']['parallel'] == parity['parallel'] == 16
    return dict(status='PASS', cells=16, requests=len(rows), selected_run=2,
                run2_acceptance_percent=acceptance)


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
