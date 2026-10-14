"""Recalculate saved Hermes production aggregates; no network or model calls."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def audit():
    folder = ROOT / 'results/20261014-hermes-real-workload'
    raw = (folder / 'observations.json').read_bytes()
    observations = json.loads(raw)
    summary = json.loads((folder / 'summary.json').read_bytes())
    assert hashlib.sha256(raw).hexdigest() == summary['observations_sha256']
    x = observations['observer_windows'][0]
    c4, tf = observations['windows']
    assert c4['label'].startswith('C4') and tf['label'].startswith('TF351')
    for row in summary['rows']:
        c = row['concurrency']
        for key, window, tps, secs, work in [
            ('X11c', x, 'output_active_tps', 'output_positive_seconds', 'output_tokens'),
            ('C4', c4, 'avg_aggregate_decode_tps', 'seconds', 'observed_decode_tokens'),
            ('TF351', tf, 'avg_aggregate_decode_tps', 'seconds', 'observed_decode_tokens'),
        ]:
            source = next((r for r in window['by_concurrency'] if r['concurrency'] == c), None)
            actual = row[key]
            assert (source is None) == (actual is None), (key, c)
            if source is None:
                continue
            assert source[secs] > 0 and actual['samples'] == source['samples']
            assert math.isclose(actual['aggregate_tok_s'], source[work] / source[secs], rel_tol=1e-10)
            assert math.isclose(actual['aggregate_tok_s'], source[tps], rel_tol=1e-10)
            assert actual['active_seconds'] == source[secs]
        base, new = row['C4'], row['TF351']
        if base is None or new is None:
            assert row['TF351_vs_C4_percent'] is None
        else:
            expected = (new['aggregate_tok_s'] / base['aggregate_tok_s'] - 1) * 100
            assert math.isclose(row['TF351_vs_C4_percent'], expected, rel_tol=1e-10)
    assert summary['TF351_pool_tokens'] == 10000000
    assert summary['TF351_window_precedes_hc_prefill_repair'] is True
    return {'status': 'PASS', 'kind': summary['kind'], 'rows': summary['rows']}


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
