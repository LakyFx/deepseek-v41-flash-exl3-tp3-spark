"""Check release/source/measurement contracts without loading a GPU model."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.launch import build


class ReleaseTests(unittest.TestCase):
    def test_current_c4_profile_and_replay_are_in_all_launches(self):
        profile = json.loads((ROOT / 'config/ds41f-1.0.0.json').read_bytes())
        self.assertEqual(profile['version'], '1.0.0')
        self.assertEqual(profile['parts'], ['N1', 'N2', 'N3', 'N4', 'N6'])
        config = json.loads((ROOT / 'config/cluster.example.json').read_bytes())
        for i, node in enumerate(config['nodes']):
            node['control_address'] = f'192.0.2.{i+1}'
            node['control_interface'] = 'management'
        config['roce_cidr'] = '192.0.2.0/24'
        for rank in range(3):
            argv = build(config, rank, 'a'*64)
            compilation = json.loads(argv[argv.index('--compilation-config')+1])
            self.assertTrue(compilation['cudagraph_decoder_replay'])
            self.assertEqual(compilation['decoder_replay_cudagraph_capture_sizes'], [128,256,384,512,768,1024])
            self.assertEqual(compilation['decoder_replay_trim_threshold'], 768)
            self.assertIn('CUDA_EXL3_NEXT_PREFILL=1', argv)
            self.assertEqual(argv[argv.index('--kv-cache-memory-bytes')+1], '8074035200')
        dockerfile = (ROOT / 'build/Dockerfile').read_text()
        self.assertIn('PYTHONPATH=/opt/dgx_next/runtime:', dockerfile)

    def test_published_source_hashes_and_all_archival_sources(self):
        for row in json.loads((ROOT / 'runtime/source-provenance.json').read_bytes()):
            self.assertEqual(hashlib.sha256((ROOT / row['file']).read_bytes()).hexdigest(), row['sha256'], row['file'])
        archive = ROOT / 'experiments/20261012'
        baseline = json.loads((archive / 'baseline.json').read_bytes())['files']
        arms = json.loads((archive / 'variants.json').read_bytes())['arms']
        entries = list(baseline.values()) + [v for arm in arms for v in arm['files'].values()]
        unique = {e['file']: e['sha256'] for e in entries}
        for name, expected in unique.items():
            self.assertEqual(hashlib.sha256((archive / name).read_bytes()).hexdigest(), expected, name)
        self.assertEqual(sum(a['measured'] for a in arms), 18)
        self.assertFalse(next(a for a in arms if a['arm']=='N9')['measured'])

    def test_summary_matches_all_envelopes_and_partial_phases(self):
        directory = ROOT / 'results/20261012-host64k-campaign'
        spec = importlib.util.spec_from_file_location('release_summary', directory / 'summarize_results.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = json.loads((directory / 'summary.json').read_bytes())
        for row in rows:
            self.assertEqual(module.extract(row['arm']), row)
        self.assertEqual({r['arm'] for r in rows if r['h_128k_complete']}, {'N6','C4'})
        self.assertTrue(all(r['owen_complete'] and r['curve_complete'] and r['quality'] for r in rows))
        # Private model output is not needed to check throughput arithmetic.
        forbidden = {'assistant', 'messages', 'reasoning_content', 'fixture_calls', 'request_id'}
        def check(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden & value.keys())
                for child in value.values(): check(child)
            elif isinstance(value, list):
                for child in value: check(child)
        for file in directory.glob('result.*.json'):
            check(json.loads(file.read_bytes()))

    def test_owen_fixture_is_the_frozen_measured_input(self):
        raw = (ROOT / 'benchmarks/owen/owen-fixture.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), '40427145003f73d3db195de5e98810fcc6ed294d68872df8028f9a3181fcb367')
        fixture = json.loads(raw)
        self.assertEqual(len(fixture['prefill']), 4)
        self.assertEqual(len(fixture['decode']), 32)


if __name__ == '__main__':
    unittest.main()
