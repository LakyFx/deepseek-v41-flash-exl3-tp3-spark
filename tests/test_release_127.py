"""Release reconstruction, live-setting parity and numerical evidence checks; CPU only."""
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch as mock_patch

from tensorfold import launch, keepalive_launch
from tensorfold.variants import patch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'results/20261028-1.2.7'


class Release127(unittest.TestCase):
    def test_every_trial_reconstructs_exact_source_from_shipped_archive(self):
        with tarfile.open(ROOT/'tensorfold/artifacts/tensorfold-source.tar.gz') as archive:
            base = {m.name: patch.digest(archive.extractfile(m).read())
                    for m in archive if m.isfile()}
        for name in ('x1', 'x2', 'x3', 'x4', 'v127'):
            with self.subTest(recipe=name):
                spec, _ = patch.recipe(name)
                actual = dict(base)
                for rel, record in spec['overlays'].items():
                    patch.relative(rel)
                    data = (patch.HERE/record['object']).read_bytes()
                    self.assertEqual(patch.digest(data), record['sha256'])
                    actual[rel] = patch.digest(data)
                for rel in spec['remove']:
                    actual.pop(rel)
                self.assertEqual(actual, spec['source_files'])
        x4, _ = patch.recipe('x4')
        final, _ = patch.recipe('v127')
        self.assertEqual(x4['source_files'], final['source_files'])
        self.assertEqual(x4['recorded_binaries'], final['recorded_binaries'])

    def test_all_three_launcher_ranks_match_live_behavior_flags(self):
        config = json.loads((ROOT/'tensorfold/cluster.example.json').read_bytes())
        parity = json.loads((DATA/'PRODUCTION-PARITY.json').read_bytes())
        final, _ = patch.recipe('v127')
        for rank in parity['ranks']:
            argv, env = launch.command(config, '1.2.7', rank['rank'])
            for key, value in rank['runtime_environment'].items():
                self.assertEqual(env[key], value, key)
            binary = 'tensorfold-native' if rank['rank'] == 0 else 'tf-dsv41-lanes'
            self.assertEqual(rank['binary_sha256'], final['recorded_binaries'][binary]['sha256'])
            self.assertIn('bin-1.2.7', argv[0])
        self.assertEqual(parity['vision_output_budget']['thinking_budget'], 8192)
        self.assertEqual(parity['vision_output_budget']['final_answer_tokens'], 32768)

    def test_trials_are_independent_and_keep_expected_policy_flags(self):
        config = json.loads((ROOT/'tensorfold/cluster.example.json').read_bytes())
        trees = set()
        for n in range(1, 5):
            spec, _ = patch.recipe(f'x{n}')
            trees.add(spec['source_tree_sha256'])
            _, env = launch.command(config, f'1.2.6x{n}', 0)
            self.assertEqual(env['TF_CACHE_BALANCED'], '1')
            self.assertEqual(env.get('TF_KV_CONTINUATION_DONOR'), '1' if n >= 2 else None)
            self.assertEqual(env.get('TF_CACHE_ADAPTIVE_PREFIX'), '1' if n >= 3 else None)
            receipt = json.loads((DATA/'history'/f'x{n}.json').read_bytes())
            for group in receipt['checks'].values():
                for row in group['checks'].values():
                    self.assertEqual(row.get('rc'), 0)
        self.assertEqual(len(trees), 4)

    def test_live_cohort_accounting_and_poster_estimates_recompute(self):
        live = json.loads((DATA/'LIVE-WORKFLOW.json').read_bytes())
        self.assertEqual(live['cached_tokens'] + live['computed_tokens'], live['input_tokens'])
        rows = live['cohorts']
        for row in rows:
            self.assertAlmostEqual(row['cache_hit_percent'],
                                   100*row['cached_tokens']/row['input_tokens'])
        self.assertEqual(rows[0]['requests']+rows[1]['requests'], rows[2]['requests'])
        self.assertEqual(rows[0]['input_tokens']+rows[1]['input_tokens'], rows[2]['input_tokens'])
        earlier, later = live['periods']
        a, b = [r['computed_tokens']/r['input_tokens'] for r in (earlier, later)]
        self.assertAlmostEqual(100*(1-b/a), live['uncached_share_reduction_percent'])
        speed = lambda u: 1/(u/2000+(1-u)/200000)
        self.assertAlmostEqual(100*(speed(b)/speed(a)-1), live['estimated_effective_prefill_gain_percent'])
        manifest = json.loads((ROOT/'assets/1.2.7/MANIFEST.json').read_bytes())
        self.assertEqual(hashlib.sha256((ROOT/'assets/1.2.7'/manifest['file']).read_bytes()).hexdigest(),
                         manifest['sha256'])

    def test_companion_uses_selected_binary_identity_and_existing_native_pid(self):
        fixture_root = ROOT/'local-results'
        fixture_root.mkdir(exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix='127-keepalive-', dir=fixture_root))
        binary = root/'bin-1.2.7'
        binary.mkdir()
        (binary/'BUILD.json').write_text(json.dumps({'binaries': {
            'tensorfold-native': {'sha256': 'a'*64}, 'tf-dsv41-lanes': {'sha256': 'b'*64}}}))
        config = {'ranks': [{'root': str(root)}]}
        with mock_patch.object(launch, 'command', return_value=(['native'], {'TF_DS_ENGRAM':'/operator/engram'})) as check, \
             mock_patch.object(keepalive_launch.subprocess, 'check_output', return_value='1234\n'):
            argv = keepalive_launch.arguments(config, '1.2.7', 0, 'tp3@0.service')
            self.assertEqual(argv[argv.index('--exe-sha256')+1], 'a'*64)
            self.assertEqual(argv[argv.index('--pid')+1], '1234')
            self.assertEqual(argv[argv.index('--period-ms')+1], '25')
            check.assert_called_once_with(config, '1.2.7', 0, validate_files=True)
            with self.assertRaisesRegex(ValueError, 'no NVMe keepalive'):
                keepalive_launch.arguments(config, '1.2.6', 0, 'tp3@0.service')
        print('Retained companion fixture:', root)


if __name__ == '__main__':
    unittest.main()
