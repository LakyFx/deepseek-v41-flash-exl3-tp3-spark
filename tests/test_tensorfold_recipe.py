"""CPU checks of public package integrity and coordinated profile launch contracts."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TF = ROOT / 'tensorfold'
spec = importlib.util.spec_from_file_location('tf_launch', TF / 'launch.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class TensorFoldRecipe(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((TF / 'cluster.example.json').read_bytes())

    def test_coordinated_pool_and_max_across_all_ranks(self):
        for name, pool in [('TF351', 10000000), ('TF351-3.3M', 3300000), ('TF29', 3300000)]:
            for rank in range(3):
                argv, env = launch.command(self.config, name, rank, benchmark=True)
                self.assertEqual(env['TF_DS_POOL_TOKENS'], str(pool))
                self.assertEqual(env['TF_TP_WORLD'], '3')
                self.assertEqual(env['TF_DS_CACHE_SHORT_CONTINUATION'], '1')
                self.assertEqual(env['TF_DS_TOOL_SEPARATOR_ROUNDTRIP'], '1')
                if rank == 0:
                    self.assertEqual(argv[argv.index('--reasoning-effort') + 1], 'max')
                    self.assertEqual(argv[argv.index('--host') + 1], '127.0.0.1')
                    self.assertIn('--benchmark-control', argv)
                else:
                    self.assertEqual(argv[argv.index('--pool') + 1], str(pool))

    def test_example_topology_cannot_start(self):
        with self.assertRaisesRegex(ValueError, 'placeholder'):
            launch.command(self.config, 'TF351', 0, validate_files=True)
        self.config.update(master_address='192.0.2.10', nccl_interface='eth0',
                           nccl_hcas=['hca0', 'hca1', 'hca2', 'hca3'], nccl_address_range='192.0.2.0/24')
        self.config['ranks'][0]['rdma_devices'] = ['hca0', 'hca1', 'hca2', 'hca3']
        with self.assertRaisesRegex(ValueError, 'example address'):
            launch.command(self.config, 'TF351', 0, validate_files=True)

    def test_rank_order_is_explicit(self):
        self.config['ranks'].reverse()
        with self.assertRaisesRegex(ValueError, 'ordered'):
            launch.command(self.config, 'TF351', 0)

    def test_all_shipped_inputs_match_hashes_and_portable_paths(self):
        manifest = json.loads((TF / 'inputs.json').read_bytes())
        for name, row in manifest['files'].items():
            self.assertNotIn('\\', name)
            raw = (TF / name).read_bytes()
            self.assertEqual(len(raw), row['bytes'], name)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row['sha256'], name)

    def test_archives_bind_every_file_and_required_export_tools(self):
        for archive, inventory in [('tensorfold-source.tar.gz', 'source-files.json'),
                                   ('tensorfold-kernels.tar.gz', 'kernel-files.json'),
                                   ('tensorfold-rope-reference.tar.gz', 'rope-reference-files.json')]:
            rows = json.loads((TF / 'artifacts' / inventory).read_bytes())
            seen = set()
            with tarfile.open(TF / 'artifacts' / archive) as pack:
                for member in pack.getmembers():
                    self.assertTrue(member.isfile(), member.name)
                    self.assertNotIn('..', Path(member.name).parts)
                    self.assertIn(member.name, rows)
                    raw = pack.extractfile(member).read()
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), rows[member.name]['sha256'])
                    seen.add(member.name)
            self.assertEqual(seen, set(rows))
        for name in ['checkpoint_plan', 'rank_cache', 'rank_verify', 'source_verify', 'io_budget',
                     'materialize_checkpoint', 'tp3_layout', 'vision_export', 'export_campaign']:
            self.assertIn('tools/dsv41-tp3/' + name + '.py', json.loads((TF / 'artifacts/source-files.json').read_bytes()))

    def test_all_public_receipts_match_inventory(self):
        folder = ROOT / 'results/20261013-tensorfold'
        rows = json.loads((folder / 'evidence-manifest.json').read_bytes())
        self.assertGreaterEqual(len(rows), 3396)
        self.assertIn('initial-v22/FINAL-SUMMARY-v22.json', rows)
        for name, row in rows.items():
            raw = (folder / 'evidence' / name).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row['public_sha256'], name)
        actual = {p.relative_to(folder / 'evidence').as_posix() for p in (folder / 'evidence').rglob('*.json')}
        self.assertEqual(actual, set(rows))

    def test_quality_fixture_frozen_before_generations(self):
        quality = TF / 'benchmark-kit/quality-q20'
        freeze = json.loads((quality / 'FROZEN-MANIFEST.json').read_bytes())
        self.assertEqual(freeze['generations_before_freeze'], 0)
        self.assertEqual(hashlib.sha256((quality / 'FIXTURES.json').read_bytes()).hexdigest(), freeze['fixture_sha256'])
        fixture = json.loads((quality / 'FIXTURES.json').read_bytes())
        self.assertEqual(len(fixture['attempts']), 24)
        self.assertEqual(len(fixture['cases']), 20)

    def test_same_computed_work_despite_different_cache_fraction(self):
        rows = json.loads((ROOT / 'results/20261013-tensorfold/summary.json').read_bytes())['final']
        a, b = [[r['usage'] for r in arm['workflows']] for arm in rows]
        for first, second in zip(a, b):
            self.assertEqual(first['computed_prompt_tokens'], second['computed_prompt_tokens'])
            self.assertNotEqual(first['cache_fraction'], second['cache_fraction'])

    def test_hermes_cold_rate_uses_actual_uncached_tokens(self):
        row = json.loads((ROOT / 'results/20261013-tensorfold/hermes-cold64k.json').read_bytes())
        # The publication keeps this derived metric separate from Owen TTFT.
        self.assertEqual(row['cached_tokens'], 0)
        self.assertTrue(row['cold_cache_verified'])
        self.assertEqual(row['computed_tokens'], row['prompt_tokens'])
        self.assertAlmostEqual(row['computed_tokens'] / row['prefill_seconds'], row['tok_s'], places=6)
        self.assertIn('raw_source_sha256', row)


if __name__ == '__main__':
    unittest.main()
