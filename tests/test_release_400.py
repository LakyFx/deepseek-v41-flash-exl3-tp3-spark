"""Offline production parity, complete TP4 delivery and TP3 separation checks."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import tarfile
import tempfile
import unittest
from unittest.mock import patch as mock_patch

from tensorfold import launch as tp3
from tensorfold.tp4 import assets, build, launch
from tensorfold.variants import patch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'results/20261031-4.0.0'


class Release400(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work = ROOT/'local-results'
        work.mkdir(exist_ok=True)
        cls.work = Path(tempfile.mkdtemp(prefix='400-public-', dir=work))
        cls.kit = cls.work/'kit'
        cls.kit.mkdir()
        with tarfile.open(ROOT/'tensorfold/artifacts/tensorfold-kernels.tar.gz') as archive:
            for item in archive:
                if not item.isfile():
                    continue
                dest = cls.kit.joinpath(*patch.relative(item.name).parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(archive.extractfile(item).read())

    def test_exact_full_source_and_measured_binary_identity(self):
        spec, _ = patch.recipe('v400')
        with tarfile.open(ROOT/'tensorfold/artifacts/tensorfold-source.tar.gz') as archive:
            actual = {m.name: patch.digest(archive.extractfile(m).read()) for m in archive if m.isfile()}
        for name, record in spec['overlays'].items():
            actual[name] = patch.digest((patch.HERE/record['object']).read_bytes())
        for name in spec['remove']:
            actual.pop(name)
        self.assertEqual(actual, spec['source_files'])
        self.assertEqual(len(actual), 1146)
        parity = json.loads((DATA/'PRODUCTION-PARITY.json').read_bytes())
        self.assertEqual(spec['source_tree_sha256'], parity['source_tree_sha256'])
        self.assertEqual(spec['private_source_manifest_sha256'], parity['source_manifest_sha256'])
        self.assertEqual({k: v['sha256'] for k, v in spec['recorded_binaries'].items()}, parity['native'])
        previous, _ = patch.recipe('v128')
        for name in ['zig/src/core/adaptive_prefix.zig', 'zig/src/core/checkpoint_policy.zig',
                     'zig/src/core/lanes/adaptive.zig', 'zig/src/families/dsv41/kv_pressure.zig']:
            self.assertEqual(actual[name], previous['source_files'][name])

    def test_stage_exact_binaries_and_complete_591_entry_kernel_set(self):
        profile = json.loads((launch.HERE/'profile.json').read_bytes())
        out = self.work/'staged'
        receipt = build.prepare(self.kit, out)
        self.assertEqual(receipt['binaries'].keys(), profile['historical_native_binaries'].keys())
        for name, expected in profile['historical_native_binaries'].items():
            self.assertEqual(patch.digest((out/'native'/name).read_bytes()), expected)
        assets.check(out/'prefill', profile['prefill_assets_sha256'])
        matrix = json.loads((out/'prefill/aot/aot.json').read_bytes())
        self.assertEqual(len(matrix['kernels']), 591)
        for entry in matrix['kernels']:
            self.assertEqual((out/'prefill/aot/cubins'/(entry['hash']+'.cubin')).read_bytes()[:4], b'\x7fELF')
        self.assertEqual(patch.digest((out/'prefill/aot/aot.json').read_bytes()),
                         '1042f9203e62efa04103904525575ba0945d22a7188606dde601174d7a81548b')
        with self.assertRaises(FileExistsError):
            build.prepare(self.kit, out)

    def test_wrong_kernel_manifest_is_rejected(self):
        with self.assertRaises(ValueError):
            assets.inputs(self.kit, '0'*64)

    def test_all_four_launches_match_current_production_capacity_and_ownership(self):
        cfg = json.loads((launch.HERE/'cluster.example.json').read_bytes())
        for rank in range(4):
            with self.subTest(rank=rank):
                argv, env = launch.command(cfg, rank)
                self.assertEqual(env['TF_TP_WORLD'], '4')
                self.assertEqual(env['TF_DS_STREAMS'], '16')
                self.assertEqual(env['TF_DS_POOL_TOKENS'], '9000000')
                self.assertEqual(env['TF_DS_ROUND_ROWS'], '48')
                self.assertEqual(env['TF_CACHE_ADAPTIVE_PREFIX'], '1')
                self.assertEqual(env['TF_DS_ADAPTIVE_LEARNING'], '1')
                self.assertEqual(env['NCCL_SWITCHLESS_RING_ONLY'], '1')
                self.assertEqual(env['NCCL_IB_PRESERVE_PCI_DOMAIN'], '1')
                self.assertTrue(env['TF_DS_ENGRAM'].endswith(f'engram-tp2-r{rank%2}'))
                self.assertNotIn('TF_DS_EXPERIMENTAL_TP3', env)
                if rank:
                    self.assertEqual(argv[3:5], [str(rank), '4'])
                    self.assertEqual(argv[argv.index('--prompt-context')+1], '401408')
                    self.assertIn('--round-ms', argv)
                else:
                    self.assertEqual(argv[argv.index('--parallel')+1], '16')
                    self.assertEqual(argv[argv.index('--prompt-cache-gib')+1], '3.5')
        argv, _ = launch.command(cfg, 0, benchmark=True)
        self.assertIn('--benchmark-control', argv)
        self.assertEqual(argv[argv.index('--host')+1], '127.0.0.1')

    def test_placeholder_and_ambiguous_topology_cannot_start(self):
        cfg = json.loads((launch.HERE/'cluster.example.json').read_bytes())
        with self.assertRaisesRegex(ValueError, 'placeholder'):
            launch.command(cfg, 0, validate_files=True)
        cfg['ranks'][0]['neighbor_second'] = cfg['ranks'][0]['neighbor_first']
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            launch.command(cfg, 0)

    def test_tp3_profile_and_launcher_remain_separate(self):
        cfg = json.loads((ROOT/'tensorfold/cluster.example.json').read_bytes())
        _, env = tp3.command(cfg, '1.2.8', 0)
        self.assertEqual(env['TF_TP_WORLD'], '3')
        self.assertEqual(env['TF_DS_STREAMS'], '12')
        self.assertEqual(env['TF_DS_POOL_TOKENS'], '7000000')
        with self.assertRaises(ValueError):
            tp3.command(cfg, '4.0.0', 0)
        with self.assertRaises(ValueError):
            launch.command(cfg, 0)

    def test_inherited_tp3_knobs_do_not_leak_into_tp4(self):
        with mock_patch.dict(os.environ, {'TF_DS_EXPERIMENTAL_TP3': '1', 'NCCL_MAX_NCHANNELS': '8'}):
            actual = launch.process_environment({'TF_TP_WORLD': '4'})
        self.assertNotIn('TF_DS_EXPERIMENTAL_TP3', actual)
        self.assertNotIn('NCCL_MAX_NCHANNELS', actual)

    def test_gateway_profile_changes_only_the_two_capacity_fields(self):
        base = json.loads((ROOT/'gateway/laguna_gateway/production-settings.json').read_bytes())
        selected = json.loads((launch.HERE/'gateway-profile.json').read_bytes())
        changed = {k for k in base if base[k] != selected[k]}
        self.assertEqual(changed, {'max_concurrency', 'max_admitted_tokens'})
        self.assertEqual(selected['max_concurrency'], 16)
        self.assertEqual(selected['max_admitted_tokens'], 8000000)
        self.assertEqual(selected['max_model_len'], 401408)
        example = (launch.HERE/'gateway.env.example').read_text()
        self.assertIn('DS41F_GATEWAY_PROFILE=', example)

    def test_measured_prefill_uses_exact_cold_samples_not_workflow_or_cache(self):
        raw = json.loads((DATA/'P2-COLD-SAMPLES.json').read_bytes())
        summary = json.loads((DATA/'P2-COLD-PREFILL.json').read_bytes())
        self.assertEqual(len(raw), 21)
        for cell in summary['results']:
            rows = [r for r in raw if r['target_tokens'] == cell['target_tokens']]
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(r['cached_tokens'] == 0 and r['computed_prompt_tokens'] == r['prompt_tokens'] for r in rows))
            self.assertAlmostEqual(statistics.median(r['native_prefill_tps'] for r in rows), cell['median_tps'])
        work = json.loads((DATA/'WORKFLOW.json').read_bytes())
        self.assertEqual(sum(c['correct'] for c in work['cells']), 78)
        self.assertEqual(sum(c['requests'] for c in work['cells']), 327)
        self.assertEqual(sum(c['completion_tokens'] for c in work['cells']), 145668)
        self.assertAlmostEqual(sum(c['wave_elapsed_seconds'] for c in work['cells']), work['sum_wave_seconds'])


if __name__ == '__main__':
    unittest.main()
