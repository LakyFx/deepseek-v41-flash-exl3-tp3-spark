"""Production identity, role-specific template and public benchmark evidence."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch as mock_patch

from tensorfold import launch, prepare_template
from tensorfold.variants import patch

ROOT = Path(__file__).resolve().parents[1]


class Release125(unittest.TestCase):
    def test_exact_native_source_and_running_executable_identities(self):
        spec, _ = patch.recipe('k2a')
        parity = json.loads((ROOT/'results/20261023-1.2.5/PRODUCTION-PARITY.json').read_bytes())
        qualification = json.loads((ROOT/'results/20261023-1.2.5/QUALIFICATION.json').read_bytes())
        self.assertEqual(len(spec['source_files']), 1137)
        self.assertEqual(spec['source_tree_sha256'], qualification['native_source_tree_sha256'])
        for name, record in spec['overlays'].items():
            self.assertEqual(patch.digest((patch.HERE/record['object']).read_bytes()), spec['source_files'][name])
        for rank in parity['ranks']:
            name = 'tensorfold-native' if rank['rank'] == 0 else 'tf-dsv41-lanes'
            self.assertEqual(rank['binary_sha256'], spec['recorded_binaries'][name]['sha256'])

    def test_new_launcher_keeps_measured_capacity_and_role_specific_metadata(self):
        cfg = json.loads((ROOT/'tensorfold/cluster.example.json').read_bytes())
        for rank in range(3):
            argv, env = launch.command(cfg, '1.2.5', rank)
            self.assertEqual([env[k] for k in ['TF_DS_POOL_TOKENS','TF_DS_STREAMS','TF_DS_ROUND_ROWS']], ['7000000','12','48'])
            self.assertEqual(env['TF_DS_ADAPTIVE_LEARNING'], '1')
            self.assertIn('bin-1.2.5', argv[0])
            if rank == 0:
                self.assertIn('TF351-1017-1.2.5', argv[2])
                self.assertEqual(env['TF_K2A_DIAGNOSTICS'], '1')
                self.assertEqual(env['TF_K2A_PRUNE_DEAD'], '1')
                self.assertEqual(env['TF_LOOP_DRY_WINDOWS'], '10')
            else:
                self.assertTrue(argv[1].endswith('TF351-1017'))
                self.assertNotIn('TF_K2A_DIAGNOSTICS', env)
        _, previous = launch.command(cfg, '1.2.4', 0)
        self.assertNotIn('TF_K2A_PRUNE_DEAD', previous)

    def test_published_template_matches_actual_head_and_original_follower_roles(self):
        p = json.loads((ROOT/'tensorfold/profiles/1.2.5.json').read_bytes())
        parity = json.loads((ROOT/'results/20261023-1.2.5/PRODUCTION-PARITY.json').read_bytes())
        self.assertEqual(hashlib.sha256((ROOT/'tensorfold/templates/1017-direct-output.jinja').read_bytes()).hexdigest(), p['head_chat_template_sha256'])
        for rank in parity['ranks']:
            self.assertEqual(rank['template_sha256'], p['head_chat_template_sha256'] if rank['rank'] == 0 else p['chat_template_sha256'])

    @unittest.skipIf(os.name == 'nt', 'Requires normal Linux symlink semantics')
    def test_template_overlay_preserves_originals_and_refuses_overwrite_or_drift(self):
        root = Path(tempfile.mkdtemp(prefix='ds41f-125-template-', dir=os.environ.get('DS41F_TEST_ARTIFACTS')))
        print('Retained 1.2.5 test resource: '+str(root))
        here = root/'recipe';(here/'profiles').mkdir(parents=True);(here/'templates').mkdir()
        original = root/'runtime/weights/models/TF351-1017';original.mkdir(parents=True)
        (original/'chat_template.jinja').write_bytes(b'ORIGINAL')
        (original/'config.json').write_bytes(b'{}')
        (here/'templates/1017-direct-output.jinja').write_bytes(b'DIRECT')
        p = {'arm':'TF351-1017','head_model_subdir':'weights/models/HEAD',
             'chat_template_sha256':patch.digest(b'ORIGINAL'),'head_chat_template_sha256':patch.digest(b'DIRECT')}
        (here/'profiles/1.2.5.json').write_text(json.dumps(p))
        with mock_patch.object(prepare_template, 'HERE', here):
            result = prepare_template.prepare(root/'runtime')
            self.assertFalse(result['original_changed'])
            self.assertEqual((original/'chat_template.jinja').read_bytes(), b'ORIGINAL')
            head=root/'runtime/weights/models/HEAD'
            self.assertEqual((head/'chat_template.jinja').read_bytes(), b'DIRECT')
            self.assertTrue((head/'config.json').samefile(original/'config.json'))
            with self.assertRaises(FileExistsError):prepare_template.prepare(root/'runtime')
            (original/'chat_template.jinja').write_bytes(b'DRIFT')
            with self.assertRaisesRegex(ValueError, 'Original model template'):prepare_template.prepare(root/'runtime')

    def test_stock_measurements_reconcile_and_never_mix_reasoning_modes(self):
        streams=0
        for name in ['prose','code']:
            job=json.loads((ROOT/f'results/20261023-sparkdash/{name}.json').read_bytes())
            self.assertEqual(job['status'],'completed')
            for row in job['results']:
                self.assertEqual(row['streamsOk'],row['concurrency'])
                self.assertEqual(row['streamsFailed'],0)
                self.assertAlmostEqual(row['meanDecodeTps'],statistics.mean(s['decodeTps'] for s in row['streams']),delta=.011)
                self.assertAlmostEqual(row['meanTtftMs'],statistics.mean(s['ttftMs'] for s in row['streams']),delta=.011)
                for stream in row['streams']:
                    self.assertEqual(stream['completionTokens'],400)
                    self.assertEqual(stream['reasoningChunks'],0)
                    streams+=1
        self.assertEqual(streams,102)
        verification=json.loads((ROOT/'results/20261023-sparkdash/VERIFICATION.json').read_bytes())
        self.assertEqual(verification['thinking_false_requests'],121)
        self.assertEqual(verification['thinking_true_requests'],0)
        workflow=json.loads((ROOT/'results/20261023-1.2.5/WORKFLOW.json').read_bytes())
        self.assertEqual(sum(x['correct'] for x in workflow['rows']),78)
        self.assertEqual(sum(x['task_failures'] for x in workflow['rows']),0)


if __name__ == '__main__':unittest.main()
