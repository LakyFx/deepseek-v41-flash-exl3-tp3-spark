"""Check complete AF reconstruction, kernel delivery and measured prefill scopes."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import tempfile
import unittest

from tensorfold import launch, prefill_assets
from tensorfold.variants import patch, build

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'results/20261030-1.2.8'


class Release128(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exact source/kit materialization, entirely offline; no GPU requests.
        (ROOT/'local-results').mkdir(exist_ok=True)
        cls.work = Path(tempfile.mkdtemp(prefix='128-public-', dir=ROOT/'local-results'))
        cls.extracted = cls.work/'extracted'
        if os.name == 'nt':
            # Upstream's Linux fixtures contain '?' names, illegal on NTFS.
            # Verify all source bytes in memory here; exercise real extraction on Linux.
            (cls.extracted/'kit').mkdir(parents=True)
            with tarfile.open(ROOT/'tensorfold/artifacts/tensorfold-kernels.tar.gz') as archive:
                for member in archive:
                    dest=cls.extracted/'kit'/member.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(archive.extractfile(member).read())
        else:
            subprocess.run([sys.executable, 'tensorfold/extract.py', '--out', str(cls.extracted)],
                           cwd=ROOT, check=True, capture_output=True)

    def test_actual_full_materialization_matches_measured_source(self):
        spec, _ = patch.recipe('v128')
        with tarfile.open(ROOT/'tensorfold/artifacts/tensorfold-source.tar.gz') as archive:
            files={m.name:patch.digest(archive.extractfile(m).read()) for m in archive}
        for name, record in spec['overlays'].items():
            files[name]=patch.digest((patch.HERE/record['object']).read_bytes())
        for name in spec['remove']: files.pop(name)
        self.assertEqual(files,spec['source_files'])
        if os.name != 'nt':
            patch.apply(self.extracted/'source', 'v128')
        q = json.loads((DATA/'QUALIFICATION.json').read_bytes())
        self.assertEqual(spec['source_tree_sha256'], q['source_tree_sha256'])
        self.assertEqual(spec['private_source_manifest_sha256'], q['source_manifest_sha256'])
        self.assertEqual(spec['recorded_binaries'], q['measured_binaries'])
        old, _ = patch.recipe('v127')
        changed = {k for k,v in spec['source_files'].items() if old['source_files'].get(k)!=v}
        self.assertEqual(len(changed), 9)
        self.assertEqual(len(spec['source_files']), 1144)
        for path in ('zig/src/core/adaptive_prefix.zig', 'zig/src/core/checkpoint_policy.zig',
                     'zig/src/core/lanes/adaptive.zig', 'zig/src/families/dsv41/kv_pressure.zig'):
            self.assertEqual(spec['source_files'][path], old['source_files'][path])

    def test_all_rank_flags_use_qualified_assets_and_context(self):
        config = json.loads((ROOT/'tensorfold/cluster.example.json').read_bytes())
        for rank in range(3):
            argv, env = launch.command(config, '1.2.8', rank)
            old_argv, old_env = launch.command(config, '1.2.7', rank)
            self.assertIn('bin-1.2.8', argv[0])
            self.assertIn('prefill-1.2.8', env['TENSORFOLD_CUDA_KERNELS'])
            self.assertIn('prefill-1.2.8', env['TF_DS_PREFILL_RAGGED_CUBIN'])
            self.assertEqual(old_env['TENSORFOLD_CUDA_KERNELS'], str(Path(config['ranks'][rank]['root'])/'kit/aot'))
            for key in ['TF_CACHE_ADAPTIVE_PREFIX','TF_DS_ADAPTIVE_LEARNING','TF_DS_POOL_TOKENS',
                        'TF_DS_STREAMS','TF_DS_ROUND_ROWS','TF_KV_RELOCATE_MAX_BYTES']:
                self.assertEqual(env[key], old_env[key])
            for key in ['TF_DS_PREFILL_ROWS','TF_DS_PREFILL_INDEX_ROWS','TF_DS_PREFILL_FRONT_ROWS',
                        'TF_DS_PREFILL_MASKED_ATTN','TF_DS_PREFILL_RAGGED_MMA','TF_DS_PREFILL_ENGRAM_AUDIT']:
                self.assertEqual(env[key], '1')
            self.assertNotIn('--prompt-context', old_argv)
            if rank:
                self.assertEqual(argv[argv.index('--prompt-context')+1], '401408')
            else:
                self.assertEqual(argv[argv.index('--context')+1], '401408')

    def test_assets_install_preserves_base_and_detects_tampering(self):
        profile = json.loads((ROOT/'tensorfold/profiles/1.2.8.json').read_bytes())
        wanted = profile['prefill_assets_sha256']
        kit = self.extracted/'kit'
        before = hashlib.sha256((kit/'aot/aot.json').read_bytes()).hexdigest()
        destination = self.work/'assets'
        prefill_assets.prepare(kit, destination, wanted)
        prefill_assets.check(destination, wanted)
        self.assertEqual(hashlib.sha256((kit/'aot/aot.json').read_bytes()).hexdigest(), before)
        base = json.loads((kit/'aot/aot.json').read_bytes())
        final = json.loads((destination/'aot/aot.json').read_bytes())
        self.assertEqual(final['kernels'][:len(base['kernels'])], base['kernels'])
        self.assertEqual(len(final['kernels'])-len(base['kernels']), 24)
        required = {v['hash']+'.cubin' for v in final['kernels']}
        present = {p.name for p in (destination/'aot/cubins').glob('*.cubin')}
        self.assertTrue(required <= present)
        (destination/'auxiliary/experts_ragged.cubin').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'Installed kernel differs'):
            prefill_assets.check(destination, wanted)
        with self.assertRaises(ValueError):
            prefill_assets.manifest('0'*64)

    def test_install_rejects_wrong_kernel_receipt_before_writing(self):
        profile = json.loads((ROOT/'tensorfold/profiles/1.2.8.json').read_bytes())
        spec, identity = patch.recipe('v128')
        built = self.work/'fake-build';built.mkdir()
        binaries={}
        for name in ['tensorfold-native','tf-dsv41-lanes']:
            (built/name).write_bytes(name.encode())
            binaries[name]={'path':name,'sha256':hashlib.sha256(name.encode()).hexdigest()}
        receipt=dict(schema='ds41f.native-build.v1',recipe='v128',recipe_manifest_sha256=identity,source_tree_sha256=spec['source_tree_sha256'],
                     source_archive_sha256=spec['base_archive_sha256'],adaptive_learning=True,binaries=binaries)
        (built/'BUILD.json').write_text(json.dumps(receipt))
        runtime=self.work/'runtime'
        with self.assertRaisesRegex(ValueError,'matching 1.2.8 kernels'):
            build.install(built,runtime,'1.2.8')
        self.assertFalse(runtime.exists())
        expected=profile['prefill_assets_sha256']
        prefill_assets.prepare(self.extracted/'kit',built/'prefill-1.2.8',expected)
        receipt['prefill_assets_sha256']=expected
        (built/'BUILD.json').write_text(json.dumps(receipt))
        install=build.install(built,runtime,'1.2.8')
        self.assertFalse(install['services_touched'])
        prefill_assets.check(runtime/'prefill-1.2.8',expected)
        config=json.loads((ROOT/'tensorfold/cluster.example.json').read_bytes())
        config['ranks'][0]['root']=str(runtime)
        config_path=self.work/'invalid-topology.json'
        config_path.write_text(json.dumps(config))
        # Exercise the systemd-style direct script entrypoint in isolated Python.
        # The placeholder topology must fail before any model process can start.
        result=subprocess.run([sys.executable,'-I',str(ROOT/'tensorfold/launch.py'),
                               '--config',str(config_path),'--variant','1.2.8','--rank','0','--run'],
                              cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Replace every topology placeholder',result.stderr)
        self.assertNotIn('ModuleNotFoundError',result.stderr)

    def test_prefill_medians_recompute_and_remain_distinct_from_decode(self):
        samples=json.loads((DATA/'SAMPLES.json').read_bytes())['samples']
        report=json.loads((DATA/'PREFILL.json').read_bytes())
        for row in report['values']['AF']:
            values=[v for v in samples if v['prompt_tokens']==row['tokens']]
            self.assertEqual(len(values),3)
            for sample in values:
                self.assertEqual(sample['cached_tokens'],0)
                self.assertAlmostEqual(sample['native_prefill_tps'],
                                       sample['prompt_tokens']/sample['native_prefill_seconds'])
            self.assertAlmostEqual(statistics.median(v['native_prefill_tps'] for v in values),row['tps'])
        for row in report['proof']['AF']['workspace'].values():
            self.assertEqual((row['pool_tokens'],row['prompt_tokens'],row['chunk_rows']),
                             (7000000,403456,4096))
        self.assertEqual(report['boundary_check']['prompt_tokens'],401344)
        self.assertFalse(json.loads((DATA/'QUALIFICATION.json').read_bytes())['full_decode_concurrency_campaign_for_AF'])


if __name__=='__main__': unittest.main()
