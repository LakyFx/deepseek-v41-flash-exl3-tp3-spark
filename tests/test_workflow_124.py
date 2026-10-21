"""Offline release checks: exact recipes, matched inputs, metrics and launch guards."""
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

from benchmarks.workflow import evaluate, fixtures, workflow
from benchmarks.workflow.capture import timing
from benchmarks.workflow.helpers import validate_receipt
from benchmarks.workflow.run import inputs
from tensorfold import launch
from tensorfold.variants import build, patch

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/20261021-workflow'


def retained(prefix):
    path=Path(tempfile.mkdtemp(prefix=prefix,dir=os.environ.get('DS41F_TEST_ARTIFACTS')))
    print('Retained 1.2.4 check resource: '+str(path))
    return path


def rows(name):
    with (DATA/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


class Workflow124(unittest.TestCase):
    @unittest.skipIf(os.name=='nt','Frozen Linux archive contains filenames not representable on Windows; run this check on Linux/WSL')
    def test_all_recipes_materialize_exact_measured_source_and_reject_drift(self):
        base=json.loads((patch.HERE/'base.json').read_bytes())
        with tarfile.open(ROOT/'tensorfold/artifacts/tensorfold-source.tar.gz') as pack:
            frozen={name:pack.extractfile(name).read() for name in base['files']}
        self.assertEqual(len(frozen),1130)
        root=retained('ds41f-124-source-')
        for manifest in sorted((patch.HERE/'recipes').glob('*.json')):
            source=root/manifest.stem
            for name,data in frozen.items():
                path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            spec,identity=patch.apply(source,manifest.stem)
            self.assertEqual(identity,patch.digest(manifest.read_bytes()))
            patch.verify(source,spec['source_files'])
            for name,record in spec['overlays'].items():
                self.assertEqual(patch.digest((source/name).read_bytes()),record['sha256'])
            # Reuse must not silently overlay an unpinned local edit.
            name=next(iter(base['files']));p=source/name;p.write_bytes(p.read_bytes()+b'\nDRIFT')
            with self.assertRaisesRegex(ValueError,'Source differs'):
                patch.apply(source,manifest.stem)

    def test_profiles_coordinate_all_three_ranks_and_preserve_capacity(self):
        config=json.loads((ROOT/'tensorfold/cluster.example.json').read_bytes())
        expected={'1.2.2TF':('8000000','12','32','2',False),'Bert1':('7000000','16','48','2',False)}
        selectors=['1.2.2TF','Bert1','1.2.3-reference','A','B','C','D','E','F','G','CF','DEF','1.2.4']
        for selector in selectors:
            pool,streams,rr,cache,learn=expected.get(selector,('7000000','12','48','3',True))
            for rank in range(3):
                argv,env=launch.command(config,selector,rank)
                self.assertEqual((env['TF_DS_POOL_TOKENS'],env['TF_DS_STREAMS'],env['TF_DS_ROUND_ROWS']),(pool,streams,rr))
                self.assertEqual(env.get('TF_DS_ADAPTIVE_LEARNING')=='1',learn)
                self.assertIn('bin-'+selector,argv[0])
                if rank==0:self.assertEqual(argv[argv.index('--prompt-cache-gib')+1],cache)
        with self.assertRaisesRegex(ValueError,'Unknown launcher'):
            launch.command(config,'../launch',0)

    def test_install_and_launch_reject_wrong_recipe_or_modified_binary(self):
        root=retained('ds41f-124-receipt-');built=root/'built';built.mkdir()
        spec,identity=patch.recipe('cdef');binaries={}
        for name in ['tensorfold-native','tf-dsv41-lanes']:
            (built/name).write_bytes(b'test-only-never-executable')
            binaries[name]=dict(path=name,sha256=patch.digest((built/name).read_bytes()))
        receipt=dict(schema='ds41f.native-build.v1',recipe='cdef',source_tree_sha256=spec['source_tree_sha256'],recipe_manifest_sha256=identity,source_archive_sha256=spec['base_archive_sha256'],adaptive_learning=True,binaries=binaries)
        (built/'BUILD.json').write_text(json.dumps(receipt))
        runtime=root/'runtime';build.install(built,runtime,'1.2.4')
        config=json.loads((ROOT/'tensorfold/cluster.example.json').read_bytes());config['ranks'][0]['root']=str(runtime)
        with self.assertRaisesRegex(ValueError,'Replace every topology'):
            launch.command(config,'1.2.4',0,validate_files=True)
        (runtime/'bin-1.2.4/tensorfold-native').write_bytes(b'drift')
        with self.assertRaisesRegex(ValueError,'binary differs'):
            launch.command(config,'1.2.4',0,validate_files=True)
        with self.assertRaisesRegex(ValueError,'does not match selected recipe'):
            build.install(built,root/'other','CF')
        (built/'tf-dsv41-lanes').write_bytes(b'drift')
        with self.assertRaisesRegex(ValueError,'Built binary changed'):
            build.install(built,root/'other','1.2.4')

    def test_frozen_inputs_and_real_evaluator(self):
        self.assertEqual(len(inputs()),78)
        self.assertEqual(workflow.expected_totals(),dict(requests=3,prompt_tokens=56000,completion_tokens=1200))
        self.assertTrue(evaluate.evaluate(fixtures.GOLDEN_CODE)['passed'])
        self.assertFalse(evaluate.evaluate(workflow.INITIAL)['passed'])
        with self.assertRaises(ValueError):evaluate.validate('import os\n')
        with self.assertRaisesRegex(ValueError,'receipt required'):
            validate_receipt({},'cdef','test',ROOT/'tensorfold/profiles/1.2.4.json',now=1000)

    def test_native_timing_excludes_queue_without_fabricating_gpu_compute(self):
        # TTFT contains 2 s queue + 3 s prefill; native total excludes server prep.
        extra={'tensorfold':dict(prefill_seconds=3,time_to_first_token=5.004,seconds=7,tokens_per_second=50)}
        r=timing(extra,dict(completion_tokens=101,computed_prompt_tokens=3000))
        self.assertAlmostEqual(r['post_prefill_first_token_upper_bound_seconds'],.004)
        self.assertEqual(r['computed_prefill_tps'],1000)
        self.assertIsNone(r['post_prefill_first_token_exact_seconds'])

    def test_export_totals_native_denominators_and_regenerated_tables(self):
        aggs=rows('aggregates.csv');cells=rows('cells.csv');tasks=rows('tasks.csv');requests=rows('requests.csv')
        self.assertEqual((len(aggs),len(cells),len(tasks),len(requests)),(14,168,1092,4846))
        self.assertTrue(all(t['correct']=='True' for t in tasks))
        meta=json.loads((DATA/'metadata.json').read_bytes());self.assertEqual(len(meta['source_cells_sha256']),168)
        counters={(r['arm'],int(r['C']),r['metric']):float(r['delta']) for r in rows('native-counters.csv') if r['delta']}
        for a in aggs:
            arm=a['arm'];cc=[c for c in cells if c['arm']==arm];rr=[r for r in requests if r['arm']==arm]
            self.assertEqual(sorted(int(c['C']) for c in cc),list(range(1,13)))
            self.assertEqual(len(rr),int(a['requests']))
            self.assertEqual(sum(int(r['completion_tokens']) for r in rr),int(a['completion_tokens']))
            self.assertAlmostEqual(sum(float(c['wave_seconds']) for c in cc),float(a['wave_seconds']))
            self.assertAlmostEqual(max(float(c['workflow_completion_tps']) for c in cc),float(a['best_wave_aggregate_tps']))
            seconds=sum(counters[arm,c,'request_decode_seconds_sum'] for c in range(1,13))
            self.assertAlmostEqual(int(a['completion_tokens'])/seconds,float(a['native_weighted_decode_tps']))
            for c in cc:
                rs=[r for r in rr if int(r['C'])==int(c['C'])]
                self.assertEqual(sum(int(r['completion_tokens']) for r in rs),int(c['completion_tokens']))
                self.assertAlmostEqual(int(c['completion_tokens'])/float(c['wave_seconds']),float(c['workflow_completion_tps']))
        spec=importlib.util.spec_from_file_location('render_124',ROOT/'tools/render_workflow_report.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        for name,text in m.render().items():self.assertEqual((ROOT/name).read_text(encoding='utf-8'),text,name)
        for name,h in json.loads((DATA/'SHA256.json').read_bytes()).items():
            self.assertEqual(hashlib.sha256((DATA/name).read_bytes()).hexdigest(),h,name)


if __name__=='__main__':unittest.main()
