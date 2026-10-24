"""Source/build selection and exact stock benchmark/poster provenance, offline."""
import json
from pathlib import Path
import statistics
import unittest
from xml.etree import ElementTree as ET

from tensorfold import launch
from tensorfold.variants import patch

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/20261024-1.2.6'


class Release126(unittest.TestCase):
    def test_exact_measured_native_source_and_binary_identity(self):
        recipe,_=patch.recipe('v126')
        q=json.loads((DATA/'QUALIFICATION.json').read_bytes())
        parity=json.loads((DATA/'PRODUCTION-PARITY.json').read_bytes())
        self.assertEqual(len(recipe['source_files']),1139)
        self.assertEqual(recipe['release_version'],'1.2.6')
        self.assertEqual(recipe['source_tree_sha256'],q['native_source_tree_sha256'])
        self.assertEqual(recipe['source_tree_sha256'],parity['source_tree_sha256'])
        for name,entry in recipe['overlays'].items():
            self.assertEqual(patch.digest((patch.HERE/entry['object']).read_bytes()),recipe['source_files'][name])
        for rank in parity['ranks']:
            self.assertEqual(rank['binary_sha256'],q['reference_binaries']['tensorfold-native' if rank['rank']==0 else 'tf-dsv41-lanes'])

    def test_cap_and_planner_on_every_rank_with_historical_profile_preserved(self):
        config=json.loads((ROOT/'tensorfold/cluster.example.json').read_bytes())
        for rank in range(3):
            argv,env=launch.command(config,'1.2.6',rank)
            self.assertEqual(env['TF_KV_PRESSURE_PLANNER'],'1')
            self.assertEqual(env['TF_KV_RELOCATE_MAX_BYTES'],'335544320')
            self.assertEqual(env['TF_DS_ADAPTIVE_LEARNING'],'1')
            self.assertEqual(env['TF_DS_POOL_TOKENS'],'7000000')
            self.assertEqual(env['TF_DS_ROUND_ROWS'],'48')
            self.assertIn('bin-1.2.6',argv[0])
            if rank==0:self.assertIn('--loop-guard',argv)
            _,prior=launch.command(config,'1.2.5',rank)
            self.assertNotIn('TF_KV_RELOCATE_MAX_BYTES',prior)

    def test_all_three_campaigns_reconcile_streams_and_medians(self):
        count=0
        for run in range(1,4):
            for name in ['prose','code']:
                job=json.loads((DATA/f'run-{run}'/(name+'.json')).read_bytes())
                self.assertEqual(job['status'],'completed')
                self.assertEqual([r['concurrency'] for r in job['results']],[1,2,3,4,5,6,8,10,12])
                for row in job['results']:
                    self.assertEqual(row['streamsOk'],row['concurrency'])
                    self.assertEqual(row['streamsFailed'],0)
                    self.assertAlmostEqual(row['medianDecodeTps'],statistics.median(s['decodeTps'] for s in row['streams']),delta=.011)
                    self.assertAlmostEqual(row['medianTtftMs'],statistics.median(s['ttftMs'] for s in row['streams']),delta=.011)
                    self.assertEqual(row['totalDecodeTokens'],399*row['concurrency'])
                    for stream in row['streams']:
                        self.assertEqual(stream['completionTokens'],400)
                        self.assertEqual(stream['reasoningChunks'],0)
                        self.assertEqual(stream['http']['request']['temperature'],0)
                        count+=1
        self.assertEqual(count,306)

    def test_both_posters_use_only_run_two_and_every_measured_concurrency(self):
        manifest=json.loads((ROOT/'assets/1.2.6/MANIFEST.json').read_bytes())
        export=json.loads((DATA/'EXPORT.json').read_bytes())
        self.assertEqual(manifest['selected_run'],2)
        for name,filename in [('prose','01-prose'),('code','02-code')]:
            self.assertEqual(manifest['source_files_sha256'][name+'.json'],export['original_private_job_sha256']['run-2/'+name+'.json'])
            svg=ET.parse(ROOT/'assets/1.2.6'/(filename+'.svg'))
            texts=[e.text for e in svg.iter() if e.tag.endswith('}text')]
            self.assertIn('1.2.6',texts)
            job=json.loads((DATA/'run-2'/(name+'.json')).read_bytes())
            for row in job['results']:
                self.assertIn('×'+str(row['concurrency']),texts)
                self.assertIn(f"{row['aggregateDecodeTps']:.1f}",texts)
                self.assertIn(f"{row['medianDecodeTps']:.1f}",texts)
            for ext in ['svg','png']:
                file=ROOT/'assets/1.2.6'/(filename+'.'+ext)
                self.assertEqual(patch.digest(file.read_bytes()),manifest['files'][file.name])


if __name__=='__main__':unittest.main()
