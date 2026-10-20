"""CPU-only checks of measured-source parity, coordinated launch and published data."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
AD=ROOT/'tensorfold/adaptive'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class AdaptiveRelease(unittest.TestCase):
    def test_exact_measured_patch_and_drift_rejection(self):
        patch=module('adaptive_patch',AD/'patch.py')
        manifest=json.loads((AD/'manifest.json').read_text(encoding='utf-8'))
        archive=ROOT/'tensorfold/artifacts/tensorfold-source.tar.gz'
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(),manifest['source_archive_sha256'])
        # Retain this small owned source fixture for audit; do not clean a shared build.
        dest=Path(tempfile.mkdtemp(prefix='ds41f-123-patch-'))
        print('Retained adaptive source check: '+str(dest))
        with tarfile.open(archive) as pack:
            for name in manifest['original_sha256']:
                path=dest/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(pack.extractfile(name).read())
        patch.apply(dest)
        patch.verify(dest)
        for name,h in manifest['patched_sha256'].items():
            self.assertEqual(hashlib.sha256((dest/name).read_bytes()).hexdigest(),h,name)
        before={name:(dest/name).read_bytes() for name in manifest['patched_sha256']}
        with self.assertRaisesRegex(ValueError,'pristine pinned source'):
            patch.apply(dest)
        self.assertEqual(before,{name:(dest/name).read_bytes() for name in before})

    def test_123_is_coordinated_and_legacy_selectors_remain_unchanged(self):
        launch=module('adaptive_launch',ROOT/'tensorfold/launch.py')
        config=json.loads((ROOT/'tensorfold/cluster.example.json').read_text(encoding='utf-8'))
        for rank in range(3):
            argv,env=launch.command(config,'1.2.3',rank)
            self.assertEqual(env['TF_DS_ADAPTIVE_LEARNING'],'1')
            self.assertEqual(env['TF_DS_POOL_TOKENS'],'7000000')
            self.assertEqual(env['TF_DS_STREAMS'],'12')
            self.assertEqual(env['TF_DS_ROUND_ROWS'],'48')
            self.assertIn('bin-1.2.3',argv[0])
            if rank==0:
                self.assertEqual(argv[argv.index('--prompt-cache-gib')+1],'3')
            _,old=launch.command(config,'1.2.0',rank)
            self.assertNotIn('TF_DS_ADAPTIVE_LEARNING',old)
            self.assertEqual(old['TF_DS_POOL_TOKENS'],'8000000')

    def test_123_rejects_old_receipt_before_execution(self):
        launch=module('adaptive_launch_receipt',ROOT/'tensorfold/launch.py')
        config=json.loads((ROOT/'tensorfold/cluster.example.json').read_text(encoding='utf-8'))
        dest=Path(tempfile.mkdtemp(prefix='ds41f-123-receipt-'))
        print('Retained adaptive receipt check: '+str(dest))
        config['ranks'][0]['root']=str(dest)
        (dest/'bin-1.2.3').mkdir()
        (dest/'bin-1.2.3/BUILD.json').write_text('{"version":"1.2.0"}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'adaptive 1.2.3 build receipt'):
            launch.command(config,'1.2.3',0,validate_files=True)

    def test_data_totals_and_all_markdown_duration_cells(self):
        folder=ROOT/'results/20261020-adaptive'
        s=json.loads((folder/'measurement-summary.json').read_text(encoding='utf-8'))
        with (folder/'comparison-all-metrics.csv').open(encoding='utf-8-sig',newline='') as f:
            rows={(r['arm'],int(r['C'])):r for r in csv.DictReader(f)}
        self.assertEqual(len(rows),36)
        with (folder/'task-results.csv').open(encoding='utf-8-sig',newline='') as f:tasks=list(csv.DictReader(f))
        with (folder/'request-timings.csv').open(encoding='utf-8-sig',newline='') as f:requests=list(csv.DictReader(f))
        self.assertEqual(len(tasks),234);self.assertEqual(len(requests),1031)
        arms=['original','bert1','xtf']
        for a in arms:
            self.assertEqual(sum(int(rows[a,c]['completion_tokens']) for c in range(1,13)),s['aggregates'][a]['generated_tokens'])
            self.assertEqual(sum(int(rows[a,c]['task_errors']) for c in range(1,13)),s['aggregates'][a]['corrected_tool_errors'])
            self.assertTrue(all(t['correct']=='True' for t in tasks if t['arm']==a))
        md=(ROOT/'docs/WORKFLOW-BENCHMARK-20261020.md').read_text(encoding='utf-8')
        first=md.split('### A1 / Full workflow durations',1)[1].split('### A2 /',1)[0]
        actual=[[v.strip() for v in line.strip('|').split('|')] for line in first.splitlines() if line.startswith('| ')][1:]
        expected=[[str(c)]+[f"{float(rows[a,c][key]):,.2f}" for key in ['wave_elapsed_seconds','task_median_seconds','task_p95_seconds'] for a in arms] for c in range(1,13)]
        self.assertEqual(actual,expected)


if __name__=='__main__':unittest.main()
