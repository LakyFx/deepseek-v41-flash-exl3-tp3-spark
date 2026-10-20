"""Check every printed 1.2.3 benchmark cell against the public numeric CSVs."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'results/20261020-adaptive'


def read(name):
    with (FOLDER/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def main():
    rows={(r['arm'],int(r['C'])):r for r in read('comparison-all-metrics.csv')}
    native={(r['arm'],int(r['C'])):r for r in read('native-counters.csv')}
    arms=['original','bert1','xtf']
    groups=[
        [('wave_elapsed_seconds',1,2),('task_median_seconds',1,2),('task_p95_seconds',1,2)],
        [('native_decode_tps_median',1,2),('native_decode_tps_p95',1,2),('weighted_decode_tps',1,2)],
        [('per_request_completion_tps_median',1,2),('aggregate_workflow_completion_tps',1,2),('correct_tasks_per_minute',1,2)],
        [('prefill_seconds_median',1000,2),('prefill_seconds_p95',1000,2),('weighted_prefill_tps',1,2)],
        [('computed_prefill_tps_median',1,2),('computed_prefill_tps_p95',1,2),('computed_prefill_tokens',1,0)],
        [('native_ttft_seconds_median',1000,2),('native_ttft_seconds_p95',1000,2),('ttft_median_seconds',1000,2)],
        [('post_prefill_first_token_upper_bound_seconds_median',1000,3),('post_prefill_first_token_upper_bound_seconds_p95',1000,3),('acceptance',100,2)],
        [('requests',1,0),('completion_tokens',1,0),('task_errors',1,0)],
        [('mtp_drafted_total',1,0),('mtp_accepted_total',1,0),('decode_rounds_total',1,0)],
        [('prefix_cache_queries_total',1,0),('prefix_cache_hits_total',1,0),('output_tokens_per_request_decode_round',1,3)],
        [('prefill_duration_seconds_total',1,3),('request_decode_seconds_sum',1,2),('prefix_token_hit_fraction',100,2)],
    ]
    text=(ROOT/'docs/WORKFLOW-BENCHMARK-20261020.md').read_text(encoding='utf-8')
    checked=0
    for i,metrics in enumerate(groups,1):
        section=text.split(f'### A{i} /',1)[1].split('\n##',1)[0]
        actual=[[v.strip() for v in line.strip('|').split('|')]
            for line in section.splitlines() if line.startswith('| ')][1:]
        data=rows if i<=8 else native
        expected=[[str(c)]+[f'{float(data[a,c][key])*scale:,.{digits}f}'
            for key,scale,digits in metrics for a in arms] for c in range(1,13)]
        if actual!=expected:raise ValueError('Markdown differs from source CSV in appendix A'+str(i))
        checked+=108
    summary=json.loads((FOLDER/'measurement-summary.json').read_text(encoding='utf-8'))
    tasks=read('task-results.csv');requests=read('request-timings.csv')
    if len(tasks)!=234 or len(requests)!=1031 or len(rows)!=36:raise ValueError('Coverage drift')
    for a in arms:
        aggregate=summary['aggregates'][a]
        if sum(int(rows[a,c]['completion_tokens']) for c in range(1,13))!=aggregate['generated_tokens']:
            raise ValueError('Generated token mismatch')
        if sum(int(rows[a,c]['task_errors']) for c in range(1,13))!=aggregate['corrected_tool_errors']:
            raise ValueError('Tool error mismatch')
        if any(t['correct']!='True' for t in tasks if t['arm']==a):raise ValueError('Unexpected failed final task')
    markdown_sha=hashlib.sha256((ROOT/'docs/WORKFLOW-BENCHMARK-20261020.md').read_bytes()).hexdigest()
    print(json.dumps(dict(status='PASS',markdown_values_checked=checked,cells=36,tasks=234,requests=1031,
        markdown_sha256=markdown_sha,model_requests_performed=0)))


if __name__=='__main__':main()
