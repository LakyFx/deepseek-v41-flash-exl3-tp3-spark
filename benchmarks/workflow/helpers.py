"""Frozen summary and exclusive-window receipt validation; no remote control."""
import hashlib
import json
import math
import statistics
import time

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)

def validate_receipt(value,arm,campaign,profile,now=None):
    now=time.time() if now is None else now
    expected=hashlib.sha256(profile.read_bytes()).hexdigest()
    if (value.get('arm')!=arm or value.get('campaign_id')!=campaign
        or value.get('profile_sha256')!=expected or value.get('exclusive_window') is not True
        or value.get('public_api_closed') is not True or not value.get('engine_generation')
        or value.get('ranks_verified')!=[0,1,2] or not value.get('user_go_reference')
        or not 0 <= now-value.get('created_epoch',0) <= 900):
        raise ValueError('fresh verified deployment/exclusive-window/GO receipt required')

def summary(tasks,seconds):
    latencies=sorted(t['elapsed_seconds'] for t in tasks)
    requests=[r for t in tasks for r in t['requests']]
    rates=[];first=[]
    for r in requests:
        start=r['started_monotonic'];delta=r['first_any_delta_monotonic'];end=r['ended_monotonic']
        if delta is not None:
            first.append(delta-start)
        # Tools/reasoning can be buffered into a late SSE chunk. Dividing all
        # tokens by its tiny delivery tail would fabricate very high TPS.
        if end>start:rates.append(r['usage']['completion_tokens']/(end-start))
    return {'correct':sum(t['correct'] for t in tasks),'total':len(tasks),
        'wave_elapsed_seconds':seconds,'task_median_seconds':statistics.median(latencies),
        'task_p95_seconds':latencies[math.ceil(.95*len(latencies))-1],
        'ttft_median_seconds':statistics.median(first) if first else None,
        'per_request_completion_tps_median':statistics.median(rates) if rates else None,
        'rate_scope':'individual request completion tokens / complete request wall time including queue/prefill/delivery; reasoning and tools included; native decode rate separate',
        'correct_tasks_per_minute':sum(t['correct'] for t in tasks)*60/seconds if seconds>0 else None,
        'failures_retained':True,'task_errors':sum(len(t['tool_errors']) for t in tasks)}
