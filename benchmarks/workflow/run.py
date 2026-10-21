"""Prepare frozen inputs offline; run only in an explicitly exclusive loopback window."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time

from .helpers import write, summary, validate_receipt
from .native_client import Client
from .network import comparison
from .workflow import body_for, run_task

HERE=Path(__file__).resolve().parent
FROZEN=json.loads((HERE/'frozen-inputs.json').read_bytes())


def campaign_for(c):
    return FROZEN['campaign_id']+(FROZEN['continuation_C10_C12'] if c>=10 else '')


def inputs():
    rows=[]
    for c in range(1,13):
        for i in range(c):
            body=body_for(campaign_for(c),c,i)
            rows.append(dict(concurrency=c,replica=i,sha256=hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False).encode()).hexdigest()))
    if rows!=FROZEN['initial_inputs']:
        raise ValueError('Initial workload changed: refuse unmatched benchmark')
    return rows


def run(args):
    rows=inputs()
    if not args.run:
        return dict(prepared=True,model_requests=0,initial_inputs=rows)
    if not args.exclusive or not all([args.port,args.receipt,args.profile,args.output,args.arm]):
        raise ValueError('--run requires --exclusive, port, receipt, profile, output and arm')
    receipt=json.loads(args.receipt.read_bytes())
    validate_receipt(receipt,args.arm,FROZEN['campaign_id'],args.profile)
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'receipt.json',receipt)
    write(args.output/'initial-inputs.json',rows)
    client=Client(args.port,armed=True,journal=lambda value:write(args.output/'owned.json',value))
    results=[];request_count=0
    try:
        for c in range(1,13):
            before=client.require_idle();barrier=threading.Barrier(c)
            def task(i):
                barrier.wait()
                return run_task(client,campaign_for(c),c,i)
            start=time.monotonic()
            with ThreadPoolExecutor(max_workers=c) as pool:
                tasks=list(pool.map(task,range(c)))
            elapsed=time.monotonic()-start
            cell=dict(arm=args.arm,concurrency=c,tasks=tasks,summary=summary(tasks,elapsed))
            write(args.output/f'C{c:02}.json',cell)
            after=client.require_idle()
            requests=[r for t in tasks for r in t['requests']]
            request_count+=len(requests)
            cell['native']=comparison(before,after,len(requests),sum(r['usage']['completion_tokens'] for r in requests))
            cell['metrics_before']=before;cell['metrics_after']=after
            write(args.output/f'C{c:02}.json',cell)
            if not cell['native']['coverage_verified']:
                raise ValueError('Counter coverage mismatch; retain result and stop')
            results.append(dict(C=c,**cell['summary']))
            write(args.output/'summary.json',dict(arm=args.arm,cells=results,complete=c==12))
            print(json.dumps(results[-1]),flush=True)
        return dict(complete=True,tasks=sum(r['total'] for r in results),correct=sum(r['correct'] for r in results),model_requests=request_count,api_changed=False)
    except BaseException as exc:
        client.abort_all_owned()
        write(args.output/'FAILURE.json',dict(error=type(exc).__name__+': '+str(exc),reconciliation_required=True))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true');p.add_argument('--exclusive',action='store_true')
    p.add_argument('--arm');p.add_argument('--port',type=int);p.add_argument('--profile',type=Path)
    p.add_argument('--receipt',type=Path);p.add_argument('--output',type=Path)
    print(json.dumps(run(p.parse_args()),indent=2))


if __name__=='__main__':main()
