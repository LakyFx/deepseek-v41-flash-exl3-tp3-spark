"""Prepared M correctness/graph gate; default CLI never imports CUDA."""
import argparse
import json
import os
from pathlib import Path


def run():
    import torch
    from dgx_campaign_runtime import mhc
    if torch.cuda.get_device_capability()!=(12,1):raise ValueError('SM121 required')
    gen=torch.Generator(device='cuda').manual_seed(4108)
    def rand(shape,dtype=torch.float32):
        return torch.randn(shape,generator=gen,device='cuda',dtype=dtype)
    fn=rand((24,20480))/20480**0.5
    scale=torch.tensor([0.4,0.5,0.6],device='cuda')
    base=rand((24,))*0.1
    norm=rand((5120,),torch.bfloat16)*0.05+1
    reports=[]
    for rows in (0,1,3,6,18,48,1366,4096):
        x=rand((rows,5120),torch.bfloat16)
        residual=rand((rows,4,5120),torch.bfloat16)
        post=torch.sigmoid(rand((rows,4,1)))*2
        comb=torch.softmax(rand((rows,4,4)),dim=-1)
        incoming=torch.sigmoid(rand((rows,4)))+1e-6
        def call(enabled):
            os.environ['DGX_CAMPAIGN_M']='1' if enabled else '0'
            if not enabled and rows == 0:
                # Installed stock TileLang post launches grid=(0,1,1) for
                # empty input. There is no numerical reference to compute:
                # verify the mathematical empty output contract instead.
                return (torch.empty_like(residual),
                        torch.empty((0,4,1),device='cuda',dtype=torch.float32),
                        torch.empty((0,4,4),device='cuda',dtype=torch.float32),
                        torch.empty_like(x),
                        torch.empty((0,4),device='cuda',dtype=torch.float32))
            return mhc.post_pre(x,residual,post,comb,fn,scale,base,1e-20,1e-6,1e-6,2.0,20,
                               pre_mix=incoming,norm_weight=norm,norm_eps=1e-20)
        def check(actual,expected,label):
            evidence=[]
            for name,a,b in zip(('residual','post','comb','normalized','premix'),actual,expected):
                if a.shape!=b.shape or a.dtype!=b.dtype or not torch.isfinite(a).all():
                    raise AssertionError('M shape/dtype/nonfinite '+label+'/'+name)
                af,bf=a.float(),b.float()
                rms=float(torch.sqrt(torch.mean((af-bf)**2))) if a.numel() else 0.0
                norm_rms=float(torch.sqrt(torch.mean(bf**2))) if b.numel() else 0.0
                relative=rms/max(norm_rms,1e-12)
                max_abs=float((af-bf).abs().max()) if a.numel() else 0.0
                # Different accumulation order is permitted; rounding noise
                # must stay below 0.3% RMS for every output independently.
                # Report the tolerance and actual errors, never imply equality.
                if relative>0.003:raise AssertionError('M numeric '+label+'/'+name+' RMS='+str(relative))
                evidence.append({'output':name,'relative_rms_error':relative,'max_abs_error':max_abs,
                                 'bit_equal':bool(torch.equal(a,b)),'relative_rms_limit':0.003})
            return evidence
        expected=call(False);actual=call(True)
        report={'rows':rows,'reference':'empty-output-contract' if rows==0 else 'installed-stock',
                'eager':check(actual,expected,'eager'),'graph_replays':[]}
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):call(True);call(True)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=call(True)
        for iteration in range(3):
            if rows:
                x.copy_(rand(x.shape,torch.bfloat16))
                incoming.copy_(torch.sigmoid(rand(incoming.shape))+1e-6)
            expected=call(False);graph.replay();torch.cuda.synchronize()
            report['graph_replays'].append(check(captured,expected,'mutated-replay'+str(iteration)))
        reports.append(report)
    return {'schema':'dgx.actual-mhc-qualification.v1','complete':True,'cases':reports,
            'comparison':'installed stock delayed mHC; original epsilon1e-20,4streams,20Sinkhorn; synthetic data',
            'scope':'kernel correctness/shape/dtype/input-mutating graphs; not model quality or speed'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--armed',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not args.armed:
        print(json.dumps({'ready_to_run_after_closed_window':True,'GPU_launches':0}));return
    if args.output is None or args.output.exists():parser.error('new immutable output required')
    result=run()
    with args.output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({'complete':result['complete'],'cases':len(result['cases'])}))


if __name__=='__main__':main()
