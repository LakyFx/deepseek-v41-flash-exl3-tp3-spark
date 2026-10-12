"""Future N7 threshold/row-collective gate; private buffers, actual TP group."""
import os
from types import SimpleNamespace

COUNTS = (255,256,257,511,512,513,2047,2048,2049,4096)


def policy_cases(runner, threshold, selector):
    import numpy as np
    reports=[]
    if os.environ.get('DGX_V089X_SP')!='1' or os.environ.get('DGX_V089X_SP_MIN_ROWS')!=str(threshold):
        raise AssertionError('actual SP activation/threshold differs')
    for rows in COUNTS:
        for phase,flags in (('prefill',[True]),('mixed',[True,False]),('decode',[False])):
            batch=SimpleNamespace(is_prefilling_np=np.array(flags,dtype=bool))
            scheduler=SimpleNamespace(scheduled_encoder_inputs={})
            actual=selector(runner,batch,scheduler,rows,False)
            expected=rows>=threshold and phase=='prefill'
            if actual is not expected:raise AssertionError('SP phase/threshold selection changed')
            if selector(runner,batch,scheduler,rows,True):raise AssertionError('SP active during dummy capture')
            scheduler.scheduled_encoder_inputs={'vision':[0]}
            if selector(runner,batch,scheduler,rows,False):raise AssertionError('SP entered multimodal encoder batch')
            reports.append({'rows':rows,'phase':phase,'active':actual})
    return reports


def run(worker, stage):
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank, tensor_model_parallel_all_reduce
    from v089x_runtime.rows import RowPartition, runner_prefill_eager, gather_indexer_output
    from dgx_campaign_runtime.worker_checks import numeric
    threshold={'N7a':512,'N7b':256}[stage]
    policies=policy_cases(worker.model_runner,threshold,runner_prefill_eager)
    device=worker.model_runner.device
    rank=get_tensor_model_parallel_rank()
    reports=[]
    for count in COUNTS:
        plan=RowPartition(count,3,rank)
        # Exact representable values expose dropped/reordered/padded rows.
        base=(torch.arange(count,device=device,dtype=torch.int32)%31).to(torch.bfloat16)
        partial=base[:,None].expand(-1,128).contiguous()+(rank+1)
        expected=tensor_model_parallel_all_reduce(partial.clone())
        local=plan.reduce_scatter(partial)
        gathered=plan.gather(local)
        reduced=numeric(gathered,expected,exact=True)
        if local.shape!=(plan.width,128):raise AssertionError('SP local row shape changed')
        if plan.real<plan.width and torch.count_nonzero(local[plan.real:]):
            raise AssertionError('SP padded reduction leaked real data')
        # Gather disjoint per-rank indexer rows with sentinel-filled padding.
        topk=torch.full((count,16),-1,device=device,dtype=torch.int32)
        candidates=torch.full((count,8),-1,device=device,dtype=torch.int32)
        for tensor in (topk,candidates):
            tensor[plan.start:plan.start+plan.real]=torch.arange(plan.start,plan.start+plan.real,
                device=device,dtype=torch.int32)[:,None]
        gather_indexer_output(plan,topk,candidates,True)
        wanted=torch.arange(count,device=device,dtype=torch.int32)[:,None]
        if not torch.equal(topk,wanted.expand_as(topk)) or not torch.equal(candidates,wanted.expand_as(candidates)):
            raise AssertionError('SP indexer/candidate row gathering changed')
        reports.append({'rows':count,'rank_width':plan.width,'real_rows':plan.real,
                        'collective':reduced,'indexer_exact':True})
    return {'complete':True,'threshold':threshold,'policy_cases':policies,'cases':reports,
            'scope':'actual TP3 reduce-scatter/gather and indexer row reconstruction at old/new thresholds; prefill model quality tested separately'}
