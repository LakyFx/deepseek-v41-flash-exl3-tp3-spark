"""Future worker RPC qualification over actual loaded TP3 parameters.

Import is CPU-only. Execution requires a current closed-window lease mounted
read-only and the exact running variant. This file never changes the sampler
or patches a forward call. It is not installed in the live baseline.
"""
import hashlib
import json
import os
from pathlib import Path
import re

LEASE=Path('/opt/dgx_campaign/lease/lease.json')
OUTPUT=Path('/opt/dgx_campaign/qualification')
_DENSE_TARGET=re.compile(r'^(?:language_model\.)?model\.layers\.(?:[0-9]|[12][0-9]|3[0-9])\.(?:'
                         r'attn\..+|ffn\.shared_experts\.(?:gate_up_proj|down_proj))$')


def authorize(payload):
    data=json.loads(payload)
    if set(data)!={'campaign_id','plan_sha256','version','stage','nonce'}:
        raise ValueError('exact qualification request required')
    if not re.fullmatch(r'[a-f0-9]{64}',data['plan_sha256']):raise ValueError('plan hash required')
    if not re.fullmatch(r'[A-Za-z0-9._-]{1,100}',data['nonce']):raise ValueError('bounded nonce required')
    if LEASE.is_symlink() or not LEASE.is_file():raise ValueError('closed-window lease unavailable')
    owner=json.loads(LEASE.read_bytes())
    if (owner.get('window_closed') is not True or owner.get('head_ready') is not True
        or any(owner.get(k)!=data[k] for k in ('campaign_id','plan_sha256'))
        or owner.get('active_version')!=data['version']
        or not re.fullmatch(r'[a-f0-9]{64}',owner.get('container_id',''))
        or os.environ.get('DGX_V089_VERSION')!=data['version']):
        raise ValueError('qualification ownership or generation differs')
    return data,owner


def numeric(actual,expected,*,limit=0.003,exact=False):
    import torch
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype:
        raise AssertionError('reference shape or dtype differs')
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError('nonfinite reference/candidate')
    a,b=actual.float(),expected.float()
    rms=float(torch.sqrt(torch.mean((a-b)**2))) if a.numel() else 0.0
    norm=float(torch.sqrt(torch.mean(b**2))) if b.numel() else 0.0
    relative=rms/max(norm,1e-12)
    equal=bool(torch.equal(actual,expected))
    if relative>limit or (exact and not equal):raise AssertionError('numeric mismatch RMS='+str(relative))
    return {'relative_rms_error':relative,'relative_rms_limit':limit,'bit_equal':equal,
            'max_abs_error':float((a-b).abs().max()) if a.numel() else 0.0}


def graph_replays(call,reference,mutate,*,limit=0.003):
    import torch
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):call();call()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):output=call()
    reports=[]
    for i in range(3):
        mutate(i);expected=reference();graph.replay();torch.cuda.synchronize()
        reports.append(numeric(output,expected,limit=limit))
    return reports


def contract(worker):
    import torch
    from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity
    runner=worker.model_runner;cfg=worker.vllm_config
    capacity,concurrency=get_kv_cache_capacity(cfg,runner.kv_cache_config)
    spec=cfg.speculative_config
    modules=os.environ['DGX_EXPERIMENT_MODULES'].split(',')
    depth=int(os.environ.get('DGX_X11_FOLLOWUP_MAX_DRAFT', '3' if 'S' in modules else '5'))
    if depth not in (3,4,5):raise AssertionError('unsupported expected draft depth')
    values={'tp':cfg.parallel_config.tensor_parallel_size,'ep':cfg.parallel_config.enable_expert_parallel,
            'max_context':cfg.model_config.max_model_len,'prefill_chunk':cfg.scheduler_config.max_num_batched_tokens,
            'max_seqs':cfg.scheduler_config.max_num_seqs,'kv_dtype':cfg.cache_config.cache_dtype,
            'kv_bytes':cfg.cache_config.kv_cache_memory_bytes,'kv_block':cfg.cache_config.block_size,
            'draft_depth':spec.num_speculative_tokens,'adaptive':spec.enable_adaptive_verification}
    wanted={'tp':3,'ep':False,'max_context':401408,'prefill_chunk':4096,'max_seqs':8,
            # DeepSeek attention rewrites the unchanged CLI fp8 policy to its
            # native fp8_ds_mla layout before KV allocation.
            'kv_dtype':'fp8_ds_mla','kv_bytes':8074035200,'kv_block':128,'draft_depth':depth,'adaptive':'A' in modules}
    if values!=wanted or capacity<3299563:raise AssertionError('effective loaded contract/KV differs '+str(values))
    if torch.cuda.get_device_capability()!=(12,1):raise ValueError('actual SM121 required')
    return {'effective':values,'kv_min_tokens':capacity,'kv_max_concurrency':concurrency,
            'num_blocks':runner.kv_cache_config.num_blocks,'modules':modules}


def transport_gate(worker):
    import torch
    from vllm.distributed import get_tp_group
    comm=get_tp_group().device_communicator;handle=comm.dgx_campaign_r
    if handle is None or handle.disabled:raise AssertionError('selected R unavailable')
    rank=get_tp_group().rank_in_group;reports=[]
    # Integer-valued inputs make different all-reduce orders exactly comparable.
    for dtype in (torch.bfloat16,torch.float32):
        for rows,width in ((1,5120),(6,5120),(18,5120),(48,5120),(3,256)):
            x=torch.full((rows,width),rank+1,device=comm.device,dtype=dtype)
            if not handle.should_custom_ar(x):raise AssertionError('required decode collective rejected')
            expected=torch.full_like(x,6)
            numeric(handle.custom_all_reduce(x),expected,exact=True)
            graph=torch.cuda.CUDAGraph()
            with handle.capture(),torch.cuda.graph(graph):out=handle.custom_all_reduce(x)
            for i in range(3):
                x.fill_((rank+1)*(i+2));graph.replay();torch.cuda.synchronize();handle.check_health()
                numeric(out,torch.full_like(x,6*(i+2)),exact=True)
            if not handle.should_all_gather(x,-1):raise AssertionError('required D gather rejected')
            gathered=handle.all_gather(x,-1)
            expected=torch.cat([torch.full_like(x,(r+1)*4) for r in range(3)],dim=-1)
            numeric(gathered,expected,exact=True)
            graph=torch.cuda.CUDAGraph()
            with handle.capture(),torch.cuda.graph(graph):out=handle.all_gather(x,-1)
            for i in range(3):
                x.fill_(rank+1+i);graph.replay();torch.cuda.synchronize();handle.check_health()
                numeric(out,torch.cat([torch.full_like(x,r+1+i) for r in range(3)],dim=-1),exact=True)
            reports.append({'dtype':str(dtype),'rows':rows,'width':width,'reduce_and_gather':True,
                            'input_mutating_graph_replays':3})
    return {'complete':True,'cases':reports,'extra_nccl_communicators':0}


def expert_gate(worker):
    import torch
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    from cuda_exl3.moe import _glu_had_in
    layers={}
    for name,layer in worker.model_runner.model.named_modules():
        if not hasattr(layer,'w13_trellis'):continue
        signature=(int(layer.w13_trellis.shape[-1]//16),int(layer.w2_trellis.shape[-1]//16),
                   int(layer.exl3_inter),int(layer.exl3_cb))
        layers.setdefault(signature,(name,layer))
    if not layers:raise AssertionError('no actual EXL3 routed parameters')
    if not {3,4}.issubset({b for shape in layers for b in shape[:2]}):
        raise AssertionError('actual mixed3/4-bit coverage missing')
    reports=[];ops=torch.ops.cuda_exl3_C
    for signature,(name,layer) in sorted(layers.items()):
        if signature[2:]!=(768,2):raise AssertionError('unexpected TP3 expert shape/codebook')
        h,i,e=5120,layer.exl3_inter,layer.exl3_num_experts
        for m in (1,6,18,48):
            gen=torch.Generator(device='cpu').manual_seed(4108+m)
            x=torch.randn((m,h),generator=gen).to(device=layer.w13_trellis.device,dtype=torch.bfloat16)
            ids=(torch.arange(m*8,device=x.device).reshape(m,8)%(min(e,7))).int()
            weights=torch.softmax(torch.randn((m,8),generator=gen),dim=-1).to(x.device)
            sorted_ids,expert_ids,nrows=moe_align_block_size(ids,16,e,pad_sorted_ids=True)
            sorted_ids,expert_ids,nrows=sorted_ids.int(),expert_ids.int(),nrows.int()
            rows=min(expert_ids.numel()*16,sorted_ids.numel());expert_ids=expert_ids[:rows//16]
            a13=torch.empty((2,rows,h),dtype=torch.half,device=x.device)
            a2=torch.empty((1,rows,i),dtype=torch.half,device=x.device)
            # Write padding rows too. Reference invokes the same unchanged
            # decoder without sorted IDs, which cannot enter campaign K.
            def projection(routed,*,fused=False):
                ops.exl3_moe_had_in(x,a13,layer.w13_suh.data,sorted_ids,expert_ids,nrows,16,8,m*8,False)
                inter=ops.exl3_moe_gemm(a13,layer.w13_trellis.data,layer.w13_suh.data,layer.w13_svh.data,
                    expert_ids,nrows,[i,i],2,16,torch.bfloat16,sorted_ids if routed else None,None,m,8)
                _glu_had_in(ops,inter,a2,layer.w2_suh.data,expert_ids,nrows,16,10.0)
                if routed and fused:
                    return ops.exl3_moe_gemm(a2,layer.w2_trellis.data,layer.w2_suh.data,layer.w2_svh.data,
                        expert_ids,nrows,[h],2,16,torch.bfloat16,sorted_ids,weights,m,8)
                out=ops.exl3_moe_gemm(a2,layer.w2_trellis.data,layer.w2_suh.data,layer.w2_svh.data,
                    expert_ids,nrows,[h],2,16,torch.bfloat16,sorted_ids if routed else None,None,m,8)
                return ops.exl3_moe_combine(out,sorted_ids,weights,m,expert_ids,16)
            # Compare the new projection tile and stock tile using the same
            # FP32 fixed-order combine. The production BF16 atomic epilogue
            # already exists in baseline and has a separate rounding budget
            # (upstream test_exl3_moe_determinism.py uses relative RMS <1e-2).
            # Mixing these sums caused a false K failure before testing K.
            error=numeric(projection(True),projection(False))
            replays=graph_replays(lambda:projection(True),lambda:projection(False),lambda n:x.mul_(-1))
            fused_error=numeric(projection(True,fused=True),projection(False),limit=0.01)
            fused_replays=graph_replays(lambda:projection(True,fused=True),lambda:projection(False),
                                        lambda n:x.mul_(-1),limit=0.01)
            reports.append({'layer':name,'bits':signature[:2],'rows':m,'numeric':error,'graphs':replays,
                            'existing_bf16_atomic_epilogue':fused_error,'fused_graphs':fused_replays})
    return {'complete':True,'cases':reports,'reference':'same loaded packed weights; candidate and stock tiles with identical fixed-order combine (0.003); unchanged production BF16 atomic epilogue separately (0.01)'}


def transport_failure_gate(worker):
    import torch
    import torch.distributed as dist
    from vllm.distributed import get_tp_group
    from sparknet import oneshot
    group=get_tp_group();comm=group.device_communicator
    # Independent, small registered buffers/proxy. The serving transport is
    # never poisoned or stopped. Existing Gloo setup avoids another NCCL
    # communicator and its multi-GB resident allocation.
    previous=os.environ.get('SPARKNET_ROCE_SPIN_LIMIT')
    os.environ['SPARKNET_ROCE_SPIN_LIMIT']='2000000'
    try:
        runtime=oneshot.AllReduce.from_exchange_group(exchange_group=comm.cpu_group,
            device=comm.device,max_size=comm.dgx_campaign_r.all_reduce_capacity_bytes,
            max_gather_bytes=comm.dgx_campaign_r.all_gather_max_bytes)
    finally:
        if previous is None:os.environ.pop('SPARKNET_ROCE_SPIN_LIMIT',None)
        else:os.environ['SPARKNET_ROCE_SPIN_LIMIT']=previous
    try:
        runtime.prepare((torch.bfloat16,))
        x=torch.ones(4096,dtype=torch.bfloat16,device=comm.device)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):runtime.all_reduce(x)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        dist.barrier(group=comm.cpu_group)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream),runtime.capture(stream=stream):
            y1=runtime.all_reduce(x);y2=runtime.all_reduce(y1);y3=runtime.all_reduce(y2)
        torch.cuda.synchronize();dist.barrier(group=comm.cpu_group)
        graph.replay();torch.cuda.synchronize();runtime.check_health()
        numeric(y3,x*27,exact=True);epoch=runtime.stats()['epoch']
        dist.barrier(group=comm.cpu_group)
        if group.rank_in_group==1:runtime._proxy.stop()
        y3.zero_();graph.replay();torch.cuda.synchronize()
        try:runtime.check_health()
        except RuntimeError as exc:
            if not re.search('poisoned|proxy failed',str(exc)):raise
        else:raise AssertionError('faulted graph did not fail-stop')
        if not runtime.poisoned or runtime.stats()['epoch']>=epoch+3 or torch.equal(y3,x*27):
            raise AssertionError('faulted runtime advanced or released valid output')
        try:runtime.all_reduce(x)
        except RuntimeError:pass
        else:raise AssertionError('poisoned runtime accepted another launch')
        return {'complete':True,'fault':'rank1 stopped ONLY independent probe proxy',
                'all_rank_graph_fail_stop':True,'no_fallback':True,'extra_nccl_communicators':0}
    finally:
        runtime.close();dist.barrier(group=comm.cpu_group)
        comm.dgx_campaign_r.check_health()


def prefetch_gate(worker):
    import torch
    from vllm.distributed import get_tp_group,tensor_model_parallel_all_reduce
    from dgx_campaign_runtime import prefetch
    if not prefetch._plans:raise AssertionError('no actual P weight ranges')
    reports=[];device=get_tp_group().device;rank=get_tp_group().rank_in_group
    # Every captured range must remain within a strong-referenced real tensor.
    for key,plan in prefetch._plans.items():
        ranges=[(t.data_ptr(),t.untyped_storage().nbytes()-t.storage_offset()*t.element_size()) for t in plan['owners']]
        for name,ptr,size in plan['segments']:
            if not any(start==ptr and 0<size<=extent for start,extent in ranges):
                raise AssertionError('P descriptor no longer belongs to a retained tensor')
    for key in list(prefetch._plans)[:2]:
        for rows in (1,18,48):
            x=torch.full((rows,5120),rank+1,device=device,dtype=torch.bfloat16)
            call=lambda:prefetch._op()(x,key)
            reference=lambda:tensor_model_parallel_all_reduce(x)
            error=numeric(call(),reference(),exact=True)
            # R capture binds the stream edges used by P's unchanged reduction.
            handle=get_tp_group().device_communicator.dgx_campaign_r
            with handle.capture():replays=graph_replays(call,reference,lambda n:x.fill_(rank+1+n))
            reports.append({'key':key,'rows':rows,'numeric':error,'graphs':replays})
    return {'complete':True,'retained_plans':len(prefetch._plans),'cases':reports}


def reference_file(data,name,rows):
    # Shared only between versions of this owned campaign, on the same rank.
    tag=hashlib.sha256(name.encode()).hexdigest()[:32]
    source_plan=data.get('reference_plan_sha256',data['plan_sha256'])
    if not re.fullmatch(r'[a-f0-9]{64}',source_plan):raise ValueError('bounded reference plan required')
    reference_output=(Path('/opt/dgx_campaign/stock-references')
        if data['campaign_id']=='dsv41-expanded-20261008' else OUTPUT)
    root=reference_output/data['campaign_id']/('references-'+source_plan[:16])
    if not re.fullmatch(r'[A-Za-z0-9._-]{1,128}',data['campaign_id']):raise ValueError('bounded campaign identity required')
    if root.is_symlink():raise ValueError('reference directory symlink')
    if reference_output==OUTPUT:root.mkdir(parents=True,exist_ok=True)
    elif not root.is_dir():raise ValueError('retained stock references missing')
    return root/(tag+'-'+str(rows)+'.pt')


def save_reference(data,name,rows,value):
    import torch
    path=reference_file(data,name,rows)
    if path.exists():raise ValueError('reference already exists; never overwrite')
    cpu=value.detach().cpu()
    with path.open('xb') as stream:torch.save(cpu,stream)
    raw=path.read_bytes()
    with path.with_suffix('.json').open('x',encoding='utf-8') as stream:
        json.dump({'campaign_id':data['campaign_id'],'plan_sha256':data['plan_sha256'],
                   'name':name,'rows':rows,'reference_version':data['version'],
                   'sha256':hashlib.sha256(raw).hexdigest(),'shape':list(cpu.shape),'dtype':str(cpu.dtype)},stream)
    return str(path)


def load_reference(data,name,rows,device):
    import torch
    path=reference_file(data,name,rows);meta=json.loads(path.with_suffix('.json').read_bytes())
    raw=path.read_bytes()
    if (meta.get('campaign_id')!=data['campaign_id'] or meta.get('plan_sha256')!=data.get('reference_plan_sha256',data['plan_sha256'])
        or meta.get('name')!=name or meta.get('rows')!=rows or meta.get('reference_version')!='0.8.9X1'
        or hashlib.sha256(raw).hexdigest()!=meta['sha256']):raise ValueError('exact stock reference differs')
    return torch.load(path,map_location=device,weights_only=True)


def markov_gate(worker,data,*,record=False):
    import torch
    model=worker.model_runner.speculator.model
    head=model.model.markov_head
    if head.markov_w1.weight.shape[-1]!=256:raise AssertionError('trained Markov rank differs')
    vocab=head.markov_w1.num_embeddings;reports=[]
    for rows in (1,3,8):
        ids=torch.tensor(([0,1,vocab-1,17,53,1001,99,vocab-2])[:rows],device=head.markov_w1.weight.device)
        embedding=head.embed(ids)
        def call():return model.markov_bias(embedding)
        value=call()
        if not isinstance(value,torch.Tensor) or value.shape!=(rows,model.logits_processor.org_vocab_size):
            raise AssertionError('full vocabulary must reach every rank')
        if record:
            if head.markov_w2.tp_size!=1:raise AssertionError('reference Markov is already sharded')
            path=save_reference(data,'D/markov-bias',rows,value)
            reports.append({'rows':rows,'stock_reference':path})
        else:
            if head.markov_w2.tp_size!=3:raise AssertionError('D sharding was not activated')
            expected=load_reference(data,'D/markov-bias',rows,value.device)
            error=numeric(value,expected)
            # Full proposal distributions, never top-k/greedy substitution.
            probabilities=numeric(torch.softmax(value.float()/0.6,dim=-1),torch.softmax(expected.float()/0.6,dim=-1),limit=0.005)
            from vllm.distributed import get_tp_group
            handle=get_tp_group().device_communicator.dgx_campaign_r
            def mutate(_):
                embedding.mul_(-1);expected.neg_()
            with handle.capture():
                replays=graph_replays(call,lambda:expected,mutate)
            reports.append({'rows':rows,'logits':error,'probabilities_temperature_0_6':probabilities,'graphs':replays})
    return {'complete':True,'cases':reports,'trained_rank':256,'record_stock':record}


def dense_layers(worker):
    # The stock-reference arm deliberately has no H runtime installed.
    # Inventory must be independent of the candidate it will later compare.
    selected={};rejected=[]
    for name,layer in worker.model_runner.model.named_modules():
        prefix=getattr(layer,'prefix',name)
        if not _DENSE_TARGET.fullmatch(prefix) or getattr(layer,'bmm_batch_size',None) is not None:continue
        method=getattr(layer,'quant_method',None)
        if method is None or not hasattr(method,'kernel'):
            rejected.append({'prefix':prefix,'reason':'not MXFP8 kernel owner','method':type(method).__name__});continue
        packed=getattr(layer,'b12x_mxfp8_packed_weight',None)
        if packed is not None:shape=(int(packed.in_features),int(packed.out_features))
        else:
            weight=getattr(layer,'weight',None)
            if weight is None or weight.ndim!=2:
                rejected.append({'prefix':prefix,'reason':'no 2D stock weight','method':type(method).__name__,
                                 'shape':None if weight is None else list(weight.shape)});continue
            # Standard vLLM column and row-parallel layer attributes avoid
            # guessing from a backend's transposed post-load weight layout.
            k=getattr(layer,'input_size_per_partition',getattr(layer,'input_size',None))
            n=getattr(layer,'output_size_per_partition',None)
            if n is None:
                sizes=getattr(layer,'output_partition_sizes',None)
                n=sum(sizes) if sizes else int(weight.shape[0])
            if k is None:raise AssertionError('H actual input width unavailable '+prefix)
            shape=(int(k),int(n))
        selected.setdefault(shape,(prefix,layer,method))
    if not selected:raise AssertionError('H target linear inventory empty '+json.dumps(rejected[:12]))
    return selected


def dense_gate(worker,data,*,record=False):
    import torch
    reports=[]
    for (k,n),(prefix,layer,method) in sorted(dense_layers(worker).items()):
        packed=getattr(layer,'b12x_mxfp8_packed_weight',None)
        if record and packed is not None:raise AssertionError('H stock reference already packed')
        if not record and packed is None:raise AssertionError('selected H target did not use B12X '+prefix)
        device=layer.weight.device if record else packed.weight.values.device
        for rows in (1,6,18,48,256,4096):
            gen=torch.Generator(device='cpu').manual_seed(4108+k+n)
            x=(torch.randn((rows,k),generator=gen)*0.25).to(device=device,dtype=torch.bfloat16)
            def call():return method.apply(layer,x,bias=None)
            value=call()
            if value.shape!=(rows,n):raise AssertionError('H inventory/output shape mismatch '+prefix)
            if record:
                path=save_reference(data,'H/'+prefix,rows,value)
                reports.append({'prefix':prefix,'shape':[n,k],'rows':rows,'stock_reference':path})
            else:
                expected=load_reference(data,'H/'+prefix,rows,value.device)
                # Both paths retain original FP8 weights. B12X uses BF16
                # activation for small rows; report its measured discrepancy
                # from the stock backend instead of claiming bit equality.
                error=numeric(value,expected,limit=0.03)
                replays=graph_replays(call,lambda:call(),lambda i:x.mul_(-1))
                reports.append({'prefix':prefix,'shape':[n,k],'rows':rows,'numeric_stock':error,'graphs':replays})
    return {'complete':True,'cases':reports,'record_stock':record,
            'scope':'one real target tensor per actual TP shape, all decode/prefill rows; sampled model quality checked separately'}


def adaptive_gate(worker):
    import numpy as np
    runner=worker.model_runner;manager=runner.adaptive_verification
    if manager is None or manager.cost_tables is None:raise AssertionError('adaptive cost profile missing')
    expected_depth=int(os.environ.get('DGX_X11_FOLLOWUP_MAX_DRAFT','5'))
    if expected_depth not in (3,4,5) or manager.num_speculative_steps!=expected_depth:
        raise AssertionError('adaptive depth differs from prepared candidate')
    if manager._cudagraph_limit<48:raise AssertionError('adaptive C8 verification graphs unavailable')
    if runner.speculator.model.model.confidence_head is None:raise AssertionError('actual checkpoint confidence head missing')
    tables=[]
    for table in manager.cost_tables:
        values=np.asarray(table)
        if values.size==0 or not np.isfinite(values).all() or (values<0).any():raise AssertionError('invalid adaptive costs')
        tables.append({'shape':list(values.shape),'min':float(values.min()),'max':float(values.max()),
                       'sha256':hashlib.sha256(values.tobytes()).hexdigest()})
    return {'complete':True,'actual_cost_tables':tables,'cudagraph_limit':manager._cudagraph_limit,
            'scope':'actual profiled fixed-base costs/confidence/graph capacity; variable-length sampling exercised by model smokes'}


def run(worker,payload):
    data,owner=authorize(payload)
    if owner.get('reference_plan_sha256'):
        if not re.fullmatch(r'[a-f0-9]{64}',owner['reference_plan_sha256']):raise ValueError('invalid reference provenance')
        data['reference_plan_sha256']=owner['reference_plan_sha256']
    stage=data['stage'];allowed={'contract','R','R-failure','K','D','I','M','P','H','A','references','E'}
    if stage not in allowed:raise ValueError('qualification stage not yet implemented: '+stage)
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed import get_tensor_model_parallel_rank
    runner=worker.model_runner
    meminfo={line.split(':',1)[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
             if line.startswith(('MemAvailable:','SwapTotal:','SwapFree:'))}
    if meminfo['MemAvailable']<2*1024**3:raise RuntimeError('insufficient free host memory for bounded qualification')
    with set_current_vllm_config(worker.vllm_config),torch.no_grad():
        locked=contract(worker)
        needed={'R':'R','R-failure':'R','K':'K','D':'D','I':'I1','M':'M','P':'P','H':'H','A':'A'}
        if stage in needed and needed[stage] not in locked['modules']:
            raise ValueError('qualification stage is not enabled in this variant')
        if stage=='contract':result={'complete':True,**locked}
        elif stage=='R':result=transport_gate(worker)
        elif stage=='R-failure':result=transport_failure_gate(worker)
        elif stage=='K':result=expert_gate(worker)
        elif stage=='P':result=prefetch_gate(worker)
        elif stage=='D':result=markov_gate(worker,data)
        elif stage=='H':result=dense_gate(worker,data)
        elif stage=='A':result=adaptive_gate(worker)
        elif stage=='E':
            from dgx_campaign_runtime.x11_gpu_checks import run as check
            result=check(worker)
        elif stage=='references':
            if data['version']!='0.8.9X1':raise ValueError('stock references only on X1 before K/D/H')
            result={'complete':True,'D':markov_gate(worker,data,record=True),'H':dense_gate(worker,data,record=True)}
        elif stage=='I':
            from dgx_campaign_runtime.qualify_candidates import run as check
            selected=os.environ.get('DGX_CAMPAIGN_INDEXER')
            try:result=check(modes=(int(selected),));result['complete']=result.get('passed') is True
            finally:os.environ['DGX_CAMPAIGN_INDEXER']=selected
        elif stage=='M':
            from dgx_campaign_runtime.qualify_mhc import run as check
            selected=os.environ.get('DGX_CAMPAIGN_M')
            try:result=check()
            finally:os.environ['DGX_CAMPAIGN_M']=selected
        torch.cuda.synchronize()
    # Discard temporary probe allocator blocks after all outputs/references
    # have been checked. Strong model/graph tensors remain allocated.
    torch.cuda.empty_cache()
    return {**result,'rank':get_tensor_model_parallel_rank(),'stage':stage,'nonce':data['nonce'],
            'campaign_id':data['campaign_id'],'plan_sha256':data['plan_sha256'],'version':data['version'],
            'container_id':owner['container_id'],'fingerprint':os.environ['DGX_V089_FINGERPRINT'],
            'qualification_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

_x11_original_load_reference=load_reference
def load_reference(data,name,rows,device):
    if data.get("reference_plan_sha256")!='ffdddb1529a2122fb960381b9fe6f1aeb75f6dbbb91890b5ec15e76ab10320a4':
        raise ValueError("exact retained stock reference plan required")
    reference_data=dict(data,campaign_id="dsv41-expanded-20261008")
    return _x11_original_load_reference(reference_data,name,rows,device)
