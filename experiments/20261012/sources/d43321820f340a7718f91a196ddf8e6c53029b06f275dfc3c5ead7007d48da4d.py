"""Future phase-sensitive N2 check, using loaded mHC parameters/private inputs."""
from types import SimpleNamespace

CASES = ((1,'decode'),(3,'decode'),(5,'decode'),(6,'decode'),
         (1,'prefill'),(6,'prefill'),(6,'mixed'),
         (7,'decode'),(18,'decode'),(48,'decode'),(256,'prefill'))


def metadata(phase):
    if phase not in ('decode','prefill','mixed'):raise ValueError('unknown mHC test phase')
    return {'gate_swa':SimpleNamespace(num_decodes=int(phase!='prefill'),
                                      num_prefills=int(phase!='decode')),
            'gate_indexer':object()}


def compare(actual, expected, numeric, *, exact):
    if len(actual)!=5 or len(expected)!=5:raise AssertionError('mHC five-state contract changed')
    return {name:numeric(a,b,limit=.003,exact=exact)
            for name,a,b in zip(('residual','post','comb','hidden','premix'),actual,expected,strict=True)}


def run(worker):
    import torch
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang, mhc_pre_delayed_tilelang
    from dgx_campaign_runtime.mhc import post_pre as candidate
    from dgx_campaign_runtime.worker_checks import numeric
    from .mhc_original import post_pre as original
    from .policy import small_decode_mhc
    layers = [m for m in worker.model_runner.model.modules() if hasattr(m,'hc_attn_fn')]
    if not layers:raise AssertionError('actual mHC layer inventory missing')
    # Two separated actual trained parameter sets, no weight copies.
    selected = [layers[0],layers[-1]]
    reports = []
    for index,layer in enumerate(selected):
        if layer.hc_attn_fn.shape!=(24,20480) or layer.hc_sinkhorn_iters!=20:
            raise AssertionError('mHC target topology changed')
        device = layer.hc_attn_fn.device
        for rows,phase in CASES:
            gen = torch.Generator(device='cpu').manual_seed(4120+rows+index)
            def rand(shape):return torch.randn(shape,generator=gen).to(device)
            x = (rand((rows,5120))*.25).to(torch.bfloat16)
            residual = (rand((rows,4,5120))*.25).to(torch.bfloat16)
            previous_post = torch.sigmoid(rand((rows,4,1)))
            previous_comb = torch.softmax(rand((rows,4,4)),dim=-1)
            incoming = torch.sigmoid(rand((rows,4)))
            args = (x,residual,previous_post,previous_comb,layer.hc_attn_fn,
                    layer.hc_attn_scale,layer.hc_attn_base,layer.rms_norm_eps,
                    layer.hc_eps,layer.hc_eps,layer.hc_post_alpha,layer.hc_sinkhorn_iters)
            kwargs = dict(pre_mix=incoming,norm_weight=layer.attn_norm.weight,
                          norm_eps=layer.attn_norm.variance_epsilon)
            context = metadata(phase)
            small = small_decode_mhc(rows,context)
            def stock():
                updated = mhc_post_tilelang(x,residual,previous_post,previous_comb)
                post,comb,y,pre = mhc_pre_delayed_tilelang(updated,*args[4:],**kwargs)
                return updated,post,comb,y,pre
            def reference():return original(*args,**kwargs)
            def call():return candidate(*args,**kwargs)
            with set_forward_context(context,worker.vllm_config,num_tokens=rows):
                value = call()
                baseline = compare(value,reference(),numeric,exact=not small)
                selected_path = compare(value,stock() if small else reference(),numeric,exact=True)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):call();call()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):captured=call()
                replays=[]
                for _ in range(3):
                    x.mul_(-1);residual.mul_(-1)
                    expected = stock() if small else reference()
                    graph.replay();torch.cuda.synchronize()
                    replays.append(compare(captured,expected,numeric,exact=True))
            reports.append({'layer_sample':index,'rows':rows,'phase':phase,'stock_path':small,
                            'baseline':baseline,'selected_path':selected_path,'graphs':replays})
            del graph,captured,expected,value
    return {'complete':True,'cases':reports,
            'scope':'loaded trained mHC parameters, private five-state inputs, phase routing, exact selected path and bounded baseline discrepancy; actual model quality checked separately'}
