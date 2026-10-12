"""Future N8 test on real packed MXFP8 weights, with no second weight copy."""
from dataclasses import replace

ROWS = (1,2,3,4,5,6,7,8,18,48,256,4096)


def baseline_query(query):
    rows = max(1,query.num_tokens,query.expected_m or 1)
    return replace(query,num_tokens=1 << (rows-1).bit_length(),expected_m=None)


def validate_selection(query, before, after, split):
    from dataclasses import asdict
    if split not in (1,2) or query.recipe!='mxfp8':
        raise ValueError('N8 exact MXFP8 split1/2 gate required')
    if query.num_tokens<=6:
        if (before.mode!='a16' or before.split_k!=4 or after.mode!='a16'
                or after.split_k!=split or before.tile_n!=after.tile_n
                or before.tile_k!=after.tile_k or before.tile_m!=after.tile_m):
            raise AssertionError('N8 did not select the requested small-row split')
    elif asdict(before)!=asdict(after):
        raise AssertionError('N8 changed a larger/prefill launch policy')


def run(worker):
    import torch
    from b12x.gemm import mxfp8_linear as api
    from b12x.gemm.blockscaled._preparation import heuristic_plan
    from b12x.preparation import PreparationSession, PreparedCall, detect_device
    from dgx_campaign_runtime.worker_checks import dense_layers, numeric, graph_replays
    from .dense_policy import SPLIT
    from .dense_reference import _default_config
    reports = []
    for (k,n),(prefix,layer,method) in sorted(dense_layers(worker).items()):
        packed = getattr(layer,'b12x_mxfp8_packed_weight',None)
        if packed is None:
            raise AssertionError('N8 actual layer has no packed MXFP8 weight: '+prefix)
        device = packed.weight.values.device
        identity = detect_device(device).identity
        for rows in ROWS:
            generator = torch.Generator(device='cpu').manual_seed(4180+k+n+rows)
            x = (torch.randn((rows,k),generator=generator)*.25).to(device=device,dtype=torch.bfloat16)
            query = api.query_from_call(x,packed,expected_m=rows)
            old_query = baseline_query(query)
            old_config = _default_config(old_query,identity)
            # Explicit override prevents patched heuristics/autotune from
            # turning the reference into another candidate invocation.
            old_plan = api.plan(old_query,override=old_config)
            with PreparationSession(device=device,autotune=False,compile_workers=0) as session:
                def reference():
                    return api.mm(x,packed,plan=old_plan,bias=None,expected_m=rows)
                request = old_plan.request(name='N8-baseline',prepare_call=lambda _:PreparedCall(run=reference))
                session.prepare((request,),autotune=False)
                def call():return method.apply(layer,x,bias=None)
                actual,expected = call(),reference()
                current = heuristic_plan(query,device)
                if current.selection is None:
                    raise AssertionError('actual B12X invocation did not materialize its plan')
                validate_selection(query,old_config,current.selection.config,SPLIT)
                if actual.shape!=(rows,n):
                    raise AssertionError('N8 actual dense geometry changed')
                # Split-K changes FP32 reduction order. Bound the numerical
                # error; untouched larger batches must remain bit-identical.
                exact = rows>6
                result = numeric(actual,expected,limit=.003,exact=exact)
                replays = graph_replays(call,reference,lambda _:x.mul_(-1),limit=.003)
                if exact and any(not r['bit_equal'] for r in replays):
                    raise AssertionError('N8 changed large-row graph results')
                reports.append({'prefix':prefix,'shape':[n,k],'rows':rows,
                    'baseline_split':old_config.split_k,'candidate_split':current.selection.config.split_k,
                    'numerical':result,'graphs':replays})
                # Session owns only private reference plans. Actual heuristic
                # plans/weights are never released or replaced by this gate.
                del actual,expected
    if not reports:raise AssertionError('N8 real layer inventory empty')
    return {'complete':True,'cases':reports,
            'scope':'same real packed weights; original capacity/default config vs actual layer; all small rows, decode/prefill boundaries and changed-input graphs'}
