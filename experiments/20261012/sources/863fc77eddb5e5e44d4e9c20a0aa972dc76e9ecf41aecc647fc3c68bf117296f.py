"""Closed-window current-worker qualification for the variable-width draft port.

The outer worker RPC already attests ownership, config, generation and CUDA
device. Test actual GPU sampler/cache/confidence stride behavior against
contiguous fixed-width references; preserve all persistent buffers afterward.
Full-backbone capture and scheduler transitions are exercised by owned smokes.
"""
def run(worker):
    import torch
    from contextlib import contextmanager
    spec=worker.model_runner.speculator
    if not getattr(type(spec),'_x11_variable_depth_installed',False):
        raise AssertionError('dynamic draft port absent')
    if set(spec._x11_managers)!={3,4,5} or spec.num_speculative_steps!=5:
        raise AssertionError('three draft managers/max storage not restored')
    if any(spec._x11_depth_counts[k]<=0 for k in (3,4,5)):
        raise AssertionError('actual sampled serving did not exercise all draft depths')
    saved={name:value.clone() for name,value in spec._x11_storage.items()}
    extra_names=('sample_indices','sample_pos','sample_idx_mapping','temperature','seeds')
    extra={name:getattr(spec,name).clone() for name in extra_names}
    input_ids=spec.input_buffers.input_ids.clone()
    reports=[]
    @contextmanager
    def contiguous(depth):
        old={name:getattr(spec,name) for name in ('draft_tokens','draft_logits','draft_token_confidence_probs')}
        try:
            for name,value in old.items():setattr(spec,name,torch.empty_like(value).contiguous())
            yield
        finally:
            for name,value in old.items():setattr(spec,name,value)
    try:
        spec.temperature.fill_(0.6);spec.seeds.fill_(4109)
        spec.input_buffers.input_ids.zero_()
        hidden=int(spec.draft_model_config.hf_config.hidden_size)
        for depth in (3,4,5):
            for num_reqs in (1,3,8):
                count=num_reqs*depth
                spec.sample_indices[:count].copy_(torch.arange(count,device=spec.device))
                spec.sample_pos[:count].copy_(torch.arange(depth,device=spec.device).repeat(num_reqs)+1)
                spec.sample_idx_mapping[:count].copy_(torch.arange(num_reqs,device=spec.device).repeat_interleave(depth))
                head=torch.randn((count,hidden),device=spec.device,dtype=spec.dtype)*0.05
                with spec._x11_selected(depth):
                    def call():
                        spec._sample_sequential(num_reqs,head)
                        return (spec.draft_tokens[:num_reqs],spec.draft_logits[:num_reqs],
                                spec.draft_token_confidence_probs[:num_reqs])
                    def reference():
                        with contiguous(depth):return tuple(x.clone() for x in call())
                    expected=reference();actual=tuple(x.clone() for x in call())
                    if not all(torch.equal(a,b) for a,b in zip(actual,expected)):
                        raise AssertionError('prefix-stride sampler/logit/confidence differs from fixed-width reference')
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):call();call()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):outputs=call()
                    for replay in range(2):
                        head.mul_(-1)
                        expected=reference();graph.replay();torch.cuda.synchronize()
                        if not all(torch.equal(a,b) for a,b in zip(outputs,expected)):
                            raise AssertionError('variable-width graph replay differs with changed GPU inputs')
                    reports.append({'depth':depth,'num_reqs':num_reqs,'draft_rows':count,
                        'exact_logits_tokens_confidence':True,'changed_input_graph_replays':2})
        return {'complete':True,'cases':reports,'observed_depth_counts':dict(spec._x11_depth_counts),
            'scope':'GPU fixed-width vs prefix-stride sampler/confidence/logit equivalence and changed-input sampler graphs; full-backbone/scheduler transitions exercised in actual serving smokes'}
    finally:
        for name,value in saved.items():spec._x11_storage[name].copy_(value)
        for name,value in extra.items():getattr(spec,name).copy_(value)
        spec.input_buffers.input_ids.copy_(input_ids)

