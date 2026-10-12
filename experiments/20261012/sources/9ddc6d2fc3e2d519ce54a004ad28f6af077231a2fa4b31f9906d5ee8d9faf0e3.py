"""Future N6 live topology and private replay-metadata kernel qualification."""
from .replay_contract import validate_model, validate_inner_graphs, INNER_SIZES


def captured_inner_sizes(manager):
    # PIECEWISE replay lives in the breakable wrapper, not manager.graphs
    # (which contains only FULL graphs). Inspect completed captures themselves.
    wrapper=getattr(manager,'breakable_cg_runner',None)
    entries=getattr(wrapper,'entries',None)
    if not entries:raise AssertionError('N6 actual inner captures missing')
    sizes=[]
    for descriptor,entry in entries.items():
        capture=entry.capture
        if (entry.batch_descriptor!=descriptor or descriptor.num_active_loras!=0
                or capture is None or capture.num_graphs<1 or not capture.segments
                or entry.output is None or not entry.input_addresses):
            raise AssertionError('N6 incomplete breakable inner capture')
        sizes.append(descriptor.num_tokens)
    if sorted(sizes)!=INNER_SIZES:
        raise AssertionError('N6 actual inner captures differ from bounded plan')
    return sorted(sizes)


def contract(worker):
    runner=worker.model_runner
    config=worker.vllm_config
    models=[m for m in runner.model.modules() if hasattr(m,'decoder_replay_start')]
    if len(models)!=1:raise AssertionError('one actual replay target required')
    model=models[0]
    validate_model(model.config,runner.speculator.draft_model_config.hf_config,
                   config.parallel_config,model.decoder_replay_start,config.cache_config)
    validate_inner_graphs(config.compilation_config,runner.max_num_tokens,runner.max_num_reqs)
    state=runner.model_state
    layers=state.decoder_replay_layers
    manager=state.replay_cudagraphs
    if (layers is None or layers is not model.decoder_replay_layers or layers.window!=128
            or layers.trim_threshold!=768 or manager is None or manager._graphs_captured is not True
            or manager.layers is not layers or manager.input_buffers is None
            or len(manager.input_buffers)!=7 or layers.replay_batch is not None):
        raise AssertionError('N6 actual compact tail/inner graphs not ready and idle')
    captured=captured_inner_sizes(manager)
    buffers=manager.input_buffers
    expected=((1024,5120),(1024,),(1024,),(1024,4),(1024,4,1),(1024,4,4),(1024,4,5120))
    if tuple(tuple(b.shape) for b in buffers)!=expected:
        raise AssertionError('N6 static input buffer geometry changed')
    if tuple(model.aux_hidden_state_layers)!=(37,38,39):
        # These are actual layer IDs selected by DSpark. Never infer them from
        # the checkpoint alone if loading rewrote the late-hidden consumer.
        raise AssertionError('N6 actual late-hidden consumer changed')
    if state._replay is None or state._replay[0]!=128:
        raise AssertionError('N6 real KV replay groups not initialized')
    return {'cut':model.decoder_replay_start,'window':layers.window,'inner_captured':captured,
            'static_input_bytes':sum(b.numel()*b.element_size() for b in buffers),
            'aux_hidden_layers':list(model.aux_hidden_state_layers)}


def oracle(counts, kept, starts, replay_starts, *, window=128):
    """Independent list oracle: retain each request's final requested rows."""
    rows=[];positions=[];bounds=[0];kv=[];offset=0
    for count,n,start,replay in zip(counts,kept,starts,replay_starts,strict=True):
        if not 0<n<=count:raise ValueError('invalid retained rows')
        rows.extend(range(offset+count-n,offset+count))
        positions.extend(range(start+count-n,start+count))
        bounds.append(bounds[-1]+n)
        kv.append(max(replay,start+count-window) if n<count else replay)
        offset+=count
    return rows,positions,bounds,kv


def metadata_kernels(device):
    import torch
    from vllm.models.deepseek_v4_1.nvidia.model_state import _gather_replay_batch_kernel, _pad_replayed_slots_kernel
    reports=[]
    for counts,kept in (((512,),(128,)),((900,600,36),(128,128,36)),
                         ((512,)*8,(128,)*8),((129,128,127),(128,128,127))):
        requests=len(counts);total=sum(counts);retained=sum(kept)
        starts=[8192+i*4096 for i in range(requests)]
        replay_starts=[start if i%2 else 0 for i,start in enumerate(starts)]
        bounds=[0];drop=[0]
        for count,n in zip(counts,kept):bounds.append(bounds[-1]+count);drop.append(drop[-1]+count-n)
        def i32(value):return torch.tensor(value,device=device,dtype=torch.int32)
        original_positions=[p for start,count in zip(starts,counts) for p in range(start,start+count)]
        query=i32(bounds);dropped=i32(drop);seq=i32([s+n for s,n in zip(starts,counts)])
        replay=i32(replay_starts)
        positions=torch.tensor(original_positions,device=device,dtype=torch.int64)
        slots=torch.arange(3*total,device=device,dtype=torch.int64).reshape(3,total)
        rows=torch.empty(retained,device=device,dtype=torch.int64)
        out_q=i32([0]*(requests+1));out_pos=torch.empty_like(rows)
        out_slots=torch.empty((3,retained),device=device,dtype=torch.int64);out_kv=i32([0]*requests)
        def call():
            _gather_replay_batch_kernel[(requests,)](query,dropped,seq,replay,positions,
                slots,slots.stride(0),rows,out_q,out_pos,out_slots,out_slots.stride(0),out_kv,
                128,NUM_GROUPS=3,BLOCK=128)
        def check():
            indices,pos,bounds2,kv=oracle(counts,kept,starts,replay_starts)
            expected=[torch.tensor(v,device=device,dtype=t.dtype)
                      for v,t in zip((indices,pos,bounds2,kv),(rows,out_pos,out_q,out_kv))]
            if any(not torch.equal(a,b) for a,b in zip((rows,out_pos,out_q,out_kv),expected)):
                raise AssertionError('N6 retained replay rows/positions/bounds changed')
            if not torch.equal(out_slots,slots.index_select(1,expected[0])):
                raise AssertionError('N6 replay KV slot gathering changed')
        call();check()
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):call();call()
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):call()
        for _ in range(3):
            starts=[s+1 for s in starts];positions.add_(1);seq.add_(1)
            graph.replay();torch.cuda.synchronize();check()
        # Prefix-cacheable groups must suppress replay writes only inside the
        # window; an unrelated group and every other position stay intact.
        protected=slots.clone();groups=i32([0,2])
        _pad_replayed_slots_kernel[(requests,)](protected,protected.stride(0),groups,
            query,positions,replay,128,-1,NUM_GROUPS=2,BLOCK=128)
        wanted=slots.clone()
        for request,(start,count,replay_start) in enumerate(zip(starts,counts,replay_starts)):
            if not replay_start:continue
            for pos in range(start,start+count):
                if replay_start<=pos<replay_start+128:
                    at=bounds[request]+pos-start
                    wanted[0,at]=wanted[2,at]=-1
        if not torch.equal(protected,wanted):raise AssertionError('N6 replay suppression touched wrong cache slots')
        reports.append({'requests':requests,'input_rows':total,'kept_rows':retained,
                        'row_slot_oracle_exact':True,'changed_input_graph_replays':3,'cache_write_mask_exact':True})
    return reports


def run(worker):
    return {'complete':True,'topology':contract(worker),
            'metadata_cases':metadata_kernels(worker.model_runner.device),
            'scope':'actual topology/captured inner graphs/private metadata kernels; full late-layer, partial-prefix and vision correctness require the common request quality gate'}
