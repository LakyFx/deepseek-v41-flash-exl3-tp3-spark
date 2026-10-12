"""Future N1/N3 gates use loaded model weights and private test buffers only.

No model, tensor or GPU allocation on import. The worker's authorized RPC
owns the execution. Actual serving request buffers and RNG state are untouched.
"""
from types import MethodType, SimpleNamespace


def adaptive(worker):
    import numpy as np
    from .baseline_methods import BaselineAdaptive
    manager = worker.model_runner.adaptive_verification
    if manager is None or manager.cost_tables is None or manager.num_speculative_steps != 5:
        raise ValueError('actual profiled K5 adaptive manager required')
    candidate = type(manager)
    reports = []
    for requests in range(1, 9):
        for prefill in (False, True):
            for clipped in (0, 2, 5):
                # Use real profiled costs but synthetic request bookkeeping;
                # never replace live scheduler slots or stale-confidence data.
                names = ['gate-'+str(i) for i in range(requests)]
                state = SimpleNamespace(req_id_to_index=dict(zip(names,range(requests))),
                    num_computed_tokens_np=np.full(requests, 0 if prefill else 64000),
                    prefill_len=SimpleNamespace(np=np.full(requests, 64000)))
                common = dict(num_speculative_steps=5, num_bonus_tokens=manager.num_bonus_tokens,
                    _max_total_logits=manager._max_total_logits, cost_tables=manager.cost_tables,
                    req_states=state, _stale_idx=0,
                    _stale_confidences=[SimpleNamespace(np=np.full((requests,5),0.23))])
                before, after = SimpleNamespace(**common), SimpleNamespace(**common)
                non_draft = 3 if prefill else 1
                scheduled = {name:clipped+non_draft for name in names}
                drafts = {name:list(range(clipped)) for name in names}
                expected = BaselineAdaptive.get_num_tokens(before, scheduled, drafts)
                actual = candidate.get_num_tokens(after, scheduled, drafts)
                bound = min(requests*clipped,
                            max(0, manager._max_total_logits-requests*manager.num_bonus_tokens))
                full_c1 = requests == 1 and not prefill and 0 < clipped <= bound
                if full_c1:
                    if actual != 1+clipped or after._batch_budget != (drafts_to_counts(drafts), {names[0]:1}, clipped):
                        raise AssertionError('N1 does not retain every valid C1 draft')
                elif actual != expected or after._batch_budget != before._batch_budget:
                    raise AssertionError('N1 changed prefill/multi-request/clipped behavior')
                capacity = np.full(requests, clipped, dtype=np.int32)
                counts = np.array(list(scheduled.values()), dtype=np.int32)
                offsets = np.concatenate((np.zeros(1,dtype=np.int32),np.cumsum(counts,dtype=np.int32)))
                got = candidate.compact_batch(after, capacity.copy(), counts.copy(), offsets.copy())
                wanted = BaselineAdaptive.compact_batch(after, capacity.copy(), counts.copy(), offsets.copy())
                if any(not np.array_equal(a,b) for a,b in zip(got,wanted)):
                    raise AssertionError('N1 changed compaction contract')
                reports.append({'requests':requests,'prefill':prefill,'scheduled_drafts':clipped,
                                'verified_tokens':actual,'full_valid_c1':full_c1})
    return {'complete':True,'cases':reports,'scope':'loaded cost tables and actual patched CPU budget/compaction; model quality checked separately'}


def drafts_to_counts(drafts):
    return {name:len(tokens) for name,tokens in drafts.items()}


def proxies(spec, requests, *, baseline_class, device=None):
    """Isolated buffers share only read-only weights and gate-owned inputs."""
    import torch
    if (spec.num_speculative_steps != 5 or spec._draft_topk is not None
            or spec.draft_logits is None or not spec.enable_adaptive_verification
            or not 1 <= requests <= spec.max_num_reqs):
        raise ValueError('full-vocabulary probabilistic K5+confidence required')
    device = spec.device if device is None else device
    scratch = getattr(spec, '_next_logits_scratch', None)
    if (scratch is None or tuple(scratch.shape) != (spec.max_num_reqs,spec.vocab_size)
            or scratch.dtype != spec.draft_logits.dtype):
        raise ValueError('actual N3 scratch allocation missing or wrong')
    n = requests*5
    num_query = spec.num_query_per_req
    pos = (torch.arange(n,dtype=torch.int64).reshape(requests,5)
           + torch.arange(requests,dtype=torch.int64)[:,None]*40000 + 8193).flatten().to(device)
    slots = (torch.arange(requests,device=device,dtype=spec.sample_idx_mapping.dtype)+1)%spec.max_num_reqs
    ids = (torch.arange(requests*num_query,device=device,dtype=spec.input_buffers.input_ids.dtype)*17+3)%spec.vocab_size
    common = dict(model=spec.model, _draft_topk=None, num_speculative_steps=5,
        enable_adaptive_verification=True, sample_indices=torch.arange(n,device=device,dtype=spec.sample_indices.dtype),
        sample_idx_mapping=slots.repeat_interleave(5), sample_pos=pos,
        input_buffers=SimpleNamespace(input_ids=ids),
        _anchor_idx=torch.arange(requests,device=device,dtype=torch.int64)*num_query,
        temperature=torch.full((spec.max_num_reqs,),0.6,device=device,dtype=spec.temperature.dtype),
        seeds=torch.tensor([410110+i*31 for i in range(spec.max_num_reqs)],device=device,dtype=spec.seeds.dtype),
        use_fp64_gumbel=spec.use_fp64_gumbel, _step_cols=spec._step_cols,
        _d2t_scatter_index=spec._d2t_scatter_index, _gate_slots=slots.to(torch.long))
    result = []
    for cls in (baseline_class, type(spec)):
        proxy = SimpleNamespace(**common)
        proxy.draft_tokens = torch.empty((spec.max_num_reqs,5),dtype=spec.draft_tokens.dtype,device=device)
        proxy.draft_logits = torch.empty((spec.max_num_reqs,5,spec.vocab_size),dtype=spec.draft_logits.dtype,device=device)
        proxy.draft_token_confidence_probs = torch.empty((spec.max_num_reqs,5),dtype=torch.float32,device=device)
        proxy._next_logits_scratch = torch.empty_like(scratch,device=device)
        proxy._draft_scatter_buf = (torch.full_like(scratch,-torch.inf,device=device)
                                    if spec._d2t_scatter_index is not None else None)
        proxy._sample_logits = MethodType(cls._sample_logits, proxy)
        proxy._sample_sequential = MethodType(cls._sample_sequential, proxy)
        result.append(proxy)
    return result


def outputs(proxy, requests):
    # Gumbel writes logits by global scheduler slot, unlike local-batch tokens
    # and confidence. Exercise a nonidentity slot mapping in every gate case.
    return (proxy.draft_tokens[:requests], proxy.draft_logits.index_select(0,proxy._gate_slots),
            proxy.draft_token_confidence_probs[:requests])


def compare_outputs(actual, expected):
    import torch
    report = {}
    for name,a,b in zip(('tokens','logits','confidence'), actual, expected, strict=True):
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a,b):
            raise AssertionError('N3 changed '+name+' relative to exact baseline sampler')
        if name == 'logits':
            if torch.isnan(a).any() or torch.isposinf(a).any() or not torch.isfinite(a).any(dim=-1).all():
                raise AssertionError('invalid full-vocabulary distribution')
        elif name == 'confidence':
            if not torch.isfinite(a).all() or (a < 0).any() or (a > 1).any():
                raise AssertionError('invalid draft confidence')
        report[name] = {'bit_equal':True,'shape':list(a.shape)}
    return report


def draft(worker):
    import torch
    from vllm.distributed import get_tp_group
    from .baseline_methods import BaselineDSpark
    spec = worker.model_runner.speculator
    handle = get_tp_group().device_communicator.dgx_campaign_r
    if handle is None or handle.disabled:
        raise ValueError('qualified Sparknet graph transport required')
    reports = []
    for requests in (1,3,8):
        baseline, candidate = proxies(spec, requests, baseline_class=BaselineDSpark)
        gen = torch.Generator(device='cpu').manual_seed(410113+requests)
        hidden = (torch.randn((requests*5,spec.hidden_states.shape[-1]),generator=gen)*0.25).to(
            device=spec.device,dtype=spec.hidden_states.dtype)
        def reference():
            baseline._sample_sequential(requests,hidden)
            return outputs(baseline,requests)
        def call():
            candidate._sample_sequential(requests,hidden)
            return outputs(candidate,requests)
        eager = compare_outputs(call(),reference())
        with handle.capture():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream): call(); call()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): captured = call()
            replays = []
            for _ in range(3):
                hidden.mul_(-1)
                # Both proxies share these gate-owned inputs, never serving
                # buffers. Change the RNG position key and preceding token.
                candidate.sample_pos.add_(7)
                candidate.input_buffers.input_ids.add_(1).remainder_(spec.vocab_size)
                wanted = reference()
                graph.replay()
                torch.cuda.synchronize()
                replays.append(compare_outputs(captured,wanted))
        reports.append({'requests':requests,'eager':eager,'graphs':replays})
        del graph, captured, wanted, baseline, candidate, hidden
    return {'complete':True,'cases':reports,
            'scope':'real loaded draft/Markov/confidence weights, actual stateless gumbel, private buffers, exact logits/tokens/confidence and changed-input graph replays'}
