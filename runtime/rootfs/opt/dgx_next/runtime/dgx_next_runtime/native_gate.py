"""Future N4 qualification on loaded weights; imported without CUDA effects.

Invoked only by the closed-window worker RPC after user test authorization.
Based on the existing expert_gate. Unlike its BM16 decode cases, these cases
exercise every new BM32/64/128 specialization with >48 input rows.
"""
import os

CASES = ((64, 32), (257, 64), (513, 128))


def run(worker):
    import torch
    ops = torch.ops.cuda_exl3_C
    previous_cap = ops.exl3_get_moe_acc_cap()
    try:
        # The dispatcher selects N4 only for unsplit projections. Isolate that
        # arithmetic explicitly; never silently qualify the fallback kernel.
        ops.exl3_set_moe_acc_cap(0)
        return _run(worker)
    finally:
        try:
            torch.cuda.synchronize()
        finally:
            ops.exl3_set_moe_acc_cap(previous_cap)


def _run(worker):
    import torch
    from cuda_exl3.moe import _glu_had_in
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    from dgx_campaign_runtime.worker_checks import numeric, graph_replays

    if os.environ.get('CUDA_EXL3_NEXT_PREFILL') != '1':
        raise RuntimeError('N4 branch must be selected before loading the extension')
    layers = {}
    for name, layer in worker.model_runner.model.named_modules():
        if not hasattr(layer, 'w13_trellis'):
            continue
        signature = (int(layer.w13_trellis.shape[-1]//16),
                     int(layer.w2_trellis.shape[-1]//16),
                     int(layer.exl3_inter), int(layer.exl3_cb))
        layers.setdefault(signature, (name, layer))
    if not layers or not {3, 4} <= {bits for sig in layers for bits in sig[:2]}:
        raise AssertionError('actual mixed K3/K4 expert coverage required')
    ops, reports = torch.ops.cuda_exl3_C, []
    for signature, (name, layer) in sorted(layers.items()):
        if signature[2:] != (768, 2) or any(b not in (3, 4) for b in signature[:2]):
            raise AssertionError('unreviewed expert dimensions or codebook')
        h, inter, experts = 5120, layer.exl3_inter, layer.exl3_num_experts
        for tokens, block in CASES:
            rng = torch.Generator(device='cpu').manual_seed(4111+tokens)
            x = torch.randn((tokens, h), generator=rng).to(
                device=layer.w13_trellis.device, dtype=torch.bfloat16)
            ids = (torch.arange(tokens*8, device=x.device).reshape(tokens, 8)
                   % min(experts, 7)).int()
            weights = torch.softmax(torch.randn((tokens, 8), generator=rng), -1).to(x.device)
            sorted_ids, expert_ids, nrows = moe_align_block_size(
                ids, block, experts, pad_sorted_ids=True)
            sorted_ids, expert_ids, nrows = sorted_ids.int(), expert_ids.int(), nrows.int()
            rows = min(expert_ids.numel()*block, sorted_ids.numel())
            expert_ids = expert_ids[:rows//block]
            a13 = torch.empty((2, rows, h), device=x.device, dtype=torch.half)
            a2 = torch.empty((1, rows, inter), device=x.device, dtype=torch.half)

            def project(routed, fused=False):
                ops.exl3_moe_had_in(x, a13, layer.w13_suh.data, sorted_ids,
                    expert_ids, nrows, block, 8, tokens*8, False)
                mid = ops.exl3_moe_gemm(a13, layer.w13_trellis.data,
                    layer.w13_suh.data, layer.w13_svh.data, expert_ids, nrows,
                    [inter, inter], 2, block, torch.bfloat16,
                    sorted_ids if routed else None, None, tokens, 8)
                _glu_had_in(ops, mid, a2, layer.w2_suh.data, expert_ids, nrows, block, 10.0)
                value = ops.exl3_moe_gemm(a2, layer.w2_trellis.data,
                    layer.w2_suh.data, layer.w2_svh.data, expert_ids, nrows,
                    [h], 2, block, torch.bfloat16, sorted_ids if routed else None,
                    weights if fused else None, tokens, 8)
                return value if fused else ops.exl3_moe_combine(
                    value, sorted_ids, weights, tokens, expert_ids, block)

            # Missing sorted_ids excludes N4 in the reference projection.
            # The same fixed-order combine isolates the GEMM change; the
            # pre-existing BF16 atomic fused combine has its own tolerance.
            report = {'layer': name, 'bits': list(signature[:2]),
                      'tokens': tokens, 'block_m': block,
                      'numeric': numeric(project(True), project(False)),
                      'graphs': graph_replays(lambda: project(True),
                          lambda: project(False), lambda _: x.neg_()),
                      'fused': numeric(project(True, True), project(False), limit=.01),
                      'fused_graphs': graph_replays(lambda: project(True, True),
                          lambda: project(False), lambda _: x.neg_(), limit=.01)}
            reports.append(report)
    return {'complete': True, 'cases': reports,
            'scope': 'K3/K4 real expert tensors, all new prefill BM tiers; unchanged stock tile reference',
            'speedup_claimed': False}
