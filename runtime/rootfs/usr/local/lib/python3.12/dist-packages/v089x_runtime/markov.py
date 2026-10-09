"""Full-distribution Markov projection with sharded weights, all ranks receive q.

Greedy distributed argmax and truncated top-k weight indexing are deliberately
not substituted for the existing probabilistic full-vocabulary draft.
"""
import os
import torch

def markov_sharding_enabled():
    if os.environ.get('DGX_V089X_MARKOV_TP') != '1':
        return False
    from vllm.config import get_current_vllm_config
    from vllm.distributed import get_tensor_model_parallel_world_size
    config = get_current_vllm_config().speculative_config
    if config is None or get_tensor_model_parallel_world_size() != 3:
        raise ValueError('sharded Markov requires the TP3 DSpark config')
    hf = config.draft_model_config.hf_config
    if getattr(hf, 'dspark_draft_topk', None) is not None:
        raise ValueError('truncated Markov direct-weight indexing is not qualified for this port')
    return True

def full_markov_bias(head, embedding, processor):
    if head.tp_size == 1:
        return processor(head, embedding)
    from vllm.distributed import tensor_model_parallel_all_gather
    if processor.logits_as_input:
        raise ValueError('Markov requires a projection logits processor')
    local = processor._apply_head(head, embedding, None)
    # CUDA's normal gather may return logits only on rank0. DSpark needs q on
    # EVERY rank for identical proposal and rejection decisions.
    logits = tensor_model_parallel_all_gather(local, dim=-1)[..., :processor.org_vocab_size]
    if processor.soft_cap is not None:
        logits = torch.tanh(logits / processor.soft_cap) * processor.soft_cap
    if processor.scale != 1.0:
        logits *= processor.scale
    return logits
