# SPDX-License-Identifier: Apache-2.0
# Exact methods from the pinned vLLM X11c baseline, used only by test gates.
from __future__ import annotations
import numpy as np
import torch
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

class BaselineAdaptive:

    def get_num_tokens(self, num_tokens_per_req: dict[str, int], draft_tokens: dict[str, list[int]]) -> int:
        """Token count once the draft budget is trimmed to fit.

        Stashes the chosen budget in ``_batch_budget`` for the compaction and
        reallocation that follow in the same step.
        """
        assert self.cost_tables is not None
        req_ids = list(num_tokens_per_req)
        num_reqs = len(req_ids)
        scheduled_tokens = np.fromiter(num_tokens_per_req.values(), dtype=np.int32, count=num_reqs)
        scheduled_drafts = np.fromiter((len(draft_tokens.get(req_id, ())) for req_id in req_ids), dtype=np.int32, count=num_reqs)
        num_non_draft_tokens = scheduled_tokens - scheduled_drafts
        slots = np.fromiter((self.req_states.req_id_to_index[req_id] for req_id in req_ids), dtype=np.int32, count=len(req_ids))
        stale_confidences = self._stale_confidences[self._stale_idx].np[slots]
        survival_probability = np.cumprod(stale_confidences.astype(np.float64), axis=1)
        steps = np.arange(self.num_speculative_steps)
        valid = steps[None, :] < scheduled_drafts[:, None]
        scores = np.sort(survival_probability[valid])[::-1]
        num_non_draft_tokens_total = int(num_non_draft_tokens.sum())
        max_draft_budget = min(int(scheduled_drafts.sum()), max(0, self._max_total_logits - num_reqs * self.num_bonus_tokens))
        scores = scores[:max_draft_budget]
        draft_cost_ms, verify_cost_ms = self.cost_tables
        num_sampling_requests = np.count_nonzero(self.req_states.num_computed_tokens_np[slots] + num_non_draft_tokens >= self.req_states.prefill_len.np[slots])
        num_tokens_to_estimated_accepted_tokens = np.concatenate(([num_sampling_requests], num_sampling_requests + np.cumsum(scores)))
        costs = draft_cost_ms[len(req_ids)] + verify_cost_ms[num_non_draft_tokens_total:num_non_draft_tokens_total + max_draft_budget + 1]
        num_drafts_per_req = {req_id: int(num_drafts) for req_id, num_drafts in zip(req_ids, scheduled_drafts, strict=True)}
        num_non_draft_tokens_per_req = {req_id: int(num_tokens) for req_id, num_tokens in zip(req_ids, num_non_draft_tokens, strict=True)}
        draft_budget = int(np.argmax(num_tokens_to_estimated_accepted_tokens / costs))
        self._batch_budget = (num_drafts_per_req, num_non_draft_tokens_per_req, draft_budget)
        return sum(num_non_draft_tokens_per_req.values()) + draft_budget

    def compact_batch(self, num_draft_tokens_per_req: np.ndarray, num_scheduled_tokens: np.ndarray, cu_num_logits_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compact the CPU batch to the chosen draft budget.

        Returns the compacted per-request token counts and the CPU cu_num_logits_np.
        If the draft budget is 0, we can know cu_num_logits_np exactly, otherwise
        its unchanged/an-upper-bound.
        """
        batch_budget = self._batch_budget
        assert batch_budget is not None
        _, _, draft_budget = batch_budget
        num_drafts = int(num_draft_tokens_per_req.sum())
        if draft_budget == num_drafts:
            return (num_scheduled_tokens, cu_num_logits_np)
        num_non_draft_tokens = num_scheduled_tokens - num_draft_tokens_per_req
        if draft_budget == 0:
            num_reqs = num_scheduled_tokens.shape[0]
            cu_num_logits_np = np.arange(num_reqs + 1, dtype=cu_num_logits_np.dtype) * self.num_bonus_tokens
            return (num_non_draft_tokens, cu_num_logits_np)
        is_verification_request = num_draft_tokens_per_req > 0
        num_verification_reqs = int(is_verification_request.sum())
        assert np.all(is_verification_request[:num_verification_reqs])
        draft_lens_cpu = np.zeros_like(num_non_draft_tokens)
        draft_lens_cpu[:num_verification_reqs] = draft_budget // num_verification_reqs
        draft_lens_cpu[:draft_budget % num_verification_reqs] += 1
        return (num_non_draft_tokens + draft_lens_cpu, cu_num_logits_np)

class BaselineDSpark:

    def _sample_logits(self, logits: torch.Tensor, idx_map: torch.Tensor, sample_pos: torch.Tensor, step: int) -> torch.Tensor:
        if self.draft_logits is None:
            return self.model.map_draft_to_target(logits.argmax(dim=-1))
        if self._d2t_scatter_index is not None:
            assert self._draft_scatter_buf is not None
            buf = self._draft_scatter_buf[:logits.shape[0]]
            buf.index_copy_(1, self._d2t_scatter_index, logits.to(buf.dtype))
            logits = buf
        return gumbel_sample(logits, idx_map, self.temperature, self.seeds, sample_pos - 1, apply_temperature=True, is_drafting=True, logits_cache=self.draft_logits, logits_cache_col=self._step_cols[step], use_fp64=self.use_fp64_gumbel)

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        if self._draft_topk is not None:
            self._sample_sequential_topk(num_reqs, head_hidden)
            return
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)
        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        for i in range(n_spec):
            markov_embed = self.model.markov_embed(prev)
            if self.enable_adaptive_verification:
                confidence_markov_embeds.append(markov_embed)
            bias = self.model.markov_bias(markov_embed)
            logits_i = base_logits[:, i] + bias
            draft_sampled_i = self._sample_logits(logits_i, idx_map[:, i], sample_pos[:, i], i)
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i
        if self.enable_adaptive_verification:
            confidence = self.model.compute_confidence(sample_hidden, torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1))
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(num_reqs, n_spec)
