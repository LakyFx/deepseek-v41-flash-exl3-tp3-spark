"""Future first-start GPU gate for I1/I2; NOT a model benchmark.

Default invocation does nothing. Requires --armed, candidate module mounted,
same captured SM121 image and a closed/drained campaign execution window.
Any score/index/capture mismatch fails the candidate, not a silent fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def native_context_lengths(batch, depth, width):
    """Match native metadata: the last query owns each request's maximum KV.

    SM121's uniform scheduler reads context_lens[request, -1]. A padded
    request therefore has ALL lengths zero; a zero trailing query beside
    live earlier queries would skip the reference work and read uninitialized
    logits. Preserve short contexts and padding without inventing that state.
    """
    result = []
    for request in range(batch):
        end = width if request == 0 else 509 if request == 1 else width - 11
        if batch > 2 and request == batch - 1:
            end = 0
        result.append([max(end - depth + offset + 1, 0) for offset in range(depth)])
    return result


def same_selected_positions(actual, reference, logits, lengths):
    """Require native exact top-k semantics, including its arbitrary ties.

    Native atomic histogram insertion does not specify output order OR the
    chosen IDs at an equal-score boundary. Stock repeated against itself
    changes both in the all-zero case. Require unique causal IDs, exact count,
    exact selected scores and all strictly better IDs. Different membership
    is legal ONLY among positions whose scores equal the kth boundary exactly.
    This is not an approximate selector or a score-error tolerance.
    """
    import torch
    if actual.shape != reference.shape:
        return False
    for row in range(actual.shape[0]):
        bound = int(lengths.reshape(-1)[row].item())
        count = min(actual.shape[1], bound)
        expected = torch.topk(logits[row, :bound], count, sorted=True).values if count else logits[row, :0]
        selections = []
        for indices in (actual[row], reference[row]):
            if ((indices < -1) | (indices >= bound)).any().item():
                return False
            valid = indices[indices >= 0].long()
            if valid.numel() != count or torch.unique(valid).numel() != count:
                return False
            values = torch.sort(logits[row, valid], descending=True).values
            if not torch.equal(values, expected):
                return False
            selections.append(valid)
        if not count:
            continue
        for first, second in (selections, selections[::-1]):
            different = first[~torch.isin(first, second)]
            if different.numel() and not torch.all(logits[row, different] == expected[-1]).item():
                return False
    return True


def run(modes=(1,2)):
    if not modes or any(mode not in (1,2) for mode in modes):
        raise ValueError('only prepared I1/I2 modes allowed')
    import torch
    from vllm import _custom_ops as ops
    from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits as score, get_paged_mqa_logits_metadata as schedule
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import apply_candidate_mask
    from dgx_campaign_runtime import candidates
    from dgx_campaign_runtime.candidate_kernels import pack_candidates
    if torch.cuda.get_device_capability() != (12, 1):
        raise ValueError("this candidate gate requires DGX Spark SM121")
    device = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=device).manual_seed(4108)
    evidence = []
    # Padded nativeK5 query atoms, C1/C3/C8, context changes and partial blocks.
    for b, next_n, width, zero_scores, varlen in (
        (1, 6, 8192, False, False), (3, 6, 32768, False, False),
        (8, 6, 131072, False, False), (1, 1, 401408, False, False),
        (18, 1, 32768, False, True),
        (3, 6, 32768, True, False)):
        # Same SM12x FP8 page size as production and the native scorer.
        rows, page, heads, dim, k = b * next_n, 64, 32, 128, 2048
        # Physical-page order is deliberately unrelated to logical order.
        pages_per_request = (width + page - 1) // page
        requests = 3 if varlen else b
        pages = requests * pages_per_request
        keys = torch.randn((pages, page, dim), generator=gen, device=device).to(torch.float8_e4m3fn)
        scales = torch.exp2(torch.randint(-4, 5, (pages, page), generator=gen, device=device).float())
        cache = torch.cat((keys.view(torch.uint8).reshape(pages, page * dim),
                           scales.view(torch.uint8).reshape(pages, page * 4)), dim=1).reshape(pages, page, 1, 132)
        table = torch.randperm(pages, generator=gen, device=device).to(torch.int32).reshape(requests, pages_per_request)
        q = torch.randn((b, next_n, heads, dim), generator=gen, device=device).to(torch.float8_e4m3fn)
        if zero_scores:
            q.zero_()
        weights = torch.randn((rows, heads), generator=gen, device=device)
        lens = torch.tensor(native_context_lengths(b, next_n, width), device=device, dtype=torch.int32)
        blocks = torch.randint(0, (width + 7) // 8, (rows, k), generator=gen, device=device, dtype=torch.int32)
        # Guarantee short rows cover every visible key, then exercise duplicate,
        # negative and out-of-range candidates without removing those keys.
        blocks[:, :128] = torch.arange(128, device=device, dtype=torch.int32)
        blocks[:, -4:] = torch.tensor([-1, 0, 0, (width + 7) // 8 + 5], device=device, dtype=torch.int32)
        request_ids = None
        if varlen:
            # Native varlen indexes label adjacent atoms; tables themselves
            # are expanded per query. Exercise pairs and unpaired ragged tails.
            assert next_n == 1
            request_ids = torch.repeat_interleave(torch.arange(requests, device=device, dtype=torch.int32),
                                                   torch.tensor([6, 7, 5], device=device))
            table = table[request_ids.long()].contiguous()
            lens = torch.tensor(list(range(width-5, width+1)) + list(range(503,510)) + [0]*5,
                                device=device, dtype=torch.int32).reshape(rows,1)
        def reference():
            native_schedule = schedule(lens, page, 48, indices=request_ids)
            full = score((q, None), cache, weights, lens, table, native_schedule,
                max_model_len=width, clean_logits=False, indices=request_ids)
            apply_candidate_mask(full, None, lens.reshape(-1), blocks, 8, 1)
            ids = torch.full((rows, 512), -1, device=device, dtype=torch.int32)
            ops.top_k_per_row_decode(full, next_n, lens, ids, rows, full.stride(0), full.stride(1), 512)
            return full, ids

        full, reference_ids = reference()
        positions, packed_cache, packed_table, counts = pack_candidates(cache, table, request_ids,
            lens.reshape(-1), blocks, next_n=next_n, width=width)
        # Verify logical IDs are unique/sorted, padding excluded and byte cache
        # carries BOTH the value bytes and fp32 scale bytes unchanged.
        for row in range(rows):
            n = int(counts[row].item())
            actual = positions[row, :n]
            valid = torch.isfinite(full[row]).nonzero().flatten().to(torch.int32)
            if not torch.equal(actual, valid):
                raise AssertionError("candidate mask/compaction differs from current stock rule")
            if n:
                request = row if varlen else row // next_n
                physical = table[request, actual.long() // page].long()
                offset = actual.long() % page
                raw = cache.reshape(pages, page * 132)
                expected_values = raw[physical[:, None], offset[:, None] * dim + torch.arange(dim, device=device)[None, :]]
                expected_scales = raw[physical[:, None], page * dim + offset[:, None] * 4 + torch.arange(4, device=device)[None, :]]
                packed_raw = packed_cache.reshape(rows, -1, page * 132)[row]
                index = torch.arange(n, device=device)
                actual_values = packed_raw[(index // page)[:, None], (index % page)[:, None] * dim + torch.arange(dim, device=device)[None, :]]
                actual_scales = packed_raw[(index // page)[:, None], page * dim + (index % page)[:, None] * 4 + torch.arange(4, device=device)[None, :]]
                if not torch.equal(actual_values, expected_values) or not torch.equal(actual_scales, expected_scales):
                    raise AssertionError("packed FP8 value/scale bytes changed")

        for selected in modes:
            os.environ["DGX_CAMPAIGN_INDEXER"] = str(selected)
            original_lens = lens.clone()
            original_blocks = blocks.clone()

            def candidate():
                result = candidates.score_decode((q, None), cache, weights, lens, table, request_ids, blocks,
                    max_model_len=width, block_size=8, use_fp4=False, source_layer=False,
                    scorer=score, schedule_builder=schedule)
                ids = torch.full((rows, 512), -1, device=device, dtype=torch.int32)
                if result.positions is None:
                    lengths, depth = lens, next_n
                else:
                    lengths, depth = result.lengths, 1
                ops.top_k_per_row_decode(result.logits, depth, lengths, ids, rows,
                    result.logits.stride(0), result.logits.stride(1), 512)
                if result.positions is not None:
                    candidates.remap_indices(ids, result.positions, result.lengths)
                return result, ids

            result, ids = candidate()
            if selected == 1:
                same_scores = torch.equal(result.logits, full)
            else:
                same_scores = all(torch.equal(result.logits[row, :int(counts[row].item())],
                    full[row, positions[row, :int(counts[row].item())].long()]) for row in range(rows))
            if not same_scores or not same_selected_positions(ids, reference_ids, full, lens):
                from dgx_campaign_runtime.candidate_kernels import scatter_scores
                compared = result.logits if selected == 1 else scatter_scores(result.logits, result.positions, result.lengths.reshape(-1), width)
                finite = torch.isfinite(compared) & torch.isfinite(full)
                score_mismatch = finite & (compared != full)
                locations = score_mismatch.nonzero()[:8]
                repeated_full, repeated_ids = reference()
                canonical_ids = torch.sort(ids, dim=1).values
                canonical_reference = torch.sort(reference_ids, dim=1).values
                detail = {"mode": selected, "rows": rows, "width": width,
                          "next_n": next_n, "ties": zero_scores, "varlen": varlen,
                          "same_scores": same_scores,
                          "score_mismatches": int(score_mismatch.sum().item()),
                          "finite_mask_mismatches": int((torch.isfinite(compared) != torch.isfinite(full)).sum().item()),
                          "max_abs": float((compared[finite] - full[finite]).abs().max().item()) if finite.any().item() else 0.0,
                          "examples": [[int(r), int(c), float(full[r, c]), float(compared[r, c])] for r, c in locations.tolist()],
                          "topk_mismatches": int((ids != reference_ids).sum().item()),
                          "topk_set_mismatches": int((canonical_ids != canonical_reference).sum().item()),
                          "stock_repeat_scores_equal": torch.equal(full, repeated_full),
                          "stock_repeat_order_mismatches": int((reference_ids != repeated_ids).sum().item()),
                          "stock_repeat_set_mismatches": int((canonical_reference != torch.sort(repeated_ids, dim=1).values).sum().item()),
                          "strides": [list(full.stride()), list(compared.stride())],
                          "reference_ids_first": reference_ids[0, :12].tolist(),
                          "candidate_ids_first": ids[0, :12].tolist(),
                          "counts": counts.tolist()}
                raise AssertionError(f"I{selected} scores/topk differ from stock; including ties: " + json.dumps(detail))
            # Warm on a side stream BEFORE graph capture; no uncached compiler
            # work is allowed inside the subsequently timed model benchmark.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    candidate()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_result, graph_ids = candidate()
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            if not same_selected_positions(graph_ids, reference_ids, full, lens):
                raise AssertionError("graph replay differs from stock indices")
            # Replay the SAME graph after changing causal bounds/candidate
            # ownership. Captured CPU values must not freeze device masks.
            changed_end = min(width, 1021)
            lens[0] = torch.tensor([max(changed_end - next_n + offset + 1, 0)
                                    for offset in range(next_n)], device=device, dtype=torch.int32)
            blocks[0, -3:] = torch.tensor([1, 1, -1], device=device, dtype=torch.int32)
            changed_full, changed_ids = reference()
            graph.replay()
            torch.cuda.synchronize()
            if not same_selected_positions(graph_ids, changed_ids, changed_full, lens):
                raise AssertionError("replayed candidate graph froze old causal bounds or candidates")
            if selected == 1 and not torch.equal(graph_result.logits, changed_full):
                raise AssertionError("graph replay scores differ after causal-bound update")
            # Restore the original inputs before evaluating the other arm.
            lens.copy_(original_lens)
            blocks.copy_(original_blocks)
            evidence.append({"rows": rows, "width": width, "mode": selected, "ties": zero_scores,
                             "varlen": varlen, "scores_equal": True, "indices_equal": True,
                             "graph_equal": True, "changed_bounds_replay_equal": True,
                             "indices_contract": "Exact unique causal global top-k IDs/count/scores; differences only at exact kth-score ties where native membership/order is unspecified"})
    return {"schema": "dgx.candidate-indexer-gpu-gate.v1", "checks": evidence,
            "model_requests": 0, "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--armed", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.armed:
        parser.error("GPU gate is unarmed; no device imports or work performed")
    if args.output.exists():
        parser.error("qualification receipts may not overwrite prior evidence")
    receipt = run()
    receipt["qualification_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(receipt, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
