# SPDX-License-Identifier: Apache-2.0
"""Device-only candidate compaction; current segregated FP8 cache format.

These kernels do not compute dot products. Scores come from the SAME installed
DeepGEMM SM12x kernel, so input packing is the only arithmetic-path change.
Prefix order is original-column order; duplicate candidates have one owner.
"""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _owners(C, O, CS: tl.constexpr, OS: tl.constexpr, K: tl.constexpr,
            WIDTH: tl.constexpr, N: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    i = tl.program_id(1) * TILE + tl.arange(0, TILE)
    block = tl.load(C + row * CS + i, i < K, other=-1)
    target = tl.minimum(block, N)
    # Exactly the stock mask's sentinel: any nonnegative block whose first
    # column lies past the score width retains ONLY width-1, when visible.
    target = tl.where(block * 8 >= WIDTH, N, target)
    tl.atomic_min(O + row * OS + target, i, (i < K) & (block >= 0))


@triton.jit
def _flags(O, L, F, OS: tl.constexpr, FS: tl.constexpr,
           K: tl.constexpr, N: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    i = tl.program_id(1) * TILE + tl.arange(0, TILE)
    end = tl.load(L + row)
    owner = tl.load(O + row * OS + i, i < N, other=K)
    tl.store(F + row * FS + i, ((owner < K) & (i * 8 < end)).to(tl.int32), i < N)


@triton.jit
def _positions(C, O, P, L, IDS, CS: tl.constexpr, OS: tl.constexpr,
               PS: tl.constexpr, IS: tl.constexpr, K: tl.constexpr,
               WIDTH: tl.constexpr, N: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    flat = tl.program_id(1) * TILE + tl.arange(0, TILE)
    ci, within = flat // 8, flat % 8
    block = tl.load(C + row * CS + ci, ci < K, other=-1)
    safe = tl.maximum(0, tl.minimum(block, N - 1))
    owner = tl.load(O + row * OS + safe)
    prefix = tl.load(P + row * PS + safe)
    end = tl.load(L + row)
    col = block * 8 + within
    valid = (ci < K) & (block >= 0) & (block < N) & (owner == ci) & (col < end) & (col < WIDTH)
    packed = (prefix - 1) * 8 + within
    tl.store(IDS + row * IS + packed, col, valid)


@triton.jit
def _counts(O, P, L, IDS, COUNT, OS: tl.constexpr, PS: tl.constexpr,
            IS: tl.constexpr, K: tl.constexpr, N: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    end = tl.load(L + row)
    blocks = tl.load(P + row * PS + N - 1)
    last = tl.maximum(0, (end - 1) // 8)
    selected = (tl.load(O + row * OS + last) < K) & (end > 0)
    # Only the final visible block can be partial. Other selected blocks are
    # full and have no holes; therefore compact sequence lengths are exact.
    count = blocks * 8 - tl.where(selected, (8 - end % 8) % 8, 0)
    edge_owner = tl.load(O + row * OS + N)
    add_edge = (end == WIDTH) & (edge_owner < K) & ~selected
    tl.store(IDS + row * IS + count, WIDTH - 1, add_edge)
    tl.store(COUNT + row, count + add_edge.to(tl.int32))


@triton.jit
def _gather(CACHE, TABLE, REQUESTS, IDS, COUNT, OUT,
            KS: tl.constexpr, TS: tl.constexpr, IS: tl.constexpr,
            PAGE: tl.constexpr, OUT_PAGE: tl.constexpr, PAGES: tl.constexpr, NEXT: tl.constexpr,
            VARLEN: tl.constexpr):
    row = tl.program_id(0)
    out_page = tl.program_id(1)
    p = out_page * OUT_PAGE + tl.arange(0, OUT_PAGE)
    d = tl.arange(0, 128)
    count = tl.load(COUNT + row)
    col = tl.load(IDS + row * IS + p)
    valid = (p < count) & (col >= 0)
    # The native varlen API receives an expanded block-table row for EVERY
    # query. Its indices tensor labels adjacent query atoms for pairing; it is
    # not an index into the block table. Uniform native-N keeps one row/request.
    request = row if VARLEN else row // NEXT
    page = tl.load(TABLE + request * TS + col // PAGE, valid, other=0)
    within = tl.maximum(col, 0) % PAGE
    source = CACHE + page.to(tl.int64) * KS
    # Storage contract from indexer_k_store.py, NOT nominal tensor stride(1).
    value = tl.load(source[:, None] + within[:, None] * 128 + d[None, :], valid[:, None], other=0)
    scale = tl.load((source + PAGE * 128 + within * 4).to(tl.pointer_type(tl.float32)), valid, other=0.)
    dest = OUT + (row * PAGES + out_page).to(tl.int64) * (OUT_PAGE * 132)
    tl.store(dest + tl.arange(0, OUT_PAGE)[:, None] * 128 + d[None, :], value)
    tl.store((dest + OUT_PAGE * 128 + tl.arange(0, OUT_PAGE) * 4).to(tl.pointer_type(tl.float32)), scale)


@triton.jit
def _scatter(S, IDS, COUNT, OUT, SS: tl.constexpr, IS: tl.constexpr,
             OS: tl.constexpr, WIDTH: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    p = tl.program_id(1) * TILE + tl.arange(0, TILE)
    count = tl.load(COUNT + row)
    col = tl.load(IDS + row * IS + p, p < count, other=-1)
    value = tl.load(S + row * SS + p, p < count, other=-float("inf"))
    tl.store(OUT + row * OS + col, value, (p < count) & (col >= 0) & (col < WIDTH))


@triton.jit
def _remap(I, IDS, L, RS: tl.constexpr, IS: tl.constexpr,
           K: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    j = tl.program_id(1) * TILE + tl.arange(0, TILE)
    p = tl.load(I + row * RS + j, j < K, other=-1)
    n = tl.load(L + row)
    col = tl.load(IDS + row * IS + p, (p >= 0) & (p < n) & (j < K), other=-1)
    tl.store(I + row * RS + j, col, j < K)


def pack_candidates(cache, table, requests, lengths, candidates, *, next_n, width):
    rows, k = candidates.shape
    if not rows or not k or width < 1:
        raise ValueError("candidate consumer must have positive rows, candidates and width")
    n = triton.cdiv(width, 8)
    # K entries can retain at most K*8 positions (an edge sentinel consumes
    # one candidate). DeepGEMM SM12x FP8 requires page64; page32 is FP4-only.
    packed_page = 64
    capacity = triton.cdiv(min(width, k * 8), packed_page) * packed_page
    owners = torch.full((rows, n + 1), k, dtype=torch.int32, device=cache.device)
    flags = torch.empty((rows, n), dtype=torch.int32, device=cache.device)
    ids = torch.full((rows, capacity), -1, dtype=torch.int32, device=cache.device)
    counts = torch.empty(rows, dtype=torch.int32, device=cache.device)
    _owners[(rows, triton.cdiv(k, 128))](candidates, owners, candidates.stride(0), owners.stride(0), k, width, n, 128)
    _flags[(rows, triton.cdiv(n, 256))](owners, lengths, flags, owners.stride(0), flags.stride(0), k, n, 256)
    prefix = torch.cumsum(flags, dim=1, dtype=torch.int32)
    _positions[(rows, triton.cdiv(k * 8, 256))](candidates, owners, prefix, lengths, ids,
        candidates.stride(0), owners.stride(0), prefix.stride(0), ids.stride(0), k, width, n, 256)
    _counts[(rows,)](owners, prefix, lengths, ids, counts, owners.stride(0), prefix.stride(0), ids.stride(0), k, n, width)
    pages = capacity // packed_page
    packed = torch.empty((rows * pages, packed_page, 1, 132), dtype=torch.uint8, device=cache.device)
    packed_table = torch.arange(rows * pages, dtype=torch.int32, device=cache.device).reshape(rows, pages)
    _gather[(rows, pages)](cache, table, requests, ids, counts, packed,
        cache.stride(0), table.stride(0), ids.stride(0), cache.shape[1], packed_page, pages, next_n, requests is not None,
        num_warps=4)
    return ids, packed, packed_table, counts


def scatter_scores(scores, positions, counts, width):
    out = torch.full((scores.shape[0], width), -torch.inf, dtype=torch.float32, device=scores.device)
    _scatter[(scores.shape[0], triton.cdiv(positions.shape[1], 256))](scores, positions, counts, out,
        scores.stride(0), positions.stride(0), out.stride(0), width, 256)
    return out


def remap_topk(indices, positions, lengths):
    _remap[(indices.shape[0], triton.cdiv(indices.shape[1], 256))](indices, positions, lengths,
        indices.stride(0), positions.stride(0), indices.shape[1], 256)
