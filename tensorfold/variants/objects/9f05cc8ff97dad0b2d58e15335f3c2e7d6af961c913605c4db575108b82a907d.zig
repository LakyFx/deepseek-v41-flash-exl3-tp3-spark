//! A prompt chunk's forward on one TP rank, the served build's Model._forward_k (model.py) step by step on the served
//! build's kernels: the streams of the chunk's tokens (embed_init), then a layer at a time the attention mixes (hc_pre, or
//! hc_pre_pf with the previous layer's gathered MoE partials posted first), attention (attention_k), the gather of its
//! partials, the FFN mixes with that post fused in (hc_pre_pf), the MoE and its gather, whose post goes into the next
//! layer's mixes (switch "hc_pf2"); before an Engram layer's mixes, its Engram (engram_apply). Compressed layers
//! (ratio 1, 2): a kv source's compressor and index keys into the caches (kv_source_update), the indexer's top-k
//! (prompt_keys: scored keys and the selection; from the candidate source on: fp32 scores and topk_indices), and the
//! sparse attention over the window (a later chunk's first keys from the ring) and the selected latents. Decode-sized
//! rows (1..128) take the grouped EXL3 linears. With replay (bounded replay prefill) the decoder's first layer cuts the
//! chunk to its rows at or past replay (replayCut); DSpark taps (tap) and the head on the last row (head) end it.
//! Not yet: ring mode (verify windows), a pool that leaves blocks out, a chunk starting inside a compressor group;
//! those return error.NotPortedYet.
const std = @import("std");
pub const row_state = @import("row_state.zig");
pub var row_state_enabled = false;
pub var row_index_enabled = false;
pub var row_front_enabled = false;
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const weights = @import("weights.zig");
const tri = @import("tri.zig");
const tri_basic = @import("tri_basic.zig");
const tri_hc = @import("tri_hc.zig");
const tri_norm = @import("tri_norm.zig");
const tri_attn = @import("tri_attn.zig");
const tri_markov = @import("tri_markov.zig");
const exl3_prefill = @import("exl3_prefill.zig");
const exl3_experts = @import("exl3_experts.zig");
const exl3_experts2d = @import("exl3_experts2d.zig");
const ops = @import("ops.zig");
const cublas = @import("cublas.zig");
const comm = @import("comm.zig");
const engram = @import("engram.zig");
const engram_io = @import("engram_io.zig");
const engram_aio = @import("engram_aio.zig");
const exact = @import("exact.zig");
const tri_index = @import("tri_index.zig");
const exl3_linear = @import("exl3_linear.zig");
const prompt2d = @import("prompt2d.zig");
const rdma = @import("rdma.zig");

/// The indexer's [rows, n_comp] keys a row block holds at most (model.py: rb * n_comp <= 2^26 / 4 past 16 rows).
const key_elems: usize = 1 << 24;
/// A row block's rows at most where the layers before the candidate source prune (prunes: n_comp > 32,768, so rb =
/// 2^26 / (4 n_comp) < 512).
const pruned_rows: usize = 512;

/// model.py's RING_EXTRA: window ring slots beyond the window (a verify window never clobbers a key it reads).
pub const ring_extra = 16;
/// model.py's RAW: per-position compressor inputs kept (a ratio-2 group's earlier row for the next chunk).
pub const raw_rows = 64;

/// Device memory carved in 256-byte aligned pieces from blocks allocated as it grows, all freed together: the first
/// block init's size, then blocks of `block` bytes; a piece of `own` bytes or more takes a block of its own (the
/// pool's caches), so a pool of any size costs what its pieces take plus at most one block. `used`: the pieces'
/// bytes; `reserved`: the blocks'.
pub const Arena = struct {
    d: *const cuda.Driver,
    blocks: [max_blocks]cuda.DeviceBuffer = undefined,
    n: usize = 0,
    cur: ?usize = null, // the block small pieces come from
    at: usize = 0, // its next free byte
    used: usize = 0,
    reserved: usize = 0,

    const max_blocks = 1024;
    const block: usize = 256 << 20;
    const own: usize = 64 << 20;

    pub fn init(d: *const cuda.Driver, bytes: usize) !Arena {
        var a: Arena = .{ .d = d };
        if (bytes > 0) {
            a.cur = try a.grow(bytes);
            a.at = 0;
        }
        return a;
    }

    fn grow(a: *Arena, bytes: usize) !usize {
        if (a.n == max_blocks) return error.ArenaFull;
        a.blocks[a.n] = try cuda.DeviceBuffer.alloc(a.d, bytes);
        a.reserved += bytes;
        a.n += 1;
        return a.n - 1;
    }

    pub fn deinit(a: *Arena) void {
        for (a.blocks[0..a.n]) |*b| b.free();
        a.n = 0;
    }

    pub fn take(a: *Arena, bytes: usize) !u64 {
        if (bytes >= own) {
            const i = try a.grow(std.mem.alignForward(usize, bytes, 256));
            a.used += bytes;
            return a.blocks[i].ptr;
        }
        var at = std.mem.alignForward(usize, a.at, 256);
        if (a.cur == null or at + bytes > a.blocks[a.cur.?].len) {
            a.cur = try a.grow(block);
            at = 0;
        }
        a.at = at + bytes;
        a.used += bytes;
        return a.blocks[a.cur.?].ptr + at;
    }
};

/// A RoPE kind's tables on the device: cos, sin fp32 [rows, rope_dim / 2] (ops.RopeTables' rows).
pub const Rope = struct { cos: u64, sin: u64 };

/// What a chunk's steps run on: one rank's weights, kernels, stream and gather.
pub const Engine = struct {
    d: *const cuda.Driver,
    s: cuda.Stream,
    t: tri.Tri,
    blas: *const cublas.Blas,
    comm: *const comm.Comm,
    pf: *const exl3_prefill.Kernels,
    lin: *const exl3_linear.Kernels,
    ex: *const exl3_experts.Kernels,
    ops: *const ops.Ops,
    exact: *const exact.Exact,
    c: *const Config,
    w: *const weights.Weights,
    world: usize,
    plain: Rope,
    compressed: Rope,
    round_rows: usize = 16, // a concurrent round's rows at most (TF_DS_ROUND_ROWS; round.max_rows at most)
    two: ?*const prompt2d.Two = null, // a node of the four-node 2D split (prompt2d.zig); `world` stays TP2's
    ring: ?*rdma.Ring = null, // TP2: the decode-size all-gathers over the RDMA ring (the served lane's), else NCCL

    /// A decode-size all-gather of `count` fp32 a rank into `dst`, rank after rank: the RDMA ring when the engine has
    /// one and the slice fits a slot (whole float4s), else NCCL. The same bytes either way.
    pub fn gatherF32(e: *const Engine, src: u64, dst: u64, count: usize) !void {
        if (e.ring) |r| if (count % 4 == 0 and count * 4 <= r.s.max_bytes) return r.gather(e.s, src, @intCast(count / 4), dst);
        return e.comm.allGather(src, dst, count, .f32, e.s);
    }

    pub fn heads(e: *const Engine) usize {
        if (e.two) |t| return t.heads;
        return e.c.heads / e.world;
    }

    pub fn slots(e: *const Engine) usize {
        return e.c.top_k + 1;
    }

    /// A sequence's window ring for one layer: bf16 [window + RING_EXTRA, head_dim].
    pub fn ringBytes(e: *const Engine) usize {
        return (e.c.window + ring_extra) * e.c.head_dim * 2;
    }
};

/// A prompt chunk's buffers for up to `cap` rows (model.py _forward_k's and its steps' torch.empty's).
/// An image span of the prompt: positions [at, at + len), its rows from row `row` of the image rows buffer.
pub const Span = struct { at: usize, len: usize, row: usize };

/// The rows [r0, r1) of the chunk (relative to its start) a span covers, if any.
fn overlap(sp: Span, start: usize, n: usize) ?[2]usize {
    const lo = @max(sp.at, start);
    const hi = @min(sp.at + sp.len, start + n);
    return if (lo < hi) .{ lo - start, hi - start } else null;
}

pub const Chunk = struct {
    row_active: bool = false,
    row_stats: row_state.Stats = .{},
    cap: usize,
    spans: []const Span = &.{}, // the prompt's image spans (absolute positions)
    img_rows: u64 = 0, // bf16 [span tokens, D]: their rows
    pick_vl: u64 = 0, // int32 [cap, slots]: the routing of span rows with the image bias, before it joins pick
    wts_vl: u64 = 0, // fp32 [cap, slots]
    n: usize = 0,
    start: usize = 0,
    ids: u64, // int64 [cap]
    pos: u64, // int64 [n]: start .. start + n (after a replay cut a view into pos0, pos[first:])
    pos0: u64, // int64 [cap]: the chunk's positions as begin() wrote them
    h: u64, // bf16 [cap, hc, D]: the streams
    h_alt: u64, // their second buffer (hc_pre_pf writes the posted streams there)
    x: u64, // bf16 [cap, D]: a block's input rows
    part: u64, // fp32 [cap * HC_BLOCKS * 32]
    pre: u64, // fp32 [cap, hc]: the layer's input pre (the previous layer's FFN pre_out)
    pre_a: u64, // fp32 [cap, hc]
    pre_f: u64, // fp32 [cap, hc]
    post: u64, // fp32 [cap, hc]
    comb: u64, // fp32 [cap, hc, hc]
    // attention
    qa: u64, // bf16 [cap, q_lora]
    y: u64, // bf16 [cap, head_dim]
    qr: u64, // bf16 [cap, q_lora]
    q: u64, // bf16 [cap, heads, head_dim]
    wsrc: u64, // bf16 [window - 1 + cap, head_dim]
    wlo: u64, // int64 [1]
    o: u64, // bf16 [cap, heads, head_dim]
    u: u64, // bf16 [cap, groups * o_lora]
    pa: u64, // fp32 [cap, D]: the attention partial
    ga: u64, // fp32 [world, cap, D]: its gather
    neg: u64, // int64 [cap], -1: no ring slot (model.py _neg)
    ring_idx: u64, // int64 [ring]
    rslot: u64 = 0, // int64 [RING_EXTRA]: ring mode's slots, pos % R
    pm_: u64 = 0, // fp32 [16, heads, 8]: the split attention's partials (ring mode's rows)
    pl_: u64 = 0,
    po_: u64 = 0, // fp32 [16, heads, 8, head_dim]
    invalid: u64, // uint32 [1]
    // compressed layers
    kvc: u64 = 0, // fp32 [cap, head_dim]: the compressor's kv projection
    scc: u64 = 0, // fp32 [cap, head_dim]: its gate scores
    lat: u64 = 0, // bf16 [cap, head_dim]: the compressed latents
    lat2: u64 = 0, // bf16 [cap, head_dim]: their RoPE'd copy (the cache's)
    ik: u64 = 0, // bf16 [cap, index_head_dim]: the index keys
    groups: u64 = 0, // int64 [cap]: the new latents' groups
    gpos: u64 = 0, // int64 [cap]: their positions (group * ratio)
    raw_idx: u64 = 0, // int64 [RAW]
    iq: u64 = 0, // bf16 [cap, index_heads * index_head_dim]
    iq4: u64 = 0, // bf16 [cap, index_heads * index_head_dim]: fp4_qd's bytes
    wl: u64 = 0, // fp32 [cap, index_heads]
    iw: u64 = 0, // bf16 [cap, index_heads]: the heads' weights
    vis: u64 = 0, // int64 [cap]: compressed entries a row sees
    vis_blk: u64 = 0, // int64 [cap]: an indexer row block's vis when its view of vis is not 16-byte aligned
    keys: u64 = 0, // int64 [rows of a block, n_comp]: at most key_elems
    score: u64 = 0, // fp32 [rows of a block, n_comp]: the candidate pool's layers' index scores
    tmax: u64 = 0, // int64 [rows of a block, cdiv(n_comp, 64)]
    cidx: u64 = 0, // int64 [cap, index_topk]: the selected latents
    max_comp: usize = 0,
    // the candidate pool past candidate_blocks blocks (model.py _candidates, apply_candidates): the chunk's block mask
    // (u8 [cap, pool_ms]), a row block's block maxima (fp32) and their top blocks (int64 [rows, candidate_blocks]), a
    // row's every_vis (no mask); the layers before the candidate source's pruned selection (tpos, cand: up to
    // pruned_rows rows)
    pool: u64 = 0,
    pool_ms: usize = 0,
    bmax: u64 = 0,
    pidx: u64 = 0,
    every: u64 = 0,
    pbufs: tri_index.PrunedBufs = .{ .tpos = 0, .cand = 0, .every = 0 },
    // MoE
    xf: u64, // fp32 [cap, D]
    gate_f: u64, // fp32 [experts, D]
    logits: u64, // fp32 [cap, experts]
    pick: u64, // int32 [cap, slots]
    wts: u64, // fp32 [cap, slots]
    pm: u64, // fp32 [cap, D]: the MoE partial
    gm: u64, // fp32 [world, cap, D]: its gather (the pending post)
    xs: exl3_experts.Scratch,
    // Model._moe_scratch's decode scratch: windows under 64 rows (a prompt's small chunk, every decode round), one for
    // every layer of the same slots and experts, its epoch carried from call to call
    xsd: exl3_experts.DecodeScratch,
    gl: u64, // fp32 [16, D / 256, experts]: rowmm_gate's chunk sums (chunks of up to 16 rows)
    cidxp: u64 = 0, // int64 [cap, index_topk + 16]: the selection padded with -1 to whole 16-column tiles
    // Engram (zero when no layer has it)
    eb: u64 = 0, // bf16 [cap, cols * engram_head_dim]: the rows read
    ek: u64 = 0, // fp32 [cap, hc * D + D]: their projection, this rank's columns
    ekg: u64 = 0, // fp32 [world, cap, hc * D + D]: its gather
    kv: u64 = 0, // bf16 [cap, hc * D + D]: the ranks' sum
    ws: exl3_prefill.Workspace,
    blas_ws: u64, // cuBLAS workspace
    // the grouped EXL3 linears of decode-sized rows (1..128): rotated rows, split-K partials, each layer's counters
    gxh: u64 = 0, // fp16 [128, the largest group's sum of K]
    gz: u64 = 0, // fp32 [the largest group's split-K partials at 128 rows]
    counters: [640]Counter = undefined,
    n_counters: usize = 0,
    ktop: u64 = 0, // int64 [cap, index_topk]: torch's top-k values for _topk_finish
    // the DSpark taps and the head
    taps: u64 = 0, // bf16 [taps, cap, D]: each tap layer's mean of the streams (Model.forward's taps)
    head_x: u64 = 0, // bf16 [1, D]: collapse_norm of the last row
    head_l: u64 = 0, // fp32 [1, this rank's vocabulary columns]: its logits
    head_g: u64 = 0, // fp32 [world, 1, columns]: their gather (the prompt's logits, rank after rank)
    pending: bool = false, // gm holds a MoE gather whose post is not in h yet

    /// Every buffer for `cap` rows from `a`, the compressed layers' for chunks ending by position `tokens`; the scratch
    /// the served build zeros (or fills with -1) is set the same.
    pub fn init(e: *const Engine, a: *Arena, cap: usize, tokens: usize) !Chunk {
        const c = e.c;
        const d = c.hidden;
        const hc = c.hc;
        const l0 = e.w.layers[0];
        const hl = e.heads();
        const hd = c.head_dim;
        const groups = l0.groups;
        const o_lora = l0.wo_a[0].n;
        const ex = l0.experts;
        const sl = e.slots();
        var ch: Chunk = undefined;
        ch.row_active = false;
        ch.row_stats = .{};
        ch.spans = &.{};
        ch.img_rows = 0;
        ch.cap = cap;
        ch.n = 0;
        ch.start = 0;
        ch.pending = false;
        ch.ids = try a.take(cap * 8);
        ch.pos0 = try a.take(cap * 8);
        ch.pos = ch.pos0;
        ch.h = try a.take(cap * hc * d * 2);
        ch.h_alt = try a.take(cap * hc * d * 2);
        ch.x = try a.take(cap * d * 2);
        ch.part = try a.take(cap * tri_basic.hc_blocks * 32 * 4);
        ch.pre = try a.take(cap * hc * 4);
        ch.pre_a = try a.take(cap * hc * 4);
        ch.pre_f = try a.take(cap * hc * 4);
        ch.post = try a.take(cap * hc * 4);
        ch.comb = try a.take(cap * hc * hc * 4);
        ch.qa = try a.take(cap * l0.wq_a.n * 2);
        ch.y = try a.take(cap * hd * 2);
        ch.qr = try a.take(cap * l0.wq_a.n * 2);
        ch.q = try a.take(cap * hl * hd * 2);
        ch.wsrc = try a.take((c.window - 1 + cap) * hd * 2);
        ch.wlo = try a.take(8);
        ch.o = try a.take(cap * hl * hd * 2);
        ch.u = try a.take(cap * groups * o_lora * 2);
        ch.pa = try a.take(cap * d * 4);
        ch.ga = try a.take(e.world * cap * d * 4);
        ch.neg = try a.take(cap * 8);
        ch.ring_idx = try a.take((c.window + ring_extra) * 8);
        ch.rslot = try a.take(ring_extra * 8);
        ch.pm_ = try a.take(ring_extra * hl * tri_attn.attn_splits * 4);
        ch.pl_ = try a.take(ring_extra * hl * tri_attn.attn_splits * 4);
        ch.po_ = try a.take(ring_extra * hl * tri_attn.attn_splits * hd * 4);
        ch.invalid = try a.take(4);
        ch.xf = try a.take(cap * d * 4);
        ch.gate_f = try a.take(c.experts * d * 4);
        ch.logits = try a.take(cap * c.experts * 4);
        ch.pick = try a.take(cap * sl * 4);
        ch.wts = try a.take(cap * sl * 4);
        ch.pick_vl = try a.take(cap * sl * 4);
        ch.wts_vl = try a.take(cap * sl * 4);
        ch.pm = try a.take(cap * d * 4);
        ch.gm = try a.take(e.world * cap * d * 4);
        // compressed layers: the smallest ratio sets the most compressed entries a row can see
        var min_ratio: usize = 0;
        for (e.w.layers) |lay| {
            if (lay.ratio != 0 and (min_ratio == 0 or lay.ratio < min_ratio)) min_ratio = lay.ratio;
        }
        if (min_ratio != 0) {
            const ih = c.index_heads * c.index_head_dim;
            ch.max_comp = tokens / min_ratio;
            ch.kvc = try a.take(cap * c.head_dim * 4);
            ch.scc = try a.take(cap * c.head_dim * 4);
            ch.lat = try a.take(cap * c.head_dim * 2);
            ch.lat2 = try a.take(cap * c.head_dim * 2);
            ch.ik = try a.take(cap * c.index_head_dim * 2);
            ch.groups = try a.take(cap * 8);
            ch.gpos = try a.take(cap * 8);
            ch.raw_idx = try a.take(raw_rows * 8);
            ch.iq = try a.take(cap * ih * 2);
            ch.iq4 = try a.take(cap * ih * 2);
            ch.wl = try a.take(cap * c.index_heads * 4);
            ch.iw = try a.take(cap * c.index_heads * 2);
            ch.vis = try a.take(cap * 8);
            ch.vis_blk = try a.take(cap * 8);
            // the indexer's row blocks keep rows * n_comp under 2^24 (model.py rb), so their keys under key_elems
            const ke: usize = @min(cap * ch.max_comp, key_elems); // (typed: @min with a comptime bound narrows)
            ch.keys = try a.take(ke * 8);
            ch.score = try a.take(ke * 4);
            ch.tmax = try a.take((ke / tri_index.tile + cap) * 8);
            ch.cidx = try a.take(cap * c.index_topk * 8);
            ch.ktop = try a.take(cap * c.index_topk * 8);
            ch.cidxp = try a.take(cap * (c.index_topk + 16) * 8);
            // the pool: only past candidate_blocks blocks, so a row block holds at most key_elems / 2048 rows there
            const cb = c.candidate_block;
            ch.pool_ms = (ch.max_comp + cb - 1) / cb;
            if (ch.pool_ms > c.candidate_blocks) {
                ch.pool = try a.take(cap * ch.pool_ms);
                ch.bmax = try a.take(@max(16 * ch.pool_ms, key_elems / cb + cap) * 4);
                ch.pidx = try a.take(cap * c.candidate_blocks * 8);
            }
            ch.every = try a.take(cap * 8);
            {
                const ev = try std.heap.page_allocator.alloc(i64, cap);
                defer std.heap.page_allocator.free(ev);
                @memset(ev, tri_index.every_vis);
                try upload(e, ch.every, ev.ptr, cap * 8);
                try e.s.synchronize(); // (the host copy is freed next)
            }
            if (tri_index.prunes(ch.max_comp, c.index_topk)) {
                ch.pbufs = .{ .tpos = try a.take(pruned_rows * c.index_topk * 8), .cand = try a.take(pruned_rows * c.index_topk * tri_index.tile * 8), .every = ch.every };
            }
        }
        for (e.w.layers) |lay| {
            const ew = lay.engram_wkv orelse continue;
            const en = if (e.two) |t| t.engramWidth() else ew.n; // the summed projection's width (2D: both pairs')
            ch.eb = try a.take(cap * ew.k * 2);
            ch.ek = try a.take(cap * ew.n * 4);
            ch.ekg = try a.take(e.world * cap * en * 4);
            ch.kv = try a.take(cap * en * 2);
            break;
        }
        const sz = exl3_experts.Scratch.sizes(cap, sl, ex.dims, ex.width, ex.count);
        var xs: exl3_experts.Scratch = undefined;
        xs.rows = cap;
        xs.slots = sl;
        inline for (.{ "xg", "xu", "xd", "z", "no_y", "ids", "count", "counts", "members", "work_gu", "work_d" }, 0..) |f, j| {
            @field(xs, f) = try a.take(sz[j]);
        }
        ch.xs = xs;
        ch.xsd = try decodeScratch(e, a, ex, sl);
        ch.gl = try a.take(16 * (d / 256) * c.experts * 4);
        // prefill.Workspace: the largest prompt GEMM's rotated input and W_q (fp16) over every layer, and the Hadamard
        var max_xk: usize = 0;
        var max_kn: usize = 0;
        for (e.w.layers) |lay| {
            var ls: [16]?weights.Linear = @splat(null);
            ls[0] = lay.wq_a;
            ls[1] = lay.wkv;
            ls[2] = lay.wq_b;
            ls[3] = lay.wo_b;
            for (lay.wo_a[0..lay.groups], 0..) |wo, g| ls[4 + g] = wo;
            ls[8] = lay.comp_wkv;
            ls[9] = lay.comp_wgate;
            ls[10] = lay.idx_wq_b;
            ls[11] = lay.idx_wk;
            ls[12] = lay.engram_wkv;
            for (ls) |ol| {
                const l = ol orelse continue;
                max_xk = @max(max_xk, l.k);
                max_kn = @max(max_kn, @as(usize, l.k) * l.n);
            }
        }
        ch.ws = .{ .xh = try a.take(cap * max_xk * 2), .w = try a.take(max_kn * 2), .h = try a.take(128 * 128 * 2) };
        ch.blas_ws = try a.take(cublas.Blas.workspace_bytes);
        // grouped linears: every layer's counters (int32 [8 N / 128], zeros, left zero by each launch), the largest
        // group's rotated rows and partials at 128 rows
        var max_gk: usize = 0;
        var max_gz: usize = 0;
        ch.n_counters = 0;
        for (e.w.layers) |lay| try layerLinears(e, &ch, a, lay, &max_gk, &max_gz);
        {
            // the head: one row through its own group
            const l = e.w.head;
            try addLinear(e, &ch, a, l, &max_gk, &max_gz);
            ch.head_x = try a.take(d * 2);
            ch.head_l = try a.take(l.n * 4);
            const hn = if (e.two) |t| t.hw[0] + t.hw[1] else l.n; // a rank's vocabulary half (2D: both pairs' parts)
            ch.head_g = try a.take(e.world * hn * 4);
        }
        if (e.w.dspark) |ds| {
            // the drafter (draft.zig): its stages' linears as a layer's, main_proj, and the stages' wkv as one group
            // (Drafter.stage_kv)
            var kv_k: usize = 0;
            var kv: [8]weights.Linear = undefined;
            if (ds.blocks.len > kv.len) return error.TooManyStages;
            for (ds.blocks, 0..) |lay, j| {
                try layerLinears(e, &ch, a, lay, &max_gk, &max_gz);
                kv[j] = lay.wkv;
                kv_k += lay.wkv.k;
            }
            max_gk = @max(max_gk, kv_k);
            max_gz = @max(max_gz, exl3_linear.zFloats(kv[0..ds.blocks.len], 128));
            try addLinear(e, &ch, a, ds.main_proj, &max_gk, &max_gz);
        }
        ch.taps = try a.take(@max(c.dspark_taps.slice().len, 1) * cap * d * 2);
        ch.gxh = try a.take(128 * max_gk * 2);
        ch.gz = try a.take(@max(max_gz, 1) * 4);
        // the scratch as experts.py makes it: zeros, the member lists -1
        for ([_]u64{ xs.xg, xs.xu, xs.xd, xs.z, xs.no_y, xs.ids, xs.count, xs.counts }, [_]usize{ sz[0], sz[1], sz[2], sz[3], sz[4], sz[5], sz[6], sz[7] }) |p, n| try fill(e, p, 0, n);
        try fill32(e, xs.members, 0xffffffff, sz[8] / 4);
        try fill(e, ch.neg, 0xff, cap * 8);
        var had: [128 * 128]u16 = undefined;
        exl3_prefill.hadamard(&had);
        try upload(e, ch.ws.h, &had, had.len * 2);
        return ch;
    }
};

/// A layer's linears in the chunk's grouped-linear sizes (its attention input group, its wo_a group, each alone at 128
/// rows) and each one's split-K counters (Chunk.init).
fn layerLinears(e: *const Engine, ch: *Chunk, a: *Arena, lay: weights.Layer, max_gk: *usize, max_gz: *usize) !void {
    const in_group = [_]weights.Linear{ lay.wq_a, lay.wkv };
    max_gk.* = @max(max_gk.*, lay.wq_a.k + lay.wkv.k);
    max_gz.* = @max(max_gz.*, exl3_linear.zFloats(&in_group, 128));
    var wo_k: usize = 0;
    for (lay.wo_a[0..lay.groups]) |wo| wo_k += wo.k;
    max_gk.* = @max(max_gk.*, wo_k);
    max_gz.* = @max(max_gz.*, exl3_linear.zFloats(lay.wo_a[0..lay.groups], 128));
    var ls: [16]?weights.Linear = @splat(null);
    ls[0] = lay.wq_a;
    ls[1] = lay.wkv;
    ls[2] = lay.wq_b;
    ls[3] = lay.wo_b;
    for (lay.wo_a[0..lay.groups], 0..) |wo, g| ls[4 + g] = wo;
    ls[8] = lay.comp_wkv;
    ls[9] = lay.comp_wgate;
    ls[10] = lay.idx_wq_b;
    ls[11] = lay.idx_wk;
    ls[12] = lay.engram_wkv;
    for (ls) |ol| {
        const l = ol orelse continue;
        try addLinear(e, ch, a, l, max_gk, max_gz);
    }
}

/// One linear alone at 128 rows in the chunk's grouped-linear sizes, and its split-K counters (int32 [8 N / 128], zeros,
/// left zero by each launch).
fn addLinear(e: *const Engine, ch: *Chunk, a: *Arena, l: weights.Linear, max_gk: *usize, max_gz: *usize) !void {
    max_gk.* = @max(max_gk.*, l.k);
    max_gz.* = @max(max_gz.*, exl3_linear.zFloats(&[_]weights.Linear{l}, 128));
    if (ch.n_counters == ch.counters.len) return error.TooManyLinears;
    const bytes = 8 * (l.n / 128) * 4;
    const ptr = try a.take(bytes);
    try fill(e, ptr, 0, bytes);
    ch.counters[ch.n_counters] = .{ .words = l.words, .ptr = ptr };
    ch.n_counters += 1;
}

/// Model._moe_scratch's decode scratch: experts.py Scratch(rows=max(n, 64), prompt=False), zeros but its member lists'
/// -1 (made at the first window under 64 rows and never replaced).
pub fn decodeScratch(e: *const Engine, a: *Arena, ex: weights.Experts, slots: usize) !exl3_experts.DecodeScratch {
    const rows = exl3_experts.exact_rows;
    const sz = if (e.two != null) try exl3_experts2d.decodeSizes(rows, slots, ex) else try exl3_experts.DecodeScratch.sizes(rows, slots, ex.dims, ex.width, ex.count); // 2D: this node's widths
    var sc: exl3_experts.DecodeScratch = undefined;
    sc.rows = rows;
    sc.slots = slots;
    inline for (.{ "xg", "xu", "xd", "z", "y", "cnt_gu", "cnt_d", "epoch", "ready", "ready_cnt", "ids", "count", "members" }, 0..) |f, j| {
        const ptr = try a.take(sz[j]);
        @field(sc, f) = ptr;
        if (j == 12) try fill32(e, ptr, 0xffffffff, sz[j] / 4) else try fill(e, ptr, 0, sz[j]);
    }
    return sc;
}

/// Host bytes to the device in the stream's order (the stream is non-blocking: the legacy stream's synchronous copies
/// would not wait for its kernels). Pageable host memory is staged before the call returns.
pub fn upload(e: *const Engine, dst: u64, src: *const anyopaque, bytes: usize) !void {
    try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(dst, src, bytes, e.s.handle), "cuMemcpyHtoDAsync");
}

/// cuMemsetD8 in the stream's order.
pub fn fill(e: *const Engine, dst: u64, value: u8, bytes: usize) !void {
    try e.d.check(e.d.api.cuMemsetD8Async(dst, value, bytes, e.s.handle), "cuMemsetD8Async");
}

/// cuMemsetD32 in the stream's order.
fn fill32(e: *const Engine, dst: u64, value: u32, words: usize) !void {
    try e.d.check(e.d.api.cuMemsetD32Async(dst, value, words, e.s.handle), "cuMemsetD32Async");
}

/// A layer's split-K counters (Exl3Linear.counters) by its trellis words.
pub const Counter = struct { words: u64, ptr: u64 };

pub fn counterOf(ch: *const Chunk, l: weights.Linear) !u64 {
    for (ch.counters[0..ch.n_counters]) |c| if (c.words == l.words) return c.ptr;
    return error.NoCounters;
}

/// Exl3Group(ls) on m (1..128) rows (linear.py): one rot_many launch rotating every layer's input rows xs[i] (bf16,
/// row stride ldxs[i]) into the group's buffer, then a glinear launch for each (bits, warps) set, outputs into outs[i]
/// (row stride ldos[i], dtype types[i]); programmatic dependent launches, split-K partials dropped from L2 (the served
/// TF_EXL3_PDL and TF_EXL3_L2_DISCARD).
pub fn grouped(e: *const Engine, ch: *const Chunk, m: usize, ls: []const weights.Linear, xs: []const u64, ldxs: []const usize, outs: []const u64, ldos: []const usize, types: []const exl3_linear.DType) !void {
    if (m == 0 or m > 128 or ls.len > 8) return error.NotADecodeWindow;
    var calls: [8]exl3_linear.Call = undefined;
    var off: usize = 0;
    for (ls, 0..) |l, i| {
        calls[i] = .{ .layer = l, .x = xs[i], .ldx = @intCast(ldxs[i]), .x_dtype = .bf16, .xh = ch.gxh + off * 2, .y = outs[i], .ldy = @intCast(ldos[i]), .y_dtype = types[i], .counters = try counterOf(ch, l) };
        off += m * l.k;
    }
    try exl3_linear.rotMany(e.lin, e.s, calls[0..ls.len], m, true);
    try exl3_linear.glinear(e.lin, e.s, calls[0..ls.len], m, ch.gz, true, true);
}

/// Exl3Group.rotated on m rows already rotated into each call's xh (a kernel that folds the rotation in wrote them):
/// the glinear launches alone, each call's counters and folds as given.
pub fn groupedRotated(e: *const Engine, ch: *const Chunk, m: usize, calls: []exl3_linear.Call) !void {
    if (m == 0 or m > 128 or calls.len > exl3_linear.gmax) return error.NotADecodeWindow;
    for (calls) |*c| c.counters = try counterOf(ch, c.layer);
    try exl3_linear.glinear(e.lin, e.s, calls, m, ch.gz, true, true);
}

/// The pool's compressed-attention caches (model.py PoolCache: comp, index_k, comp_raw), by kv-source layer: the
/// compressed latents as packed FP4 (codes [rows, head_dim / 2], E8M0 scales [rows, head_dim / 16]), the index keys
/// (codes [rows, index_head_dim / 2], scales [rows, index_head_dim / 32], 127 when unwritten) over `cap` positions of
/// streams' extents (rows = cap / ratio + 2), and, at ratio 2, each slot's positional store of the compressor's raw kv
/// and scores (fp32 [RAW, head_dim] each, `slots` of them). One stream's caches (SeqCache) are a view.
pub const Caches = struct {
    comp_codes: [64]u64 = @splat(0),
    comp_scales: [64]u64 = @splat(0),
    idx_codes: [64]u64 = @splat(0),
    idx_scales: [64]u64 = @splat(0),
    raw_kv: [64]u64 = @splat(0),
    raw_score: [64]u64 = @splat(0),
    cap: usize = 0, // positions
    slots: usize = 1, // the positional stores' slots

    pub fn init(e: *const Engine, a: *Arena, cap: usize, slots: usize) !Caches {
        const c = e.c;
        var cs: Caches = .{ .cap = cap, .slots = slots };
        for (e.w.layers, 0..) |lay, i| {
            if (lay.comp_wkv == null) continue;
            const r: usize = lay.ratio;
            const rows = cap / r + 2;
            cs.comp_codes[i] = try a.take(rows * c.head_dim / 2);
            cs.comp_scales[i] = try a.take(rows * c.head_dim / 16);
            if (lay.idx_wk != null) {
                cs.idx_codes[i] = try a.take(rows * c.index_head_dim / 2);
                cs.idx_scales[i] = try a.take(rows * c.index_head_dim / 32);
            }
            if (r > 1) {
                cs.raw_kv[i] = try a.take(slots * raw_rows * c.head_dim * 4);
                cs.raw_score[i] = try a.take(slots * raw_rows * c.head_dim * 4);
            }
        }
        try cs.clear(e);
        return cs;
    }

    /// A new pool's caches, as PoolCache makes them: zeros, the index keys' E8M0 scales 127 (1.0).
    pub fn clear(cs: *const Caches, e: *const Engine) !void {
        const c = e.c;
        for (e.w.layers, 0..) |lay, i| {
            if (lay.comp_wkv == null) continue;
            const r: usize = lay.ratio;
            const rows = cs.cap / r + 2;
            try fill(e, cs.comp_codes[i], 0, rows * c.head_dim / 2);
            try fill(e, cs.comp_scales[i], 0, rows * c.head_dim / 16);
            if (lay.idx_wk != null) {
                try fill(e, cs.idx_codes[i], 0, rows * c.index_head_dim / 2);
                try fill(e, cs.idx_scales[i], 127, rows * c.index_head_dim / 32);
            }
            if (r > 1) {
                try fill(e, cs.raw_kv[i], 0, cs.slots * raw_rows * c.head_dim * 4);
                try fill(e, cs.raw_score[i], 0, cs.slots * raw_rows * c.head_dim * 4);
            }
        }
    }

    /// The stream in pool slot `slot` with its extent from position `base` (MultiDecoder: pool_view): its compressed
    /// rows from base / ratio, its slot's positional stores (one slot's view: its prompt chunks read positions from 0).
    pub fn view(cs: *const Caches, e: *const Engine, slot: usize, base: usize) !Caches {
        const c = e.c;
        if (slot >= cs.slots or base >= cs.cap) return error.BadView;
        var v = cs.*;
        v.slots = 1;
        v.cap = cs.cap - base;
        for (e.w.layers, 0..) |lay, i| {
            if (lay.comp_wkv == null) continue;
            const r: usize = lay.ratio;
            if (base % r != 0) return error.BadView;
            const g = base / r;
            v.comp_codes[i] += g * c.head_dim / 2;
            v.comp_scales[i] += g * c.head_dim / 16;
            if (lay.idx_wk != null) {
                v.idx_codes[i] += g * c.index_head_dim / 2;
                v.idx_scales[i] += g * c.index_head_dim / 32;
            }
            if (r > 1) {
                v.raw_kv[i] += slot * raw_rows * c.head_dim * 4;
                v.raw_score[i] += slot * raw_rows * c.head_dim * 4;
            }
        }
        return v;
    }
};

/// What one forward carries from layer to layer (model.py _forward_k's `shared`): the latest kv-source layer, and the
/// latest indexer's selection (in the chunk's cidx, kk a row).
pub const Shared = struct { kv_layer: ?usize = null, kk: ?usize = null, pool: bool = false };

/// The chunk's rows: token ids at positions start .., their streams h [n, hc, D] (every stream the token's embedding
/// row) and pre [n, hc] = (1, 0, 0, 0): embed_init, the bytes of forward()'s embedding rows expanded over the streams.
pub fn begin(e: *const Engine, ch: *Chunk, ids: []const i64, start: usize, host_pos: []i64) !void {
    const n = ids.len;
    if (n > ch.cap or host_pos.len < n) return error.ChunkTooLong;
    ch.n = n;
    ch.start = start;
    ch.pending = false;
    for (host_pos[0..n], 0..) |*p, i| p.* = @intCast(start + i);
    try upload(e, ch.ids, ids.ptr, n * 8);
    ch.pos = ch.pos0;
    try upload(e, ch.pos, host_pos.ptr, n * 8);
    if (ch.row_active) {
        const r = try row_state.rows(n, e.comm.rank, e.world);
        try tri_basic.embedInit(e.t, e.w.embed, ch.ids + r.first * 8,
            ch.h + r.first * e.c.hc * e.c.hidden * 2,
            ch.pre + r.first * e.c.hc * 4, r.count, e.c.hidden, e.c.hc);
    } else try tri_basic.embedInit(e.t, e.w.embed, ch.ids, ch.h, ch.pre, n, e.c.hidden, e.c.hc);
}

/// merge_image_embeddings on the chunk: each span row's embedding (in every one of the hc streams) replaced by its
/// image row (the delimiters' learned rows and the aligner's), before the first layer.
pub fn embedSpans(e: *const Engine, ch: *Chunk) !void {
    const c = e.c;
    const row = c.hidden * 2;
    for (ch.spans) |sp| {
        const o = overlap(sp, ch.start, ch.n) orelse continue;
        const src = ch.img_rows + (sp.row + (ch.start + o[0] - sp.at)) * row;
        for (0..c.hc) |s| try e.ops.copyRows(e.s, src, row, ch.h + (o[0] * c.hc + s) * row, c.hc * row, row, o[1] - o[0]);
    }
}

/// The streams' second buffer becomes the streams (hc_pre_pf wrote the posted ones there) and the old one the spare.
fn swapStreams(ch: *Chunk, src: u64) void {
    std.debug.assert(src == ch.h_alt);
    ch.h_alt = ch.h;
    ch.h = src;
}

/// Layer `li`'s attention mixes: x = RMSNorm of the streams mixed by pre, pre_a / post / comb from their own mixes;
/// with a pending MoE gather, hc_pre_pf posts it into the streams first.
pub fn attnMixes(e: *const Engine, ch: *Chunk, li: usize) !void {
    if (ch.row_active) return rowAttnMixes(e, ch, li);
    return attnMixesRows(e, ch, li);
}

fn attnMixesRows(e: *const Engine, ch: *Chunk, li: usize) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    if (ch.pending) {
        const src = try tri_hc.hcPrePf(e.t, ch.h, lay.hc_attn[0], lay.hc_attn[1], lay.hc_attn[2], ch.pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, ch.x, ch.pre_a, ch.post, ch.comb, ch.part, .{ .gathered = ch.gm, .world = e.world, .h_out = ch.h_alt }, ch.n, c.hidden);
        swapStreams(ch, src);
        ch.pending = false;
    } else {
        try tri_basic.hcPre(e.t, ch.h, lay.hc_attn[0], lay.hc_attn[1], lay.hc_attn[2], ch.pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, ch.x, ch.pre_a, ch.post, ch.comb, ch.part, ch.n, c.hidden);
    }
}

/// mm(layer, x, out_dtype) of m rows: the prompt GEMM beyond 128 rows, the layer's own group (GROUPED) up to 128.
fn mmRows(e: *const Engine, ch: *const Chunk, m: usize, l: weights.Linear, x: u64, ldx: usize, out: u64, out_type: tri_markov.OutType, os: usize) !void {
    if (m > 128) return exl3_prefill.matmul(e.pf, e.t, ch.ws, l, x, .bf16, ldx, out, out_type, os, m);
    const dt: exl3_linear.DType = switch (out_type) {
        .bf16 => .bf16,
        .fp32 => .f32,
    };
    try grouped(e, ch, m, &.{l}, &.{x}, &.{ldx}, &.{out}, &.{os}, &.{dt});
}

/// mm of the chunk's rows.
pub fn mm(e: *const Engine, ch: *const Chunk, l: weights.Linear, x: u64, ldx: usize, out: u64, out_type: tri_markov.OutType, os: usize) !void {
    try mmRows(e, ch, ch.n, l, x, ldx, out, out_type, os);
}

/// TF_DS_REPLAY_FLOOR=1 (model.zig sets it on every rank): a small chunk whose window reaches below the floor reads it
/// by the gather path (off: ring mode, as the served lane reads, the slot's previous request's keys below the floor).
pub var replay_floor = false;
/// Short decoder continuation can reach compressed prefill shapes with <=16
/// rows. Their non-based ring split variant is absent from the TP3 AOT kit.
/// Use the already captured gathered, one-split prompt path for this opt-in mode.
pub var short_continuation_gather = false;

fn promptRingMode(n: usize, below: bool, clip_floor: bool, gather_small: bool) bool {
    return n <= ring_extra and !gather_small and !(clip_floor and below);
}

/// attention_k: the partial pa [n, D] fp32 of this rank's heads; the chunk's keys go into the layer's window ring
/// (bf16 [window + RING_EXTRA, head_dim]). A compressed layer also attends to its kv source's latents the indexer picks
/// (a kv source first compresses the chunk into them). `floor`: no window key before this position.
pub fn attention(e: *const Engine, ch: *Chunk, cs: *const Caches, sh: *Shared, li: usize, ring: u64, floor: usize, kv_done: bool) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    const n = ch.n;
    const start = ch.start;
    const hd = c.head_dim;
    const rd = c.rope_dim;
    const hl = e.heads();
    // _cs: the layer's RoPE kind (compressed layers rotate by the compressed table) for every rotation it makes
    const rope = if (lay.ratio != 0) e.compressed else e.plain;
    const ring_rows = c.window + ring_extra;
    // ring mode (a chunk of RING_EXTRA rows or fewer): the rows' keys into the ring first, the attention reads the ring.
    // With replay_floor, not when the window reaches below `floor` (the replay's cut: below it the ring holds the slot's
    // previous request's keys): such a chunk reads by the gather path from the floor, as one large chunk's rows would
    const below = floor > start -| (c.window - 1);
    const ring_mode = promptRingMode(n, below, replay_floor, short_continuation_gather);
    const one_split = !ring_mode and n <= tri.decode_rows;
    const owned_front = ch.row_active and row_front_enabled;
    if (owned_front) {
        try rowAttentionFront(e, ch, lay);
    } else if (n > 128) {
        try mm(e, ch, lay.wq_a, ch.x, c.hidden, ch.qa, .bf16, lay.wq_a.n);
        try mm(e, ch, lay.wkv, ch.x, c.hidden, ch.y, .bf16, hd);
    } else {
        try grouped(e, ch, n, &.{ lay.wq_a, lay.wkv }, &.{ ch.x, ch.x }, &.{ c.hidden, c.hidden }, &.{ ch.qa, ch.y }, &.{ lay.wq_a.n, hd }, &.{ .bf16, .bf16 });
    }
    if (!owned_front) try tri_basic.rmsnorm(e.t, ch.qa, lay.wq_a.n, lay.q_norm, ch.qr, lay.wq_a.n, c.eps, n, lay.wq_a.n);
    try mm(e, ch, lay.wq_b, ch.qr, lay.wq_a.n, ch.q, .bf16, hl * hd);
    try tri_norm.ropeHeads(e.t, ch.q, rope.cos, rope.sin, ch.pos, rd, false, n, hl, hd);
    // the window keys: the ring's last window - 1 positions before start, then this chunk's rows
    const lo = @max(floor, start -| (c.window - 1));
    const wsrc_rows = if (ring_mode) ring_rows else start - lo + n;
    if (ring_mode) {
        // kv_norm_rope into ring rows pos % R (and its rows out), the attention's window keys the ring, WLO zero
        var slots: [ring_extra]i64 = undefined;
        for (0..n) |j| slots[j] = @intCast((start + j) % ring_rows);
        try upload(e, ch.rslot, &slots, n * 8);
        try tri_norm.kvNormRope(e.t, ch.y, lay.kv_norm, rope.cos, rope.sin, ch.pos, ring, ring_rows, ch.rslot, c.eps, true, rd, ch.wsrc, n, hd);
    } else if (start > lo) {
        // wsrc[:start - lo] = ring[arange(lo, start) % R]
        var ridx: [256]i64 = undefined;
        if (start - lo > ridx.len) return error.WindowTooLong;
        for (0..start - lo) |j| ridx[j] = @intCast((lo + j) % ring_rows);
        try upload(e, ch.ring_idx, &ridx, (start - lo) * 8);
        try fill(e, ch.invalid, 0, 4);
        try e.ops.gatherRows(e.s, ring, ring_rows, ch.ring_idx, ch.wsrc, hd * 2, start - lo, ch.invalid);
    }
    var lo64: i64 = if (ring_mode) 0 else @intCast(lo);
    try upload(e, ch.wlo, &lo64, 8);
    const kv = ch.wsrc + (start - lo) * hd * 2;
    // slots -1: the keys go to wsrc only (ring_mode is off); the ring is written after the attention
    if (!ring_mode) try tri_norm.kvNormRope(e.t, ch.y, lay.kv_norm, rope.cos, rope.sin, ch.pos, ring, ring_rows, ch.neg, c.eps, true, rd, kv, n, hd);
    var comp: tri_attn.Comp = .none;
    var n_idx: usize = 0;
    var padded = false;
    if (lay.ratio != 0) {
        const r: usize = lay.ratio;
        if (lay.comp_wkv != null and !kv_done) try kvSourceUpdate(e, ch, cs, sh, li);
        const src = sh.kv_layer orelse return error.NoKvSource;
        const n_comp_end = (start + n) / r;
        // vis = (pos + 1) // ratio: the compressed entries row i may see (host-made: the same integers)
        var vis_host: [4096]i64 = undefined;
        if (n > vis_host.len) return error.ChunkTooLong;
        for (0..n) |i| vis_host[i] = @intCast((start + i + 1) / r);
        try upload(e, ch.vis, &vis_host, n * 8);
        if (lay.idx_wq_b) |wqb| try indexer(e, ch, cs, sh, li, wqb, src, n_comp_end, rope);
        const kk = sh.kk orelse return error.NoIndexerSelection;
        comp = .{ .fp4 = .{ .codes = cs.comp_codes[src], .scales = cs.comp_scales[src] } };
        n_idx = kk;
        if (kk % 16 != 0 and (n > tri.decode_rows or one_split)) {
            // sparse_attn: a prompt chunk's pick list padded with -1 to whole 16-column tiles (F.pad), read from there
            const kp = (kk + 15) / 16 * 16;
            try fill(e, ch.cidxp, 0xff, n * kp * 8);
            try e.ops.copyRows(e.s, ch.cidx, kk * 8, ch.cidxp, kp * 8, kk * 8, n);
            padded = true;
        }
    }
    try tri_attn.sparseAttn(e.t, .{
        .q = ch.q,
        .out = ch.o,
        .rows = n,
        .h = hl,
        .hd = hd,
        .sink = lay.sink,
        .wsrc = if (ring_mode) ring else ch.wsrc,
        .wsrc_rows = wsrc_rows,
        .wlo = ch.wlo,
        .ring = ring_mode,
        .parts = .{ .pm = ch.pm_, .pl = ch.pl_, .po = ch.po_ },
        .comp = comp,
        .idx = if (n_idx > 0) (if (padded) ch.cidxp else ch.cidx) else 0,
        .n_idx = n_idx,
        .pos = ch.pos,
        .scale = scale(hd),
        .window = c.window,
        .one_split = one_split,
    });
    try tri_norm.ropeHeads(e.t, ch.o, rope.cos, rope.sin, ch.pos, rd, true, n, hl, hd);
    // ring[pos[-keep:] % R] = kv[-keep:] (ring mode wrote them before the attention)
    const keep: usize = if (ring_mode) 0 else @min(n, ring_rows);
    if (keep > 0) {
        var idx: [256]i64 = undefined;
        for (0..keep) |j| idx[j] = @intCast((start + n - keep + j) % ring_rows);
        try upload(e, ch.ring_idx, &idx, keep * 8);
        try e.ops.scatterRows(e.s, kv + (n - keep) * hd * 2, hd * 2, ch.ring_idx, ring, hd * 2, hd * 2, keep);
    }
    // wo_a: each group's column block of o read in place, written into its column block of u; then wo_b to fp32
    const groups = lay.groups;
    const gk = hl * hd / groups;
    const uw = groups * lay.wo_a[0].n;
    var col: usize = 0;
    if (n > 128) {
        for (lay.wo_a[0..groups], 0..) |wo, g| {
            try mm(e, ch, wo, ch.o + g * gk * 2, hl * hd, ch.u + col * 2, .bf16, uw);
            col += wo.n;
        }
    } else {
        // one group: the column blocks of o read in place, written into u's column blocks
        var xs: [4]u64 = undefined;
        var outs: [4]u64 = undefined;
        for (lay.wo_a[0..groups], 0..) |wo, g| {
            xs[g] = ch.o + g * gk * 2;
            outs[g] = ch.u + col * 2;
            col += wo.n;
        }
        const lds = [_]usize{ hl * hd, hl * hd, hl * hd, hl * hd };
        const oss = [_]usize{ uw, uw, uw, uw };
        const dts = [_]exl3_linear.DType{ .bf16, .bf16, .bf16, .bf16 };
        try grouped(e, ch, n, lay.wo_a[0..groups], xs[0..groups], lds[0..groups], outs[0..groups], oss[0..groups], dts[0..groups]);
    }
    if (e.two) |t| return t.woB(e, ch, lay); // 2D: the column partner's groups first, then this pair's columns
    try mm(e, ch, lay.wo_b, ch.u, uw, ch.pa, .fp32, c.hidden);
}

/// kv_source_update of a kv-source layer on the chunk (no earlier pending row: a chunk starting on a group boundary):
/// _compress's new latents (ratio 1: RMSNorm of the bf16 projection; ratio 2: of the pairs' softmax-weighted kv), its
/// positional store's last RAW rows, then the index keys (RMSNorm of idx_wk's projection, RoPE) and the RoPE'd latents
/// into the caches as FP4 (fp4_store).
fn kvSourceUpdate(e: *const Engine, ch: *Chunk, cs: *const Caches, sh: *Shared, li: usize) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    const n = ch.n;
    const start = ch.start;
    const hd = c.head_dim;
    const rd = c.rope_dim;
    const r: usize = lay.ratio;
    const cwkv = lay.comp_wkv.?;
    if (start % r != 0) return notPorted(@src()); // the group's earlier rows from the positional store
    const full = n / r;
    if (r == 1) {
        try mm(e, ch, cwkv, ch.x, c.hidden, ch.lat2, .bf16, hd);
        try e.exact.rmsNorm(e.s, ch.lat2, hd, lay.comp_norm, ch.lat, hd, n, hd, c.eps);
    } else {
        if (r != 2) return notPorted(@src());
        try mm(e, ch, cwkv, ch.x, c.hidden, ch.kvc, .fp32, hd);
        try mm(e, ch, lay.comp_wgate.?, ch.x, c.hidden, ch.scc, .fp32, hd);
        // rk[pw] = kv[-keep:], rs[pw] = score[-keep:]: pw = positions start + n - keep .. % RAW
        const keep: usize = @min(n, raw_rows);
        var idx: [raw_rows]i64 = undefined;
        for (0..keep) |j| idx[j] = @intCast((start + n - keep + j) % raw_rows);
        try upload(e, ch.raw_idx, &idx, keep * 8);
        try e.ops.scatterRows(e.s, ch.kvc + (n - keep) * hd * 4, hd * 4, ch.raw_idx, cs.raw_kv[li], hd * 4, hd * 4, keep);
        try e.ops.scatterRows(e.s, ch.scc + (n - keep) * hd * 4, hd * 4, ch.raw_idx, cs.raw_score[li], hd * 4, hd * 4, keep);
        if (full == 0) { // a one-row chunk (a prompt one past a chunk end): its row waits in the store, no new group
            sh.kv_layer = li;
            return;
        }
        try e.exact.compress2(e.s, ch.kvc, ch.scc, ch.lat2, full, hd);
        try e.exact.rmsNorm(e.s, ch.lat2, hd, lay.comp_norm, ch.lat, hd, full, hd, c.eps);
    }
    sh.kv_layer = li;
    // groups first // r .. and their positions group * r (the rotations _rot reads)
    var g_host: [4096]i64 = undefined;
    var p_host: [4096]i64 = undefined;
    if (full > g_host.len) return error.ChunkTooLong;
    for (0..full) |j| {
        g_host[j] = @intCast(start / r + j);
        p_host[j] = @intCast((start / r + j) * r);
    }
    try upload(e, ch.groups, &g_host, full * 8);
    try upload(e, ch.gpos, &p_host, full * 8);
    const rot = e.compressed;
    if (lay.idx_wk) |iwk| {
        const id = c.index_head_dim;
        try mmRows(e, ch, full, iwk, ch.lat, hd, ch.lat2, .bf16, id);
        try e.exact.rmsNorm(e.s, ch.lat2, id, lay.idx_k_norm, ch.ik, id, full, id, c.eps);
        try e.exact.rope(e.s, ch.ik, id, id - rd, rot.cos, rot.sin, ch.gpos, full, rd / 2, false);
        // switch "comp": the packed cache's rows in one launch (fp4_store: store_rows' bytes)
        try tri_attn.fp4Store(e.t, ch.ik, id, cs.idx_codes[li], cs.idx_scales[li], ch.groups, full, id, 32, false);
    }
    try e.ops.copyRows(e.s, ch.lat, hd * 2, ch.lat2, hd * 2, hd * 2, full);
    try e.exact.rope(e.s, ch.lat2, hd, hd - rd, rot.cos, rot.sin, ch.gpos, full, rd / 2, false);
    try tri_attn.fp4Store(e.t, ch.lat2, hd, cs.comp_codes[li], cs.comp_scales[li], ch.groups, full, hd, 16, true);
}

/// vis[r0..][0..m] at a 16-byte-aligned address, as the kit's indexer variants take VIS: the view, else a copy.
fn visBlock(e: *const Engine, ch: *const Chunk, r0: usize, m: usize) !u64 {
    const view = ch.vis + r0 * 8;
    if (view % 16 == 0) return view;
    try e.ops.copyRows(e.s, view, m * 8, ch.vis_blk, m * 8, m * 8, 1);
    return ch.vis_blk;
}

/// The indexer of a layer with index queries (switch "prompt_keys", layers before the candidate source): the
/// queries (idx_wq_b of q's latent, RoPE, fp4_qd's bytes), the heads' weights (x.float() @ idx_proj.t() through cuBLAS,
/// to bf16, times idx_dim ** -0.5 * idx_heads ** -0.5), each row's int64 keys of the kv source's index keys and the
/// top-k selection into cidx.
fn indexer(e: *const Engine, ch: *Chunk, cs: *const Caches, sh: *Shared, li: usize, wqb: weights.Linear, src: usize, n_comp_end: usize, rope: Rope) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    const n = ch.n;
    const ih = c.index_heads;
    const id = c.index_head_dim;
    if (n_comp_end == 0) { // cidx [n, 0]: no compressed entry yet (a one-token prompt), the attention reads none
        sh.kk = 0;
        return;
    }
    if (n_comp_end > ch.max_comp) return error.ChunkTooLong;
    const owned: ?row_state.Rows = if (row_state.indexEligible(row_index_enabled,
        ch.row_active, n_comp_end, c.candidate_source, c.layers))
        try row_state.rows(n, e.comm.rank, e.world) else null;
    if (owned) |r| {
        // Same K/N kernel, only independent M rows are owned here. This does
        // not pretend that the full EXL3 weight unpack has become smaller.
        try mmRows(e, ch, r.count, wqb, ch.qr + r.first * lay.wq_a.n * 2,
            lay.wq_a.n, ch.iq + r.first * ih * id * 2, .bf16, ih * id);
        try tri_norm.ropeHeads(e.t, ch.iq + r.first * ih * id * 2, rope.cos,
            rope.sin, ch.pos + r.first * 8, c.rope_dim, false, r.count, ih, id);
        try tri_attn.fp4QdP2(e.t, ch.iq + r.first * ih * id * 2,
            ch.iq4 + r.first * ih * id * 2, r.count * ih * id);
        ch.row_stats.query_rows += r.count;
    } else {
    try mm(e, ch, wqb, ch.qr, lay.wq_a.n, ch.iq, .bf16, ih * id);
    try tri_norm.ropeHeads(e.t, ch.iq, rope.cos, rope.sin, ch.pos, c.rope_dim, false, n, ih, id);
    // switch "idx": fp4_qd's bytes in one launch
    try tri_attn.fp4QdP2(e.t, ch.iq, ch.iq4, n * ih * id);
    }
    if (n <= tri.decode_rows) {
        // a decode-sized chunk: the row-invariant matmul of the fp16 projection (rowmm2)
        try tri_norm.rowmm2(e.t, ch.x, c.hidden, lay.idx_proj_h, ch.wl, n, c.hidden, ih);
    } else {
        try e.ops.toF32(e.s, ch.x, ch.xf, n * c.hidden);
        try e.blas.xwT(ch.xf, lay.idx_proj, ch.wl, n, c.hidden, ih);
    }
    try e.exact.bf16Scale(e.s, ch.wl, ch.iw, n * ih, exact.indexScale(id, ih));
    const kk = @min(c.index_topk, n_comp_end);
    // rows in blocks so the [rows, n_comp] keys stay bounded at long contexts
    const rb = @max(16, @min(n, (@as(usize, 1) << 26) / (4 * @max(n_comp_end, 1))));
    if (rb * n_comp_end > key_elems) return error.ChunkTooLong;
    const keyed = li != c.candidate_source and !(c.candidate_source < li) and kk & (kk - 1) == 0;
    const k: tri_index.IndexK = .{ .fp4 = .{ .codes = cs.idx_codes[src], .scales = cs.idx_scales[src] } };
    const torch_topk: tri_index.TopK = .{ .top = ch.ktop, .ctx = @ptrCast(@constCast(e)), .run = torchTopk };
    if (keyed and tri_index.prunes(n_comp_end, kk) and (rb > pruned_rows or ch.pbufs.cand == 0)) return error.ChunkTooLong;
    // the candidate source and the layers after it: fp32 scores, the pool (_candidates at the source, apply_candidates
    // after it), topk_indices and -1 past vis. While every block of the scores fits in the pool (cand_blocks of them)
    // the pool keeps each block with a finite score (the newest pinned: it holds the row's last visible key) and
    // apply_candidates rewrites only scores already -inf: no launch changes a byte. Past it the source's block maxima
    // (the newest block pinned to +inf) pick the pool's cand_blocks blocks (topk_indices: ties to the lower block), the
    // picked blocks with a finite maximum make the chunk's mask, and the later layers' scores outside it become -inf.
    const cb = c.candidate_block;
    const nbk = (n_comp_end + cb - 1) / cb;
    const pooled = !keyed and nbk > c.candidate_blocks;
    if (pooled and (ch.pool == 0 or nbk > ch.pool_ms)) return error.ChunkTooLong;
    const pk = @min(c.candidate_blocks, nbk);
    const row_end = if (owned) |r| r.end() else n;
    var r0: usize = if (owned) |r| r.first else 0;
    while (r0 < row_end) : (r0 += rb) {
        // the block's rows: iq[r0:r1] and wts[r0:r1] as views (their offsets the served pointers'), vis[r0:r1] at a
        // 16-byte-aligned address (visBlock), its selection into cidx[r0:r1]
        const m = @min(row_end, r0 + rb) - r0;
        const q = ch.iq4 + r0 * ih * id * 2;
        const wb = ch.iw + r0 * ih * 2;
        const vr = try visBlock(e, ch, r0, m);
        const out = ch.cidx + r0 * kk * 8;
        if (keyed) {
            try tri_index.indexScore(e.t, q, k, wb, vr, n_comp_end, ch.keys, null, true, ch.tmax, false, m, ih, id);
            try tri_index.topkSelectPruned(e.t, ch.keys, n_comp_end, ch.tmax, kk, vr, out, ch.pbufs, torch_topk, m, n_comp_end);
        } else {
            try tri_index.indexScore(e.t, q, k, wb, vr, n_comp_end, ch.score, null, false, null, false, m, ih, id);
            const mrows = ch.pool + r0 * ch.pool_ms;
            if (pooled and li == c.candidate_source) {
                try e.ops.blockMax(e.s, ch.score, n_comp_end, n_comp_end, cb, vr, ch.bmax, nbk, m);
                try e.ops.topkIndices(e.s, ch.bmax, nbk, m, nbk, pk, ch.every, ch.pidx);
                try e.ops.poolPick(e.s, ch.bmax, nbk, nbk, ch.pidx, pk, mrows, ch.pool_ms, 0, m);
            } else if (pooled and c.candidate_source < li) {
                if (!sh.pool) return error.NoCandidatePool;
                try e.ops.applyPool(e.s, ch.score, n_comp_end, n_comp_end, cb, mrows, ch.pool_ms, m);
            }
            try e.ops.topkIndices(e.s, ch.score, n_comp_end, m, n_comp_end, kk, vr, out);
        }
    }
    if (owned) |r| {
        // Only the finished integer selections cross the network. The full
        // latent-key cache remains available to every row owner.
        try rowGather(e, ch, ch.cidx, kk * 8, 0, n);
        ch.row_stats.index_rows += r.count;
    }
    if (pooled and li == c.candidate_source) sh.pool = true;
    sh.kk = kk;
}

/// topk_select's torch step, keys.topk(k).values: the top-k set of the unique int64 keys (ops.topkI64).
pub fn torchTopk(ctx: ?*anyopaque, _: tri.Tri, keys: u64, ks: usize, rows: usize, n: usize, k: usize, top: u64) anyerror!void {
    const e: *const Engine = @ptrCast(@alignCast(ctx.?));
    try e.ops.topkI64(e.s, keys, ks, rows, n, k, top);
}

/// hd ** -0.5 as Triton passes the Python float: rounded to fp32.
pub fn scale(hd: usize) f32 {
    return @floatCast(std.math.pow(f64, @floatFromInt(hd), -0.5));
}

/// Comm.gather: [world, n, D] fp32 of every rank's partial, in rank order.
pub fn gather(e: *const Engine, ch: *Chunk, src: u64, dst: u64) !void {
    if (ch.row_active) return rowPartials(e, ch, src, dst, e.c.hidden);
    if (e.two) |t| return t.quarters(e, src, dst, ch.n, t.ow, 4); // 2D: the same [2, n, D] from the four quarters
    try e.comm.allGather(src, dst, ch.n * e.c.hidden, .f32, e.s);
}

/// Layer `li`'s FFN mixes: hc_pre_pf with the gathered attention partials posted into the streams first; x the MoE's
/// input rows, pre_f / post / comb the FFN's.
pub fn ffnMixes(e: *const Engine, ch: *Chunk, li: usize) !void {
    if (ch.row_active) return rowFfnMixes(e, ch, li);
    return ffnMixesRows(e, ch, li);
}

fn ffnMixesRows(e: *const Engine, ch: *Chunk, li: usize) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    const src = try tri_hc.hcPrePf(e.t, ch.h, lay.hc_ffn[0], lay.hc_ffn[1], lay.hc_ffn[2], ch.pre_a, lay.ffn_norm, c.eps, c.hc_eps, c.hc_iters, ch.x, ch.pre_f, ch.post, ch.comb, ch.part, .{ .gathered = ch.ga, .world = e.world, .h_out = ch.h_alt }, ch.n, c.hidden);
    swapStreams(ch, src);
}

/// Model.moe of a prompt chunk: the gate's logits in fp32 (x.float() @ gate_w.float().t(), one cuBLAS GEMM as torch
/// issues it), the routing (the shared expert in every row's last slot) and the routed experts: the partial pm [n, D].
pub fn moe(e: *const Engine, ch: *Chunk, li: usize) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    const n = ch.n;
    const d = c.hidden;
    const sl = e.slots();
    const shared_id = lay.experts.count - 1;
    var kc: usize = 0;
    if (n <= tri.decode_rows) {
        // a decode-sized chunk: the row-invariant gate (rowmm_gate's chunk sums) and its routing
        kc = try tri_norm.rowmmGate(e.t, ch.x, d, lay.gate_w, ch.gl, n, d, c.experts);
        try tri_norm.route(e.t, ch.gl, kc, lay.gate_b, c.top_k, c.routed_scaling, shared_id, ch.pick, ch.wts, n, c.experts, sl);
    } else {
        try e.ops.toF32(e.s, ch.x, ch.xf, n * d);
        try e.ops.f16ToF32(e.s, lay.gate_w, ch.gate_f, c.experts * d);
        try e.blas.xwT(ch.xf, ch.gate_f, ch.logits, n, d, c.experts);
        try tri_norm.route(e.t, ch.logits, 0, lay.gate_b, c.top_k, c.routed_scaling, shared_id, ch.pick, ch.wts, n, c.experts, sl);
    }
    try routeSpans(e, ch, lay, shared_id, sl, kc);
    if (e.two) |t| return t.experts(e, ch, lay); // 2D: gate / up here, the intermediate's exchange, down here
    // routed(): a chunk of fewer than EXACT_ROWS rows takes the decode window's fused path and scratch
    if (n < exl3_experts.exact_rows) return exl3_experts.decode(e.ex, e.s, lay.experts, ch.xsd, ch.x, d, ch.pick, ch.wts, ch.pm, n, c.swiglu_limit);
    try exl3_experts.prompt(e.ex, e.s, lay.experts, ch.xs, ch.x, d, ch.pick, ch.wts, ch.pm, n, c.swiglu_limit);
}

/// The VL routing (Gate.forward's bias = where(image_mask, bias_vl, bias)) as the served build runs it (recording v0's
/// launches): every row routed with the text bias, then each span's rows of the gate's output (the logits, or a
/// decode-sized chunk's chunk sums [n, kc, NE]) routed again with the image bias in a launch of their own, their picks and
/// weights put over the text ones. Row by row the routing is independent, so the text rows keep theirs.
fn routeSpans(e: *const Engine, ch: *Chunk, lay: weights.Layer, shared_id: usize, sl: usize, kc: usize) !void {
    if (ch.spans.len == 0) return;
    if (lay.gate_b_vl == 0) return error.NoImageRoutingBias;
    const c = e.c;
    const n = ch.n;
    const src = if (n <= tri.decode_rows) ch.gl else ch.logits;
    const row = @max(kc, 1) * c.experts * 4;
    for (ch.spans) |sp| {
        const o = overlap(sp, ch.start, n) orelse continue;
        const rows = o[1] - o[0];
        try tri_norm.route(e.t, src + o[0] * row, kc, lay.gate_b_vl, c.top_k, c.routed_scaling, shared_id, ch.pick_vl, ch.wts_vl, rows, c.experts, sl);
        const dd = e.d;
        try dd.check(dd.api.cuMemcpyDtoDAsync_v2(ch.pick + o[0] * sl * 4, ch.pick_vl, rows * sl * 4, e.s.handle), "cuMemcpyDtoDAsync");
        try dd.check(dd.api.cuMemcpyDtoDAsync_v2(ch.wts + o[0] * sl * 4, ch.wts_vl, rows * sl * 4, e.s.handle), "cuMemcpyDtoDAsync");
    }
}

/// Engram on the host: the hasher (the compressed token map, the multipliers), this rank's hash columns [lo, hi) of
/// each Engram layer's, its tables and readers, and host buffers for a chunk's rows (model.py's Engram).
pub const EngramHost = struct {
    hasher: engram.Hasher,
    tables: *const engram_io.Tables,
    pool: *engram_io.Pool,
    lo: usize,
    hi: usize,
    hashes: []i64, // [cap][layers][cols]
    flat: []i64, // [cap * (hi - lo)]
    w: []u8, // FP8 rows
    s: []u8, // their E8M0 scales
    rows: []u16, // bf16 [cap, (hi - lo) * head_dim]
    aio: ?*engram_aio.Aio = null, // a round's reads by Linux AIO on the tables' O_DIRECT descriptors (else the pool)
    paio: ?*engram_aio.Aio = null, // a prompt chunk's reads the same way, on a ring of its own (else the pool)
    // the rounds' Engram host time (ns, summed; a profile reads it): hashing, the table reads, the decode, the upload
    io: ?std.Io = null,
    t_hash: u64 = 0,
    t_read: u64 = 0,
    t_decode: u64 = 0,
    t_upload: u64 = 0,
    calls: u64 = 0,
    // a prompt chunk's rows read ahead on a thread (engramReadAhead): Engram layer j's rows of the chunk at
    // [ahead_start, ahead_start + ahead_n), ready when ahead_ready[j]
    ahead_w: [2][]u8 = .{ &.{}, &.{} },
    ahead_s: [2][]u8 = .{ &.{}, &.{} },
    ahead_rows: [2][]u16 = .{ &.{}, &.{} },
    ahead_flat: [2][]i64 = .{ &.{}, &.{} },
    ahead_ready: [2]bool = .{ false, false },
    ahead_start: usize = 0,
    ahead_n: usize = 0,
    ahead_err: ?anyerror = null,
    // a profile's prompt Engram split (ns summed, the stream synchronized at each cut): the rows ready on the host,
    // their upload, the projection, the quarters' exchange, the sum and gate
    p_rows: u64 = 0,
    p_upload: u64 = 0,
    p_mm: u64 = 0,
    p_x: u64 = 0,
    p_gate: u64 = 0,
    p_calls: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, c: *const Config, hasher: engram.Hasher, tables: *const engram_io.Tables, pool: *engram_io.Pool, rank: usize, world: usize, cap: usize) !EngramHost {
        const cols = hasher.cols();
        const lo = rank * cols / world;
        const hi = (rank + 1) * cols / world;
        const k = hi - lo;
        const t = tables.layers.get(@intCast(c.engram_layers.slice()[0])) orelse return error.MissingEngramTable;
        var aw: [2][]u8 = .{ &.{}, &.{} };
        var as: [2][]u8 = .{ &.{}, &.{} };
        var ar: [2][]u16 = .{ &.{}, &.{} };
        var af: [2][]i64 = .{ &.{}, &.{} };
        for (0..@min(2, c.engram_layers.slice().len)) |j| {
            const tj = tables.layers.get(@intCast(c.engram_layers.slice()[j])) orelse return error.MissingEngramTable;
            aw[j] = try gpa.alloc(u8, cap * k * tj.row_w);
            as[j] = try gpa.alloc(u8, cap * k * tj.row_s);
            ar[j] = try gpa.alloc(u16, cap * k * tj.row_w);
            af[j] = try gpa.alloc(i64, cap * k);
        }
        return .{
            .ahead_w = aw,
            .ahead_s = as,
            .ahead_rows = ar,
            .ahead_flat = af,
            .hasher = hasher,
            .tables = tables,
            .pool = pool,
            .lo = lo,
            .hi = hi,
            .hashes = try gpa.alloc(i64, cap * hasher.layers * cols),
            .flat = try gpa.alloc(i64, cap * k),
            .w = try gpa.alloc(u8, cap * k * t.row_w),
            .s = try gpa.alloc(u8, cap * k * t.row_s),
            .rows = try gpa.alloc(u16, cap * k * t.row_w),
        };
    }
};

/// A prompt chunk's Engram layer j rows ahead of its layer, on a thread of the caller's while the GPU runs the layers
/// before it: the n-grams hashed (j 0: every layer's at once), the rows read and decoded into ahead_rows[j], exactly
/// as engramApply would at the layer. Errors land in ahead_err.
pub fn engramReadAhead(eh: *EngramHost, c: *const Config, seq: []const i32, start: usize, n: usize, j: usize) void {
    engramAhead1(eh, c, seq, start, n, j) catch |err| {
        eh.ahead_err = err;
    };
}

/// The next chunk's rows of both Engram layers (fillChunk starts it once this chunk's last Engram layer is applied).
pub fn engramReadAheadAll(eh: *EngramHost, c: *const Config, seq: []const i32, start: usize, n: usize) void {
    for (0..@min(2, c.engram_layers.slice().len)) |j| {
        engramAhead1(eh, c, seq, start, n, j) catch |err| {
            eh.ahead_err = err;
            return;
        };
    }
}

fn engramAhead1(eh: *EngramHost, c: *const Config, seq: []const i32, start: usize, n: usize, j: usize) !void {
    const li = c.engram_layers.slice()[j];
    const t = eh.tables.layers.get(@intCast(li)) orelse return error.MissingEngramTable;
    const cols = eh.hasher.cols();
    const k = eh.hi - eh.lo;
    if (j == 0) eh.hasher.hashes(seq, start, n, eh.hashes[0 .. n * eh.hasher.layers * cols]);
    for (0..n) |r| {
        for (0..k) |q| eh.ahead_flat[j][r * k + q] = eh.hashes[(r * eh.hasher.layers + j) * cols + eh.lo + q];
    }
    const m = n * k;
    // by AIO on the prompt's ring when it is open (the main thread reads by it only after joining this one)
    if (eh.paio) |aio| try aioGather(aio, t, eh.ahead_flat[j][0..m], eh.ahead_w[j][0 .. m * t.row_w], eh.ahead_s[j][0 .. m * t.row_s]) else try eh.pool.gather(t, eh.ahead_flat[j][0..m], eh.ahead_w[j][0 .. m * t.row_w], eh.ahead_s[j][0 .. m * t.row_s]);
    engram.decodeRows(eh.ahead_w[j][0 .. m * t.row_w], eh.ahead_s[j][0 .. m * t.row_s], eh.ahead_rows[j][0 .. m * t.row_w], m, t.row_w, t.row_s, 8);
    eh.ahead_ready[j] = true;
}

/// Layer li's Engram on the chunk (engram_apply, no image span), as _forward_k runs it before the attention mixes: a
/// pending MoE post into the streams first (hc_post), then the rows of each position's n-grams (hashed on the host from
/// `seq`, the whole sequence's ids so far; read from the tables; decoded to bf16 as Engram._decode does), their
/// projection summed over the ranks in rank order, rounded to bf16, and the gate into the streams.
pub fn engramApply(e: *const Engine, ch: *Chunk, eh: *EngramHost, li: usize, seq: []const i32) !void {
    if (ch.row_active) try rowPost(e, ch);
    const lay = e.w.layers[li];
    const ew = lay.engram_wkv orelse return;
    const c = e.c;
    const n = ch.n;
    if (ch.pending) {
        try tri_basic.hcPost(e.t, ch.gm, ch.h, ch.post, ch.comb, ch.h, e.world, n, c.hidden);
        ch.pending = false;
    }
    const li32: u16 = @intCast(li);
    const l = std.mem.indexOfScalar(u16, c.engram_layers.slice(), li32) orelse return error.NotAnEngramLayer;
    const t = eh.tables.layers.get(@intCast(li)) orelse return error.MissingEngramTable;
    const cols = eh.hasher.cols();
    const k = eh.hi - eh.lo;
    const m = n * k;
    const prof = eh.io != null;
    var tq = if (prof) pstamp(e, eh) else 0;
    // the rows read ahead (fillChunk's thread) when they are this chunk's, else read here
    var rows: []u16 = eh.rows;
    if (l < 2 and eh.ahead_ready[l] and eh.ahead_start == ch.start and eh.ahead_n == n) {
        rows = eh.ahead_rows[l];
        eh.ahead_ready[l] = false;
    } else {
        eh.hasher.hashes(seq, ch.start, n, eh.hashes[0 .. n * eh.hasher.layers * cols]);
        for (0..n) |r| {
            for (0..k) |j| eh.flat[r * k + j] = eh.hashes[(r * eh.hasher.layers + l) * cols + eh.lo + j];
        }
        if (eh.paio) |aio| try aioGather(aio, t, eh.flat[0..m], eh.w[0 .. m * t.row_w], eh.s[0 .. m * t.row_s]) else try eh.pool.gather(t, eh.flat[0..m], eh.w[0 .. m * t.row_w], eh.s[0 .. m * t.row_s]);
        engram.decodeRows(eh.w[0 .. m * t.row_w], eh.s[0 .. m * t.row_s], eh.rows[0 .. m * t.row_w], m, t.row_w, t.row_s, 8); // the rows on 8 threads
    }
    if (m * t.row_w != n * ew.k) return error.EngramShape;
    if (prof) eh.p_rows += lap(e, eh, &tq);
    try upload(e, ch.eb, rows.ptr, m * t.row_w * 2);
    if (prof) eh.p_upload += lap(e, eh, &tq);
    try mm(e, ch, ew, ch.eb, ew.k, ch.ek, .fp32, ew.n);
    if (prof) eh.p_mm += lap(e, eh, &tq);
    // 2D (model.py esum): this node's columns of its rank's projection; the quarters give both ranks' whole ones
    const en = if (e.two) |two| two.engramWidth() else ew.n;
    // 2D chunks past the rings: the pair's sum then the pairs' bf16 halves (Two.engramSum), the same bits
    const psum = if (e.two) |two| two.rings == null or n >= 64 else false;
    if (ch.row_active) {
        const r = try row_state.rows(n, e.comm.rank, e.world);
        try rowPartials(e, ch, ch.ek, ch.ekg, en);
        try e.ops.sumRanksBf16(e.s, ch.ekg, ch.kv, r.count * en, e.world);
        if (prof) eh.p_x += lap(e, eh, &tq);
    } else if (psum) {
        try e.two.?.engramSum(e, ch.ek, ch.ekg, ch.kv, n);
        if (prof) eh.p_x += lap(e, eh, &tq);
    } else {
        if (e.two) |two| try two.quarters(e, ch.ek, ch.ekg, n, two.ew, 4) else try e.comm.allGather(ch.ek, ch.ekg, n * ew.n, .f32, e.s);
        if (prof) eh.p_x += lap(e, eh, &tq);
        try e.ops.sumRanksBf16(e.s, ch.ekg, ch.kv, n * en, if (e.two != null) 2 else e.world);
    }
    if (ch.row_active) {
        const r = try row_state.rows(n, e.comm.rank, e.world);
        const offset = r.first * c.hc * c.hidden * 2;
        try tri_basic.engramGate(e.t, ch.h + offset, ch.kv, lay.engram_qk,
            ch.h_alt + offset, c.eps, r.count, c.hidden);
        ch.row_stats.gate_rows += r.count;
    } else try tri_basic.engramGate(e.t, ch.h, ch.kv, lay.engram_qk, ch.h_alt, c.eps, n, c.hidden);
    swapStreams(ch, ch.h_alt);
    // image spans take no Engram contribution: their rows keep the streams they came with (engram_mask)
    const rowb = c.hc * c.hidden * 2;
    for (ch.spans) |sp| {
        const o = overlap(sp, ch.start, n) orelse continue;
        try e.ops.copyRows(e.s, ch.h_alt + o[0] * rowb, rowb, ch.h + o[0] * rowb, rowb, rowb, o[1] - o[0]);
    }
    if (prof) {
        eh.p_gate += lap(e, eh, &tq);
        eh.p_calls += 1;
    }
}

/// A profile's clock after the stream drains (prompt Engram split).
fn pstamp(e: *const Engine, eh: *const EngramHost) u64 {
    e.s.synchronize() catch {};
    return @intCast(std.Io.Timestamp.now(eh.io.?, .awake).nanoseconds);
}

fn lap(e: *const Engine, eh: *const EngramHost, t: *u64) u64 {
    const now = pstamp(e, eh);
    defer t.* = now;
    return now - t.*;
}

/// A chunk's rows `ids` of table `t` by AIO (the pool's bytes, at ~0.07 s for 24,576 rows where the pool's preads take
/// ~0.5 s on one GB10): batches of a quarter of the ring's slots (two reads a row), all submitted (a submit reaps while the
/// ring is full), then waited for in order.
fn aioGather(aio: *engram_aio.Aio, t: engram_io.Table, ids: []const i64, out_w: []u8, out_s: []u8) !void {
    const per = @max(aio.slots / 4, 1);
    var batch: [64]u64 = undefined;
    var nb: usize = 0;
    var at: usize = 0;
    errdefer for (batch[0..nb]) |id| aio.wait(id) catch {};
    while (at < ids.len) : (at += per) {
        const n = @min(per, ids.len - at);
        if (nb == batch.len) return error.TooManyRows;
        batch[nb] = try aio.start(t, ids[at..][0..n], out_w[at * t.row_w ..][0 .. n * t.row_w], out_s[at * t.row_s ..][0 .. n * t.row_s]);
        nb += 1;
    }
    for (batch[0..nb], 0..) |id, i| {
        aio.wait(id) catch |err| {
            for (batch[i + 1 .. nb]) |rest| aio.wait(rest) catch {};
            nb = 0;
            return err;
        };
    }
    nb = 0;
}

/// The decoder's bounded replay at its first layer (_forward_k's `replay`, CED's prefill): the layer's attention mixes
/// (the pending MoE post fused in) and its kv-source update over every row of the chunk, then the chunk cut to its rows
/// at positions >= replay, the streams and their pre copied down (h[first:], pre[first:]). The decoder layers then run
/// on those rows with no window key before replay and this layer's update done. False: an encoder-only chunk (no
/// row at or past replay), which ends here.
pub fn replayCut(e: *const Engine, ch: *Chunk, cs: *const Caches, sh: *Shared, li: usize, replay: usize, host_pos: []i64) !bool {
    try attnMixes(e, ch, li);
    if (e.w.layers[li].comp_wkv == null) return error.NoKvSource;
    try kvSourceUpdate(e, ch, cs, sh, li);
    const first = @max(ch.start, replay) - ch.start;
    if (first >= ch.n) return false;
    if (ch.row_active) try rowLeave(e, ch, li, first);
    if (first == 0) return true;
    const c = e.c;
    const n = ch.n - first;
    const row_h = c.hc * c.hidden * 2;
    try e.ops.copyRows(e.s, ch.h + first * row_h, row_h, ch.h_alt, row_h, row_h, n);
    std.mem.swap(u64, &ch.h, &ch.h_alt);
    const row_p = c.hc * 4;
    try e.ops.copyRows(e.s, ch.pre + first * row_p, row_p, ch.pre_f, row_p, row_p, n);
    std.mem.swap(u64, &ch.pre, &ch.pre_f);
    ch.start += first;
    ch.n = n;
    // pos[first:]: a view, first * 8 bytes in (its alignment is the one the served kernels were specialized for)
    _ = host_pos;
    ch.pos += first * 8;
    return true;
}

/// DSpark tap j (_forward_k at a tap layer): a pending MoE gather posted into the streams first (hc_post, not fused),
/// then h.to(fp32).mean(1).to(bf16) into tap j's rows.
pub fn tap(e: *const Engine, ch: *Chunk, j: usize) !void {
    if (ch.row_active) try rowPost(e, ch);
    const c = e.c;
    if (ch.pending) {
        try tri_basic.hcPost(e.t, ch.gm, ch.h, ch.post, ch.comb, ch.h, e.world, ch.n, c.hidden);
        ch.pending = false;
    }
    if (ch.row_active) {
        const r = try row_state.rows(ch.n, e.comm.rank, e.world);
        try e.exact.hcMean4(e.s, ch.h + r.first * c.hc * c.hidden * 2,
            ch.taps + (j * ch.cap + r.first) * c.hidden * 2, c.hidden, r.count, c.hidden);
    } else try e.exact.hcMean4(e.s, ch.h, ch.taps + j * ch.cap * c.hidden * 2, c.hidden, ch.n, c.hidden);
}

/// Tap j's rows (bf16 [n, D]).
pub fn tapRows(e: *const Engine, ch: *const Chunk, j: usize) u64 {
    return ch.taps + j * ch.cap * e.c.hidden * 2;
}

/// The head on the chunk's last row (_forward_k without all_logits): the pending MoE post, collapse_norm of the row with
/// its pre, the head's columns (one row through the head's own group, fp32) and their gather: head_g holds the
/// prompt's logits, the ranks' columns in rank order.
pub fn head(e: *const Engine, ch: *Chunk) !void {
    if (ch.row_active) return error.RowStateMissingCedHandoff;
    const c = e.c;
    const n = ch.n;
    if (ch.pending) {
        try tri_basic.hcPost(e.t, ch.gm, ch.h, ch.post, ch.comb, ch.h, e.world, n, c.hidden);
        ch.pending = false;
    }
    const last_h = ch.h + (n - 1) * c.hc * c.hidden * 2;
    const last_pre = ch.pre + (n - 1) * c.hc * 4;
    try tri_basic.collapseNorm(e.t, last_h, last_pre, e.w.norm, ch.head_x, c.eps, 1, c.hidden);
    const hl = e.w.head;
    try mmRows(e, ch, 1, hl, ch.head_x, c.hidden, ch.head_l, .fp32, hl.n);
    if (e.two) |t| return t.quarters(e, ch.head_l, ch.head_g, 1, t.hw, 4); // 2D: the vocabulary quarters, TP2's layout
    try e.comm.allGather(ch.head_l, ch.head_g, hl.n, .f32, e.s);
}

/// A layer's end: the MoE's gather is pending (the next layer's attention mixes post it) and the FFN's pre_out is the
/// next layer's pre (_forward_k's `pre, pre_f = pre_f, pre`).
pub fn endLayer(ch: *Chunk) void {
    ch.pending = true;
    std.mem.swap(u64, &ch.pre, &ch.pre_f);
}

test "hd ** -0.5 as the served build passes it" {
    try std.testing.expectEqual(@as(u32, 0x3d3504f3), @as(u32, @bitCast(scale(512))));
}

/// error.NotPortedYet with its site on stderr (a request that reaches a path the port lacks names it in the log).
fn notPorted(src: std.builtin.SourceLocation) error{NotPortedYet} {
    std.debug.print("{{\"not_ported\": \"{s}:{d}\"}}\n", .{ src.file, src.line });
    return error.NotPortedYet;
}

test "short continuation avoids uncaptured unbased compressed ring launches" {
    for ([_]usize{ 1, 2, 15, 16 }) |n| {
        try std.testing.expect(promptRingMode(n, false, false, false));
        try std.testing.expect(!promptRingMode(n, false, false, true));
        try std.testing.expect(!promptRingMode(n, true, true, false));
    }
    for ([_]usize{ 17, 128, 2048 }) |n| {
        try std.testing.expect(!promptRingMode(n, false, false, false));
        try std.testing.expect(!promptRingMode(n, false, false, true));
    }
}

fn rowView(e: *const Engine, ch: *const Chunk) !Chunk {
    const r = try row_state.rows(ch.n, e.comm.rank, e.world);
    const c = e.c;
    var v = ch.*;
    v.row_active = false; // the original kernels see one contiguous local view
    v.n = r.count;
    v.h += r.first * c.hc * c.hidden * 2;
    v.h_alt += r.first * c.hc * c.hidden * 2;
    v.x += r.first * c.hidden * 2;
    v.pre += r.first * c.hc * 4;
    v.pre_a += r.first * c.hc * 4;
    v.pre_f += r.first * c.hc * 4;
    v.post += r.first * c.hc * 4;
    v.comb += r.first * c.hc * c.hc * 4;
    return v;
}

fn rowMerge(e: *const Engine, ch: *Chunk, v: *const Chunk) !void {
    const r = try row_state.rows(ch.n, e.comm.rank, e.world);
    const offset = r.first * e.c.hc * e.c.hidden * 2;
    ch.h = v.h - offset;
    ch.h_alt = v.h_alt - offset;
    ch.pending = v.pending;
}

fn rowGather(e: *const Engine, ch: *Chunk, buffer: u64, row_bytes: usize, first: usize, end: usize) !void {
    if (first > end or end > ch.n) return error.BadRowRange;
    var firsts: [3]usize = undefined;
    var counts: [3]usize = undefined;
    for (0..3) |rank| {
        const r = row_state.intersection(try row_state.rows(ch.n, rank, e.world), first, end);
        firsts[rank] = r.first; counts[rank] = r.count;
    }
    try e.comm.gatherOwnedRows(buffer, firsts, counts, row_bytes, e.s);
    ch.row_stats.exchanges += 1;
    ch.row_stats.gathered_rows += end - first;
    ch.row_stats.sent_bytes += counts[e.comm.rank] * row_bytes * (e.world - 1);
}

fn rowPartials(e: *const Engine, ch: *Chunk, src: u64, dst: u64, width: usize) !void {
    var firsts: [3]usize = undefined;
    var counts: [3]usize = undefined;
    for (0..3) |rank| {
        const r = try row_state.rows(ch.n, rank, e.world);
        firsts[rank] = r.first; counts[rank] = r.count;
    }
    const rank = e.comm.rank;
    const bytes = width * 4;
    // Self contribution goes into its original source-rank position.
    try e.ops.copyRows(e.s, src + firsts[rank] * bytes, bytes,
        dst + rank * counts[rank] * bytes, bytes, bytes, counts[rank]);
    try e.comm.scatterRows(src, dst, firsts, counts, bytes, e.s);
    ch.row_stats.exchanges += 1;
    ch.row_stats.partial_rows += counts[rank];
    ch.row_stats.sent_bytes += (ch.n - counts[rank]) * bytes;
}

fn rowPost(e: *const Engine, ch: *Chunk) !void {
    if (!ch.pending) return;
    var v = try rowView(e, ch);
    try tri_basic.hcPost(e.t, v.gm, v.h, v.post, v.comb, v.h, e.world, v.n, e.c.hidden);
    v.pending = false;
    try rowMerge(e, ch, &v);
}

fn rowAttnMixes(e: *const Engine, ch: *Chunk, li: usize) !void {
    var v = try rowView(e, ch);
    try attnMixesRows(e, &v, li);
    try rowMerge(e, ch, &v);
    ch.row_stats.mhc_rows += v.n;
    const lay = e.w.layers[li];
    // Compressor and original cuBLAS index-weight projection still require
    // global X. Ordinary layers need only the owner's local X for QR/KV.
    if (!row_front_enabled or row_state.attentionNeedsFullX(lay.idx_wq_b != null, lay.comp_wkv != null)) {
        try rowGather(e, ch, ch.x, e.c.hidden * 2, 0, ch.n);
    } else ch.row_stats.wide_x_gathers_skipped += 1;
}

fn rowFfnMixes(e: *const Engine, ch: *Chunk, li: usize) !void {
    var v = try rowView(e, ch);
    try ffnMixesRows(e, &v, li);
    try rowMerge(e, ch, &v);
    ch.row_stats.mhc_rows += v.n;
    try rowGather(e, ch, ch.x, e.c.hidden * 2, 0, ch.n);
}

/// The CED encoder-only chunks need no full residual exchange. On the last
/// chunk gather just the retained tail and the already produced drafter taps.
fn rowLeave(e: *const Engine, ch: *Chunk, li: usize, first: usize) !void {
    const c = e.c;
    try rowGather(e, ch, ch.h, c.hc * c.hidden * 2, first, ch.n);
    try rowGather(e, ch, ch.pre, c.hc * 4, first, ch.n);
    for (c.dspark_taps.slice(), 0..) |layer, j| {
        if (layer < li) try rowGather(e, ch, ch.taps + j * ch.cap * c.hidden * 2,
            c.hidden * 2, first, ch.n);
    }
    ch.row_active = false;
}

pub fn rowReport(e: *const Engine, ch: *const Chunk, start: usize, input_rows: usize) void {
    if (ch.row_stats.owned_rows == 0) return;
    const s = ch.row_stats;
    std.debug.print("prefill_row_state: rank={d} start={d} input_rows={d} owned_rows={d} mhc_rows={d} gate_rows={d} partial_rows={d} gathered_rows={d} exchanges={d} sent_bytes={d} query_rows={d} index_rows={d} front_rows={d} wide_x_gathers_skipped={d}\n", .{
        e.comm.rank, start, input_rows, s.owned_rows, s.mhc_rows, s.gate_rows,
        s.partial_rows, s.gathered_rows, s.exchanges, s.sent_bytes, s.query_rows, s.index_rows, s.front_rows, s.wide_x_gathers_skipped });
}

/// ga's previous FP32 partials have been consumed by attnMixesRows on the same
/// stream. Reuse it until attention returns and gather(pa, ga) overwrites it.
/// Pack only finished QR and KV; qa need not exist globally.
fn rowAttentionFront(e: *const Engine, ch: *Chunk, lay: weights.Layer) !void {
    const r = try row_state.rows(ch.n, e.comm.rank, e.world);
    const q = lay.wq_a.n;
    const hd = e.c.head_dim;
    const stride = (q + hd) * 2;
    const x = ch.x + r.first * e.c.hidden * 2;
    try mmRows(e, ch, r.count, lay.wq_a, x, e.c.hidden,
        ch.qa + r.first * q * 2, .bf16, q);
    try mmRows(e, ch, r.count, lay.wkv, x, e.c.hidden,
        ch.y + r.first * hd * 2, .bf16, hd);
    try tri_basic.rmsnorm(e.t, ch.qa + r.first * q * 2, q, lay.q_norm,
        ch.qr + r.first * q * 2, q, e.c.eps, r.count, q);
    const local = ch.ga + r.first * stride;
    try e.ops.copyRows(e.s, ch.qr + r.first * q * 2, q * 2,
        local, stride, q * 2, r.count);
    try e.ops.copyRows(e.s, ch.y + r.first * hd * 2, hd * 2,
        local + q * 2, stride, hd * 2, r.count);
    try rowGather(e, ch, ch.ga, stride, 0, ch.n);
    try e.ops.copyRows(e.s, ch.ga, stride, ch.qr, q * 2, q * 2, ch.n);
    try e.ops.copyRows(e.s, ch.ga + q * 2, stride, ch.y, hd * 2, hd * 2, ch.n);
    ch.row_stats.front_rows += r.count;
}
