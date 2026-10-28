//! The DSpark drafter of the served lane's concurrent decoder (dspark.py's Drafter, DraftPool and BatchDraftGraph as
//! multi.py runs them) on one TP rank: the streams' window rings in one pool, the target's taps absorbed into them
//! (absorb after a prompt chunk; absorb_many of every drafting stream's window rows after a round's forward, the served
//! TF_DS_EAGER_ABSORB), and N streams' drafts and confidences from one pass: BatchDraftGraph._body with
//! TF_DS_DRAFT_FUSED and the hc switch (the three stages, each sublayer's post fused into the next mixes; q's RMSNorm
//! writing wq_b's rotated rows and q's RoPE in wq_b's epilogue; the block rows' keys normed and roped, stored nowhere;
//! the sparse attention over the stream's 128 newest absorbed positions and its own block rows; wo_a's epilogue
//! rotating wo_b's rows; the MoE at top 3 of the routed experts and the shared one), the Markov loop (markov.py: the
//! vocabulary split over the ranks, the bias rows of 256 frequent tokens cached, each step's bests gathered) and the
//! confidence head (one fp32 GEMV as torch issues it). The served L2 prefetch forks move no bytes and are left out; the
//! RDMA gathers are NCCL all-gathers (the same bytes).
const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("prompt.zig");
const weights = @import("weights.zig");
const plan = @import("plan.zig");
const tri_basic = @import("tri_basic.zig");
const tri_hc = @import("tri_hc.zig");
const tri_norm = @import("tri_norm.zig");
const tri = @import("tri.zig");
const tri_attn = @import("tri_attn.zig");
const tri_markov = @import("tri_markov.zig");
const exl3_linear = @import("exl3_linear.zig");
const exl3_experts = @import("exl3_experts.zig");
const markov_tokens = @import("markov_tokens.zig");
const draft2d = @import("draft2d.zig");

const Engine = prompt.Engine;
const Chunk = prompt.Chunk;

/// MultiDecoder's stream slots in the served lane (--parallel 4): the pool's streams.
pub const max_streams = 16; // the most streams any stream setting (Model.streams) gives the pool
/// The served pool's streams when unset (--parallel 4); the four-node lane serves 16.
pub const default_streams = 4;
/// A pass's rows at most: every stream's block (four streams: 20). Past kernels.decode_rows() (16 rows: four streams) the MoE gate is a
/// cuBLAS matmul with the plain routing and the attention one split (its pick list padded to 16 columns of -1).
pub const max_rows = max_streams * 5; // every stream of the pool, a five-row block each
/// kernels.decode_rows(): the row-invariant decode kernels' rows at most.
/// sparse_attn's pick list of a single-split call: padded with -1 to a multiple of 16 entries.
const pick_pad = 16;
/// dspark.py: ring_size = window + 16.
pub const ring_extra = 16;
/// The stages (DSpark blocks) and a stream's block rows at most.
pub const max_stages = 4;
pub const max_block = 8;
/// absorb_many's rows at most: every drafting stream's verify window (16 rows a round).
pub const max_absorb = 48; // a round's drafting rows at most (round.max_rows)

/// DraftPool: one ring plane a stage, bf16 [slots * ring, head_dim]; slot s's rows [s * ring, (s + 1) * ring). Positions
/// below `absorbed[s]` are in slot s's rings.
pub const Pool = struct {
    rings: [max_stages]u64 = @splat(0),
    stages: usize,
    slots: usize,
    ring: usize,
    hd: usize,
    absorbed: [max_streams]usize = @splat(0),

    /// Zeros (torch.zeros), as the served pool starts.
    pub fn init(e: *const Engine, a: *prompt.Arena, slots: usize) !Pool {
        const ds = e.w.dspark orelse return error.NoDrafter;
        if (ds.blocks.len > max_stages or slots > max_streams) return error.DrafterTooLarge;
        var p: Pool = .{ .stages = ds.blocks.len, .slots = slots, .ring = e.c.window + ring_extra, .hd = e.c.head_dim };
        for (0..p.stages) |j| {
            p.rings[j] = try a.take(p.planeBytes());
            try prompt.fill(e, p.rings[j], 0, p.planeBytes());
        }
        return p;
    }

    pub fn planeBytes(p: *const Pool) usize {
        return p.slots * p.ring * p.hd * 2;
    }

    /// Slot s's rows of stage j's plane (DraftPool.views[s].rings[j]).
    pub fn view(p: *const Pool, j: usize, s: usize) u64 {
        return p.rings[j] + s * p.ring * p.hd * 2;
    }
};

/// The points of a pass a checker sees, in the order the served build's recorder takes them (stage j: layer
/// layers + j): each stage's attention input rows and its stage's ring plane, its partial and gather, the MoE's input
/// rows, partial and gather; the streams and pre after the stages; the head's logits (this rank's columns); the drafts.
pub const Point = enum { attn_in, attn_ring, attn_out, attn_gather, moe_in, moe_out, moe_gather, stages_h, stages_pre, local, markov };

/// A checker of a pass's buffers between its steps (synchronized first by the checker): false stops the pass.
pub const Probe = struct {
    ctx: *anyopaque,
    at: *const fn (ctx: *anyopaque, what: Point, stage: usize, dev: u64) anyerror!bool,

    fn check(pb: ?Probe, what: Point, stage: usize, dev: u64) !void {
        const p = pb orelse return;
        if (!try p.at(p.ctx, what, stage, dev)) return error.DraftMismatch;
    }
};

/// An absorb_many item: a drafting stream's slot, its window rows' taps (bf16 [n, taps * D], contiguous rows) and the
/// window's first position.
pub const Item = struct { slot: usize, taps: u64, n: usize, start: usize };

/// The per-step gather of the Markov loop's bests: [world, rows, 4] fp32 into dst.
const MarkovGather = struct {
    e: *const Engine,
    dst: u64,

    fn run(ctx: *anyopaque, send: u64, rows: usize) anyerror!u64 {
        const g: *MarkovGather = @ptrCast(@alignCast(ctx));
        try g.e.gatherF32(send, g.dst, rows * 4);
        return g.dst;
    }
};

/// The drafter pass's CUDA graphs (BatchDraftGraph), one a (streams, steps): every pointer it reads is a fixed
/// buffer and every per-pass value (ids, positions, slots, the pick lists) is uploaded ahead of the replay.
const storage_guard = @import("graph_storage.zig");

pub const PassGraphs = struct {
    const Entry = struct { exec: cuda.graph.Exec, storage: ?[]u64 = null };
    gpa: std.mem.Allocator,
    map: std.AutoHashMapUnmanaged(Key, Entry) = .empty,
    guard: bool = false,

    pub const Key = struct { streams: u32, steps: u32 };

    pub fn init(gpa: std.mem.Allocator) PassGraphs {
        return .{ .gpa = gpa, .guard = storage_guard.enabled() };
    }

    pub fn deinit(g: *PassGraphs) void {
        var it = g.map.valueIterator();
        while (it.next()) |x| { x.exec.deinit(); if (x.storage) |v| g.gpa.free(v); }
        g.map.deinit(g.gpa);
        g.* = undefined;
    }

    pub fn count(g: *const PassGraphs) usize {
        return g.map.count();
    }

    fn run(g: *PassGraphs, dr: *Drafter, e: *const Engine, ch: *const Chunk, pool: *const Pool, N: usize, steps: usize) !void {
        const key: Key = .{ .streams = @intCast(N), .steps = @intCast(steps) };
        var signature: storage_guard.Signature = .{};
        if (g.guard) {
            try signature.add(@intFromPtr(e));
            try signature.add(@intFromPtr(dr.ds.blocks.ptr));
            try signature.storage(dr.*, &.{});
            try signature.storage(dr.xsd, &.{}); try signature.storage(dr.xs, &.{});
            try signature.storage(dr.mk, &.{}); try signature.storage(dr.msc, &.{});
            try signature.words(&dr.gate_f); try signature.words(&dr.ak);
            try signature.words(&pool.rings);
            try signature.add(ch.invalid); try signature.add(ch.gxh); try signature.add(ch.gz);
        }
        if (g.map.getPtr(key)) |x| {
            if (g.guard) try signature.check(x.storage orelse return error.GraphStorageChanged);
            return x.exec.launchOn(e.s);
        }
        try cuda.graph.beginCapture(e.s, .thread_local);
        dr.body(e, ch, pool, N, steps, null) catch |err| {
            if (cuda.graph.endCapture(e.s)) |got| {
                var x = got;
                x.deinit();
            } else |_| {}
            return err;
        };
        var graph = try cuda.graph.endCapture(e.s);
        defer graph.deinit();
        var exec = try graph.instantiate();
        errdefer exec.deinit();
        const storage = if (g.guard) try g.gpa.dupe(u64, signature.slice()) else null;
        errdefer if (storage) |v| g.gpa.free(v);
        try g.map.put(g.gpa, key, .{ .exec = exec, .storage = storage });
        errdefer _ = g.map.remove(key);
        try exec.launchOn(e.s);
    }
};

pub const Drafter = struct {
    ds: weights.DSpark,
    n: usize, // a stream's block rows: [token, noise x (n - 1)]
    noise: i64,
    topk: usize,
    slots_moe: usize, // the MoE's slots a row: top-k routed and the shared expert
    taps_w: usize, // the taps' row width: taps * D
    // the pass's index rows (int64, max_rows a row: ids, positions, window positions, ring bases, zeros, -1), bidx
    // [max_rows, n] (each row's stream's block rows; a single-split pass's padded to 16 columns of -1), wlo int64 [1] =
    // 0 (Model._zero)
    rows: u64,
    bidx: u64,
    wlo: u64,
    // the streams and the mHC state
    h: u64,
    h_alt: u64,
    x: u64,
    part: u64,
    pre: u64,
    pre_a: u64,
    pre_f: u64,
    post: u64,
    comb: u64,
    // attention
    qa: u64,
    ykv: u64,
    qr: u64,
    xhq: u64, // fp16 [R, q_lora]: wq_b's rotated input rows (the RMSNorm writes them)
    q: u64,
    kvb: u64, // bf16 [R, head_dim]: the block rows' keys
    o: u64,
    pm_: u64,
    pl_: u64,
    po_: u64,
    u: u64,
    xb: u64, // fp16 [R, wo_b.k]: wo_b's rotated input rows (wo_a's epilogue writes them)
    pa: u64,
    ga: u64,
    // MoE
    gl: u64,
    xf: u64, // fp32 [R, D]: a wide pass's rows for the gate's cuBLAS matmul
    gate_f: [max_stages]u64, // fp32 [experts, D]: each stage's gate widened (gate_w.float())
    pick: u64,
    mw: u64,
    pm: u64,
    gm: u64,
    xsd: exl3_experts.DecodeScratch,
    xs: exl3_experts.Scratch, // a pass of EXACT_ROWS rows or more (13+ streams): the chunk path on a scratch of its own (d48)
    // the head
    xc: u64, // bf16 [R, D]: the collapsed streams (pre-norm: the confidence head reads them)
    xn: u64,
    local: u64, // fp32 [R, this rank's vocabulary columns]
    // the Markov loop
    mk: tri_markov.Markov,
    msc: tri_markov.Scratch,
    mg: u64, // fp32 [world, max_streams, 4]: a step's gathered bests
    out: u64, // int64 [max_streams, n + 1]: each stream's last token, then its drafts
    // the confidence head
    embs: u64, // bf16 [max_rows, rank]: the Markov embedding rows of the tokens each step read
    xcf: u64, // fp32 [max_rows, D]
    ef: u64, // fp32 [max_rows, rank]
    conf_in: u64, // fp32 [max_rows, D + rank]
    confw: u64, // fp32 [D + rank]: the head's weight widened (conf.float())
    conf: u64, // fp32 [max_rows]
    // absorbs: the taps rows (one item's or the items' cat), main_proj's rows, main_x, each stage's wkv rows, the
    // keys' normed rows (kv_norm_rope's out, not read), positions and ring rows
    at: u64,
    ay: u64,
    ax: u64,
    ak: [max_stages]u64,
    ao: u64,
    apos: u64,
    arows: u64,
    // the pass's graphs (BatchDraftGraph: one a streams and steps; null: eager)
    graphs: ?*PassGraphs = null,
    // the last pass on the host: each stream's drafts and confidences (`steps` of them)
    streams: usize = 0,
    steps: usize = 0,
    drafts: [max_streams][max_block]i64 = undefined,
    confs: [max_streams][max_block]f32 = undefined,

    /// Every buffer for passes of up to max_rows rows and absorbs of up to a ring of rows; the Markov loop's cached
    /// bias rows (Markov.fill, at load as served) for rank s.rank of s.world.
    pub fn init(e: *const Engine, a: *prompt.Arena, sp: plan.Split) !Drafter {
        const c = e.c;
        const ds = e.w.dspark orelse return error.NoDrafter;
        if (ds.blocks.len > max_stages or c.dspark_block > max_block or c.dspark_block == 0) return error.DrafterTooLarge;
        const d = c.hidden;
        const hc = c.hc;
        const hd = c.head_dim;
        const R = max_rows;
        const b0 = ds.blocks[0];
        const hl = b0.wq_b.n / hd; // the blocks' heads (TP2's on a 2D node too: TP2 inside the pair)
        var dr: Drafter = undefined;
        dr.ds = ds;
        dr.n = c.dspark_block;
        dr.noise = c.dspark_noise;
        dr.topk = c.draft_top_k;
        dr.slots_moe = c.draft_top_k + 1;
        dr.taps_w = c.dspark_taps.slice().len * d;
        dr.streams = 0;
        dr.steps = 0;
        dr.graphs = null;
        dr.rows = try a.take(6 * R * 8);
        dr.bidx = try a.take(R * pick_pad * 8);
        dr.wlo = try a.take(8);
        try prompt.fill(e, dr.wlo, 0, 8);
        dr.h = try a.take(R * hc * d * 2);
        dr.h_alt = try a.take(R * hc * d * 2);
        dr.x = try a.take(R * d * 2);
        dr.part = try a.take(R * tri_basic.hc_blocks * 32 * 4);
        dr.pre = try a.take(R * hc * 4);
        dr.pre_a = try a.take(R * hc * 4);
        dr.pre_f = try a.take(R * hc * 4);
        dr.post = try a.take(R * hc * 4);
        dr.comb = try a.take(R * hc * hc * 4);
        dr.qa = try a.take(R * b0.wq_a.n * 2);
        dr.ykv = try a.take(R * hd * 2);
        dr.qr = try a.take(R * b0.wq_a.n * 2);
        dr.xhq = try a.take(R * b0.wq_b.k * 2);
        dr.q = try a.take(R * hl * hd * 2);
        dr.kvb = try a.take(R * hd * 2);
        dr.o = try a.take(R * hl * hd * 2);
        dr.pm_ = try a.take(R * hl * tri_attn.attn_splits * 4);
        dr.pl_ = try a.take(R * hl * tri_attn.attn_splits * 4);
        dr.po_ = try a.take(R * hl * tri_attn.attn_splits * hd * 4);
        dr.u = try a.take(R * b0.groups * b0.wo_a[0].n * 2);
        dr.xb = try a.take(R * b0.wo_b.k * 2);
        dr.pa = try a.take(R * d * 4);
        dr.ga = try a.take(e.world * R * d * 4);
        const ne: usize = b0.experts.count - 1;
        dr.gl = try a.take(@max(16 * (d / 256), R) * ne * 4);
        dr.xf = try a.take(R * d * 4);
        dr.gate_f = @splat(0);
        for (ds.blocks, 0..) |lay, j| {
            dr.gate_f[j] = try a.take(ne * d * 4);
            try e.ops.f16ToF32(e.s, lay.gate_w, dr.gate_f[j], ne * d);
        }
        dr.pick = try a.take(R * dr.slots_moe * 4);
        dr.mw = try a.take(R * dr.slots_moe * 4);
        dr.pm = try a.take(R * d * 4);
        dr.gm = try a.take(e.world * R * d * 4);
        // Model._moe_scratch(("moe", top-k + 1, the stages' experts, decode)): one decode scratch of their own
        dr.xsd = try prompt.decodeScratch(e, a, b0.experts, dr.slots_moe);
        {
            const ex = b0.experts;
            const sz = exl3_experts.Scratch.sizes(R, dr.slots_moe, ex.dims, ex.width, ex.count);
            dr.xs.rows = R;
            dr.xs.slots = dr.slots_moe;
            inline for (.{ "xg", "xu", "xd", "z", "no_y", "ids", "count", "counts", "members", "work_gu", "work_d" }, 0..) |f, j| {
                @field(dr.xs, f) = try a.take(sz[j]);
            }
        }
        dr.xc = try a.take(R * d * 2);
        dr.xn = try a.take(R * d * 2);
        dr.local = try a.take(R * e.w.head.n * 4);
        // the Markov loop: every token's slot -1 (none), then the cached rows of the first 256 frequent tokens
        const none = try a.take(c.vocab * 4);
        try e.d.check(e.d.api.cuMemsetD32Async(none, 0xffffffff, c.vocab, e.s.handle), "cuMemsetD32Async");
        dr.mk = if (e.two) |tw| try draft2d.markov(c.*, tw, ds.markov_head, ds.markov_embed, none) else try tri_markov.Markov.init(c.*, sp, ds.markov_head, ds.markov_embed, none);
        const k = markov_tokens.cached.len;
        const tok = try a.take(k * 8);
        try prompt.upload(e, tok, &markov_tokens.cached, k * 8);
        const cache = try a.take(k * dr.mk.cols * 2);
        const slot = try a.take(c.vocab * 4);
        {
            const host = try std.heap.page_allocator.alloc(i32, c.vocab);
            defer std.heap.page_allocator.free(host);
            @memset(host, -1);
            for (markov_tokens.cached, 0..) |t, j| host[@intCast(t)] = @intCast(j);
            try prompt.upload(e, slot, host.ptr, c.vocab * 4);
            try e.s.synchronize(); // (the host table is freed next)
        }
        try dr.mk.fill(e.t, tok, k, cache, slot);
        dr.msc = .{
            .pv = try a.take(max_streams * dr.mk.n_part * 4),
            .pi = try a.take(max_streams * dr.mk.n_part * 4),
            .send = try a.take(max_streams * 4 * 4),
            .stage = try a.take(max_streams * dr.mk.cols * 2),
        };
        try prompt.fill(e, dr.msc.send, 0, max_streams * 4 * 4);
        dr.mg = try a.take(dr.mk.world * max_streams * 4 * 4);
        dr.out = try a.take(max_streams * (max_block + 1) * 8);
        // the confidence head
        const rank = c.markov_rank;
        dr.embs = try a.take(R * rank * 2);
        dr.xcf = try a.take(R * d * 4);
        dr.ef = try a.take(R * rank * 4);
        dr.conf_in = try a.take(R * (d + rank) * 4);
        dr.confw = try a.take((d + rank) * 4);
        try e.ops.f16ToF32(e.s, ds.conf, dr.confw, d + rank);
        dr.conf = try a.take(R * 4);
        // absorbs
        const ring = c.window + ring_extra;
        dr.at = try a.take(ring * dr.taps_w * 2);
        dr.ay = try a.take(ring * d * 2);
        dr.ax = try a.take(ring * d * 2);
        dr.ak = @splat(0);
        for (0..ds.blocks.len) |j| dr.ak[j] = try a.take(ring * hd * 2);
        dr.ao = try a.take(ring * hd * 2);
        dr.apos = try a.take(ring * 8);
        dr.arows = try a.take(ring * 8);
        return dr;
    }

    fn row(dr: *const Drafter, i: usize) u64 {
        return dr.rows + i * max_rows * 8;
    }

    /// Drafter.absorb into slot `slot`'s view: the target's taps rows of positions start .. start + n - 1 (bf16
    /// [n, taps * D], contiguous) into every stage's ring, the last ring rows of them: main_x = RMSNorm(main_proj @
    /// taps), each stage's wkv of it normed and roped into its ring row (position % ring).
    pub fn absorb(dr: *Drafter, e: *const Engine, ch: *const Chunk, pool: *Pool, slot: usize, taps: u64, n: usize, start: usize) !void {
        if (n == 0 or slot >= pool.slots) return error.BadAbsorb;
        const keep = @min(n, pool.ring);
        if (keep > 128) return notPorted(@src()); // main_proj's prompt GEMM (the served lane's replay keeps 128 rows)
        const first = start + n - keep;
        var pos: [max_absorb * 9]i64 = undefined;
        var rows: [max_absorb * 9]i64 = undefined;
        for (0..keep) |r| {
            pos[r] = @intCast(first + r);
            rows[r] = @intCast((first + r) % pool.ring);
        }
        try prompt.upload(e, dr.apos, &pos, keep * 8);
        try prompt.upload(e, dr.arows, &rows, keep * 8);
        try dr.mainKeys(e, ch, taps + (n - keep) * dr.taps_w * 2, keep);
        for (dr.ds.blocks, 0..) |lay, j| {
            try tri_norm.kvNormRope(e.t, dr.ak[j], lay.kv_norm, e.plain.cos, e.plain.sin, dr.apos, pool.view(j, slot), pool.ring, dr.arows, e.c.eps, true, e.c.rope_dim, dr.ao, keep, e.c.head_dim);
        }
        pool.absorbed[slot] = start + n;
    }

    /// Drafter.absorb_many: every item's rows in one pass over the pool's planes (ring row slot * ring + position %
    /// ring); the caller sets each slot's `absorbed` (eager: the kept length, known after sampling).
    pub fn absorbMany(dr: *Drafter, e: *const Engine, ch: *const Chunk, pool: *const Pool, items: []const Item) !void {
        var n: usize = 0;
        var pos: [max_absorb]i64 = undefined;
        var rows: [max_absorb]i64 = undefined;
        for (items) |it| {
            if (it.slot >= pool.slots or n + it.n > max_absorb) return error.BadAbsorb;
            try e.ops.copyRows(e.s, it.taps, dr.taps_w * 2, dr.at + n * dr.taps_w * 2, dr.taps_w * 2, dr.taps_w * 2, it.n);
            for (0..it.n) |r| {
                const p = it.start + r;
                pos[n + r] = @intCast(p);
                rows[n + r] = @intCast(it.slot * pool.ring + p % pool.ring);
            }
            n += it.n;
        }
        if (n == 0) return;
        try prompt.upload(e, dr.apos, &pos, n * 8);
        try prompt.upload(e, dr.arows, &rows, n * 8);
        try dr.mainKeys(e, ch, dr.at, n);
        for (dr.ds.blocks, 0..) |lay, j| {
            try tri_norm.kvNormRope(e.t, dr.ak[j], lay.kv_norm, e.plain.cos, e.plain.sin, dr.apos, pool.rings[j], pool.slots * pool.ring, dr.arows, e.c.eps, true, e.c.rope_dim, dr.ao, n, e.c.head_dim);
        }
    }

    /// main_x = RMSNorm(main_proj @ taps) of n rows, then every stage's wkv of it (stage_kv: one group).
    fn mainKeys(dr: *Drafter, e: *const Engine, ch: *const Chunk, taps: u64, n: usize) !void {
        const c = e.c;
        const d = c.hidden;
        const hd = c.head_dim;
        try prompt.grouped(e, ch, n, &.{dr.ds.main_proj}, &.{taps}, &.{dr.taps_w}, &.{dr.ay}, &.{d}, &.{.bf16});
        try tri_basic.rmsnorm(e.t, dr.ay, d, dr.ds.main_norm, dr.ax, d, c.eps, n, d);
        const S = dr.ds.blocks.len;
        var ls: [max_stages]weights.Linear = undefined;
        var xs: [max_stages]u64 = undefined;
        var ldxs: [max_stages]usize = undefined;
        var ldos: [max_stages]usize = undefined;
        var types: [max_stages]exl3_linear.DType = undefined;
        for (dr.ds.blocks, 0..) |lay, j| {
            ls[j] = lay.wkv;
            xs[j] = dr.ax;
            ldxs[j] = d;
            ldos[j] = hd;
            types[j] = .bf16;
        }
        try prompt.grouped(e, ch, n, ls[0..S], xs[0..S], ldxs[0..S], dr.ak[0..S], ldos[0..S], types[0..S]);
    }

    /// BatchDraftGraph.run for N streams (tokens at positions q0 in slots): each stream's `steps` drafts and their
    /// confidences into drafts / confs. Each slot's rings hold its positions before q0 (absorbed).
    pub fn pass(dr: *Drafter, e: *const Engine, ch: *const Chunk, pool: *const Pool, tokens: []const i64, q0: []const i64, slots: []const i64, steps: usize, probe: ?Probe) !void {
        const N = tokens.len;
        const n = dr.n;
        const R = N * n;
        if (N == 0 or N > max_streams or R > max_rows or q0.len != N or slots.len != N) return notPorted(@src());
        if (steps == 0 or steps > n) return error.BadDraftPass;
        // _body's index rows: ids [token, noise ...], positions q0 + j, window positions q0 - 1, ring bases
        var hv: [6 * max_rows]i64 = @splat(0);
        var hb: [max_rows * pick_pad]i64 = @splat(-1);
        const bw: usize = if (R > tri.decode_rows) pick_pad else n; // the pick list's columns
        for (0..N) |i| {
            if (slots[i] < 0 or slots[i] >= pool.slots) return error.BadSlot;
            for (0..n) |j| {
                const r = i * n + j;
                hv[0 * max_rows + r] = if (j == 0) tokens[i] else dr.noise;
                hv[1 * max_rows + r] = q0[i] + @as(i64, @intCast(j));
                hv[2 * max_rows + r] = q0[i] - 1;
                hv[3 * max_rows + r] = slots[i] * @as(i64, @intCast(pool.ring));
                hv[4 * max_rows + r] = 0;
                hv[5 * max_rows + r] = -1;
                for (0..n) |k| hb[r * bw + k] = @intCast(i * n + k);
            }
        }
        try prompt.upload(e, dr.rows, &hv, hv.len * 8);
        try prompt.upload(e, dr.bidx, &hb, R * bw * 8);
        // the Markov loop's out [N, steps + 1]: each stream's last token first
        const ts = steps + 1;
        var ho: [max_streams * (max_block + 1)]i64 = @splat(0);
        for (0..N) |i| ho[i * ts] = tokens[i];
        try prompt.upload(e, dr.out, &ho, N * ts * 8);
        if (dr.graphs != null and probe == null) try dr.graphs.?.run(dr, e, ch, pool, N, steps) else try dr.body(e, ch, pool, N, steps, probe);
        // one host read: the drafts and the confidences
        try e.s.synchronize();
        var hc: [max_streams * max_block]f32 = undefined;
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(&ho, dr.out, N * ts * 8), "cuMemcpyDtoH");
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(&hc, dr.conf, N * steps * 4), "cuMemcpyDtoH");
        for (0..N) |i| for (0..steps) |j| {
            dr.drafts[i][j] = ho[i * ts + 1 + j];
            dr.confs[i][j] = hc[i * steps + j];
        };
        dr.streams = N;
        dr.steps = steps;
    }

    /// The pass's device work for N streams of `steps` drafts, from the rows' embeddings to the confidence head (its
    /// inputs uploaded: the index rows, the pick lists, the Markov loop's first column): what a pass graph captures.
    fn body(dr: *Drafter, e: *const Engine, ch: *const Chunk, pool: *const Pool, N: usize, steps: usize, probe: ?Probe) !void {
        const c = e.c;
        const t = e.t;
        const n = dr.n;
        const R = N * n;
        const d = c.hidden;
        // the streams: every row's token embedding in each stream, pre (1, 0, 0, 0)
        try tri_basic.embedInit(t, e.w.embed, dr.row(0), dr.h, dr.pre, R, d, c.hc);
        // _stages_fused
        var h = dr.h;
        var spare = dr.h_alt;
        var pre = dr.pre;
        var pre_f = dr.pre_f;
        var pending: ?u64 = null;
        for (dr.ds.blocks, 0..) |lay, j| {
            h = try dr.mix(e, h, &spare, &pending, lay.hc_attn, pre, lay.attn_norm, dr.pre_a, R);
            try Probe.check(probe, .attn_in, j, dr.x);
            try Probe.check(probe, .attn_ring, j, pool.rings[j]);
            try dr.attention(e, ch, pool, j, R);
            try Probe.check(probe, .attn_out, j, dr.pa);
            if (e.two) |tw| try tw.pairGather(e, dr.pa, dr.ga, R, d, 4) else try e.gatherF32(dr.pa, dr.ga, R * d);
            try Probe.check(probe, .attn_gather, j, dr.ga);
            pending = dr.ga;
            h = try dr.mix(e, h, &spare, &pending, lay.hc_ffn, dr.pre_a, lay.ffn_norm, pre_f, R);
            try Probe.check(probe, .moe_in, j, dr.x);
            try dr.moe(e, j, R);
            try Probe.check(probe, .moe_out, j, dr.pm);
            if (e.two) |tw| try tw.pairGather(e, dr.pm, dr.gm, R, d, 4) else try e.gatherF32(dr.pm, dr.gm, R * d);
            try Probe.check(probe, .moe_gather, j, dr.gm);
            pending = dr.gm;
            std.mem.swap(u64, &pre, &pre_f);
        }
        try tri_basic.hcPost(t, dr.gm, h, dr.post, dr.comb, h, e.world, R, d);
        const S = dr.ds.blocks.len;
        try Probe.check(probe, .stages_h, S, h);
        try Probe.check(probe, .stages_pre, S, pre);
        // the head on every row (this rank's columns: the Markov loop scores them without a gather)
        try tri_basic.collapse(t, h, pre, dr.xc, R, d);
        try tri_basic.rmsnorm(t, dr.xc, d, dr.ds.norm, dr.xn, d, c.eps, R, d);
        try prompt.grouped(e, ch, R, &.{e.w.head}, &.{dr.xn}, &.{d}, &.{dr.local}, &.{e.w.head.n}, &.{.f32});
        // Padded tokens must never win the Markov draft, including when every
        // valid logit is negative. Zero padded weights alone are insufficient.
        if (c.logical_vocab != 0) {
            const width: usize = e.w.head.n;
            const start = @as(usize, e.comm.rank) * width;
            const valid = @min(width, c.tokenVocab() -| start);
            if (valid < width) for (0..R) |r| {
                try e.d.check(e.d.api.cuMemsetD32Async(dr.local + (r * width + valid) * 4,
                    0xff800000, width - valid, e.s.handle), "TP3 draft vocab mask");
            };
        }
        try Probe.check(probe, .local, S, dr.local);
        // the Markov loop: out [N, steps + 1], each stream's last token first
        const ts = steps + 1;
        var g: MarkovGather = .{ .e = e, .dst = dr.mg };
        try dr.mk.steps(t, dr.local, dr.out, ts, N, n, steps, dr.msc, .{ .ctx = &g, .run = MarkovGather.run });
        try Probe.check(probe, .markov, S, dr.out);
        // the confidence head: [xc (fp32), the embedding rows the steps read (fp32)] @ conf^T, a stream's first steps rows
        const rank = c.markov_rank;
        const w = d + rank;
        try e.d.check(e.d.api.cuMemsetD8Async(ch.invalid, 0, 4, e.s.handle), "cuMemsetD8Async");
        for (0..N) |i| {
            try e.ops.gatherRows(e.s, dr.ds.markov_embed, c.vocab, dr.out + i * ts * 8, dr.embs + i * steps * rank * 2, rank * 2, steps, ch.invalid);
            try e.ops.toF32(e.s, dr.xc + i * n * d * 2, dr.xcf + i * steps * d * 4, steps * d);
        }
        try e.ops.toF32(e.s, dr.embs, dr.ef, N * steps * rank);
        try e.ops.copyRows(e.s, dr.xcf, d * 4, dr.conf_in, w * 4, d * 4, N * steps);
        try e.ops.copyRows(e.s, dr.ef, rank * 4, dr.conf_in + d * 4, w * 4, rank * 4, N * steps);
        try e.blas.xv(dr.conf_in, dr.confw, dr.conf, N * steps, w);
    }

    /// pre_mix of _stages_fused: hc_pre2 of the streams h with the pending gathered partials posted in (into the spare
    /// buffer, which becomes the streams; no side stream in the drafter); returns the streams.
    fn mix(dr: *Drafter, e: *const Engine, h: u64, spare: *u64, pending: *?u64, params: [3]u64, pre_in: u64, norm: u64, pre_out: u64, R: usize) !u64 {
        const c = e.c;
        if (pending.*) |gp| {
            const out = try tri_hc.hcPre2(e.t, h, params[0], params[1], params[2], pre_in, norm, c.eps, c.hc_eps, c.hc_iters, dr.x, pre_out, dr.post, dr.comb, dr.part, .{ .gathered = gp, .world = e.world, .h_out = spare.* }, null, R, c.hidden);
            pending.* = null;
            spare.* = h;
            return out;
        }
        return tri_hc.hcPre2(e.t, h, params[0], params[1], params[2], pre_in, norm, c.eps, c.hc_eps, c.hc_iters, dr.x, pre_out, dr.post, dr.comb, dr.part, null, null, R, c.hidden);
    }

    /// BatchDraftGraph._attention (fused): stage j's attention partial pa [R, D] fp32 from its mixed rows x.
    fn attention(dr: *Drafter, e: *const Engine, ch: *const Chunk, pool: *const Pool, j: usize, R: usize) !void {
        const c = e.c;
        const t = e.t;
        const lay = dr.ds.blocks[j];
        const hd = c.head_dim;
        const rd = c.rope_dim;
        const hl = lay.wq_b.n / hd; // the block's heads (TP2's on a 2D node too)
        const rope = e.plain; // a window-only stage (Model._cs: ratio 0)
        const pos = dr.row(1);
        // attn_in(comp=False): wq_a and wkv, one group
        try prompt.grouped(e, ch, R, &.{ lay.wq_a, lay.wkv }, &.{ dr.x, dr.x }, &.{ c.hidden, c.hidden }, &.{ dr.qa, dr.ykv }, &.{ lay.wq_a.n, hd }, &.{ .bf16, .bf16 });
        // q_proj with rope (rot_q): q's RMSNorm writes wq_b's rotated rows; wq_b's epilogue applies q's RoPE
        const rot = [_]tri_norm.Rot{.{ .suh = lay.wq_b.suh, .h = dr.xhq }};
        try tri_norm.rmsnormRot(t, dr.qa, lay.wq_a.n, lay.q_norm, c.eps, &rot, dr.qr, lay.wq_a.n, R, lay.wq_a.n);
        var qc = [_]exl3_linear.Call{.{ .layer = lay.wq_b, .x = 0, .ldx = 0, .x_dtype = .bf16, .xh = dr.xhq, .y = dr.q, .ldy = @intCast(lay.wq_b.n), .y_dtype = .bf16, .counters = 0, .rope = .{ .cos = rope.cos, .sin = rope.sin, .pos = pos, .hd = @intCast(hd), .rd = @intCast(rd) } }};
        try prompt.groupedRotated(e, ch, R, &qc);
        // the block rows' keys: normed and roped, stored nowhere (ring row -1)
        const plane_rows = pool.slots * pool.ring;
        try tri_norm.kvNormRope(t, dr.ykv, lay.kv_norm, rope.cos, rope.sin, pos, pool.rings[j], plane_rows, dr.row(5), c.eps, true, rd, dr.kvb, R, hd);
        // each row sees its stream's 128 newest absorbed positions (to q0 - 1) and every block row of its stream
        try tri_attn.sparseAttn(t, .{
            .q = dr.q,
            .out = dr.o,
            .rows = R,
            .h = hl,
            .hd = hd,
            .sink = lay.sink,
            .wsrc = pool.rings[j],
            .wsrc_rows = plane_rows,
            .wlo = dr.wlo,
            .ring = true,
            .comp = .{ .bf16 = dr.kvb },
            .idx = dr.bidx,
            .n_idx = dr.n,
            .pos = dr.row(2),
            .scale = prompt.scale(hd),
            .window = c.window,
            .wbase = dr.row(3),
            .cbase = dr.row(4),
            .ring_rows = pool.ring,
            .parts = .{ .pm = dr.pm_, .pl = dr.pl_, .po = dr.po_ },
        });
        try tri_norm.ropeHeads(t, dr.o, rope.cos, rope.sin, pos, rd, true, R, hl, hd);
        // wo_ab: wo_a's inputs rotated from o's column blocks, its epilogue writing wo_b's rotated rows; wo_b (fp32)
        var calls: [exl3_linear.gmax]exl3_linear.Call = undefined;
        const uw = lay.groups * lay.wo_a[0].n;
        const gw = hl * hd / lay.groups;
        var col: usize = 0;
        var off: usize = 0;
        for (lay.wo_a[0..lay.groups], 0..) |wo, gi| {
            calls[gi] = .{ .layer = wo, .x = dr.o + gi * gw * 2, .ldx = @intCast(hl * hd), .x_dtype = .bf16, .xh = ch.gxh + off * 2, .y = dr.u + col * 2, .ldy = @intCast(uw), .y_dtype = .bf16, .counters = try prompt.counterOf(ch, wo), .rot = .{ .suh = lay.wo_b.suh, .xh = dr.xb, .ldr = @intCast(lay.wo_b.k), .off = @intCast(col) } };
            col += wo.n;
            off += R * wo.k;
        }
        try exl3_linear.rotMany(e.lin, e.s, calls[0..lay.groups], R, true);
        try exl3_linear.glinear(e.lin, e.s, calls[0..lay.groups], R, ch.gz, true, true);
        var cb = [_]exl3_linear.Call{.{ .layer = lay.wo_b, .x = 0, .ldx = 0, .x_dtype = .bf16, .xh = dr.xb, .y = dr.pa, .ldy = @intCast(lay.wo_b.n), .y_dtype = .f32, .counters = 0 }};
        try prompt.groupedRotated(e, ch, R, &cb);
    }

    /// Model.moe(lay, x, topk) of a pass's rows: the gate's chunk sums (rowmm_gate; past decode_rows rows the gate's fp32
    /// matmul), the routing at the stages' top-k with the shared expert in every row's last slot, the routed experts'
    /// decode path on the stages' own scratch.
    fn moe(dr: *Drafter, e: *const Engine, j: usize, R: usize) !void {
        const c = e.c;
        const lay = dr.ds.blocks[j];
        const ne: usize = lay.experts.count - 1;
        var kc: usize = 0;
        if (R <= tri.decode_rows) {
            kc = try tri_norm.rowmmGate(e.t, dr.x, c.hidden, lay.gate_w, dr.gl, R, c.hidden, ne);
        } else {
            // x.float() @ gate_w.float().t() through cuBLAS, then the plain routing
            try e.ops.toF32(e.s, dr.x, dr.xf, R * c.hidden);
            try e.blas.xwT(dr.xf, dr.gate_f[j], dr.gl, R, c.hidden, ne);
        }
        try tri_norm.route(e.t, dr.gl, kc, lay.gate_b, dr.topk, c.routed_scaling, lay.experts.count - 1, dr.pick, dr.mw, R, ne, dr.slots_moe);
        // routed(): fewer than EXACT_ROWS rows take the decode window's fused path, more the chunk's (13+ streams)
        if (R < exl3_experts.exact_rows) return exl3_experts.decode(e.ex, e.s, lay.experts, dr.xsd, dr.x, c.hidden, dr.pick, dr.mw, dr.pm, R, c.swiglu_limit);
        try exl3_experts.prompt(e.ex, e.s, lay.experts, dr.xs, dr.x, c.hidden, dr.pick, dr.mw, dr.pm, R, c.swiglu_limit);
    }
};

/// MultiDecoder._depth: the most drafts a stream verifies in a round of `live` streams (TF_DS_PARALLEL_DEPTH unset:
/// 5, 5, 3, 3), at most the engine's drafts and the rows a stream may take of a round (max_rows / live - 1).
pub fn depth(live: usize, drafts: usize, round_rows: usize) usize {
    const by = [_]usize{ 5, 5, 3, 3 };
    const k = by[@min(live, by.len) - 1];
    return @min(@min(k, drafts), (round_rows / @max(live, 1)) -| 1);
}

/// The batched pass's Markov steps for N drafting streams (BatchDraftGraph's `steps` under the served "conf" policy):
/// as deep as any round with N or more live streams may verify.
pub fn passSteps(n_streams: usize, slots: usize, drafts: usize, round_rows: usize, block: usize) usize {
    var s: usize = 0;
    var live = n_streams;
    while (live <= slots) : (live += 1) s = @max(s, depth(live, drafts, round_rows));
    return @max(1, @min(block, s));
}

/// The served ROUND_MS table (TF_DS_ROUND_MS unset): a round's forward ms by its rows, and DRAFT_MS.
pub const round_ms = [_]f64{ 30.0, 35.2, 39.9, 44.9, 48.6, 52.1, 55.6, 59.1, 62.7, 66.3, 69.6, 72.9, 75.5, 78.2, 81.3, 84.4, 87.5, 90.6, 93.7, 96.8, 99.9, 103.0, 106.1, 109.2 };
pub const draft_ms: f64 = 3.5;

extern "c" fn exp(x: f64) f64;

/// MultiDecoder._choose_k: the k in 0 .. kmax with the most expected tokens a millisecond (1 + the prefix survivals'
/// sum, survival the sigmoid of each confidence, over the round's table cost with the other streams at their table
/// depth), in Python's float64 arithmetic (libm's exp, as Python's math.exp calls it); the first best wins.
pub fn chooseK(conf: []const f32, kmax: usize, live: usize, drafts: usize, round_rows: usize) usize {
    const others = (live - 1) * (depth(live, drafts, round_rows) + 1);
    var best: usize = 0;
    var best_v: f64 = -1.0;
    var surv: f64 = 1.0;
    var exp_tokens: f64 = 1.0;
    for (0..kmax + 1) |kk| {
        if (kk > 0) {
            surv *= 1.0 / (1.0 + exp(-@as(f64, conf[kk - 1])));
            exp_tokens += surv;
        }
        const rows = @min(round_ms.len, others + kk + 1);
        const v = exp_tokens * @as(f64, @floatFromInt(live)) / (round_ms[rows - 1] + draft_ms);
        if (v > best_v) {
            best = kk;
            best_v = v;
        }
    }
    return best;
}

test "draft depths and pass steps as multi.py takes them" {
    // 16-row rounds, 5 drafts: 5 alone or beside one other stream, 3 at three or four
    try std.testing.expectEqual(@as(usize, 5), depth(1, 5, 16));
    try std.testing.expectEqual(@as(usize, 5), depth(2, 5, 16));
    try std.testing.expectEqual(@as(usize, 3), depth(3, 5, 16));
    try std.testing.expectEqual(@as(usize, 3), depth(4, 5, 16));
    try std.testing.expectEqual(@as(usize, 5), passSteps(1, 4, 5, 16, 5));
    try std.testing.expectEqual(@as(usize, 5), passSteps(2, 4, 5, 16, 5));
    try std.testing.expectEqual(@as(usize, 3), passSteps(3, 4, 5, 16, 5));
    try std.testing.expectEqual(@as(usize, 3), passSteps(4, 4, 5, 16, 5));
}

/// error.NotPortedYet with its site on stderr (a request that reaches a path the port lacks names it in the log).
fn notPorted(src: std.builtin.SourceLocation) error{NotPortedYet} {
    std.debug.print("{{\"not_ported\": \"{s}:{d}\"}}\n", .{ src.file, src.line });
    return error.NotPortedYet;
}
