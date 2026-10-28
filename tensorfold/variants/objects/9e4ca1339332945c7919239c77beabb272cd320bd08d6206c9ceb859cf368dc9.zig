//! A decode round on one TP rank: the served build's RoundDecoder (rounds.py) for the rows of one or more streams' windows,
//! with the served switches (hc fused with the Sinkhorn half deferred and the posted mixes' dots on it; glue, rot_q,
//! rot_attn, rot_wob, idx, comp, rowmm on; hc_rot and SHARED_OVERLAP off): the rows' streams from their token ids
//! (embed_init), then a stretch at a time (cut before each Engram layer) every layer's attention mixes with the last
//! sublayer's post fused in, its DSpark tap, attention (_attention2 and _kv_idx: the q norm with the window KV into the
//! ring and the rotations folded in, q's RoPE in wq_b's epilogue, the compressor and the indexer, the sparse attention
//! whose merge applies the inverse RoPE and rotates wo_a's inputs, wo_a's epilogue rotating wo_b's), its gather, the
//! FFN mixes, the MoE and its gather; a stretch's last post alone; the head on every row. The served side streams (the
//! Sinkhorn half, the L2 prefetches) run in this stream's order here, the RDMA gathers are NCCL all-gathers (the same
//! bytes), and the round's index arithmetic (rounds.py _ix: integer ops on the positions and the stream's extent) is
//! made on the host. A node of the four-node 2D split takes its column parts and their exchanges through round2d.zig's
//! hooks (Engine.two).
const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("prompt.zig");
const weights = @import("weights.zig");
const tri = @import("tri.zig");
const tri_basic = @import("tri_basic.zig");
const tri_hc = @import("tri_hc.zig");
const tri_norm = @import("tri_norm.zig");
const tri_attn = @import("tri_attn.zig");
const tri_index = @import("tri_index.zig");
const exl3_linear = @import("exl3_linear.zig");
const exl3_experts = @import("exl3_experts.zig");
const round2d = @import("round2d.zig");
const exl3_experts2d = @import("exl3_experts2d.zig");
const engram = @import("engram.zig");
const exact = @import("exact.zig");
const ops_mod = @import("ops.zig");

const Engine = prompt.Engine;
const Chunk = prompt.Chunk;
const Caches = prompt.Caches;

/// kernels.ROUND_ROWS (TF_DS_ROUND_ROWS unset: TF_DS_DECODE_ROWS, 16): a round's rows at most.
pub const max_rows = 48; // the most rows any round-row setting (Engine.round_rows) gives a round
/// The served round-row bound when TF_DS_ROUND_ROWS is unset (kernels.DECODE_ROWS); the four-node lane serves 48.
pub const default_rows = 16;
/// graph.py BUCKET_MIN (TF_DS_BUCKET_MIN unset).
pub const bucket_min = 1024;
/// The window ring's rows a slot (model.py: window + RING_EXTRA).
const ring_extra = prompt.ring_extra;
/// The compressor's positional store's rows a slot (model.py RAW).
const raw_rows = prompt.raw_rows;

/// graph.py bucket_for: the context bucket of a round whose deepest row sees n positions, in a pool of `cap`.
pub fn bucketFor(n: usize, cap: usize) usize {
    var b: usize = bucket_min;
    while (b < n) b *= 2;
    return @min(b, @max(cap, bucket_min));
}

/// The points a round exchanges or ends with, in the order the served build's recorder sees them (a checker's hook).
pub const Point = enum { engram_proj, engram_gather, attn_in, attn_out, attn_gather, moe_in, moe_out, moe_gather, head_cols, head_gather, logits, taps };

/// A checker of a round's buffers between its steps (synchronized first by the checker): false stops the round.
pub const Probe = struct {
    ctx: *anyopaque,
    at: *const fn (ctx: *anyopaque, what: Point, layer: usize, dev: u64) anyerror!bool,

    fn check(pb: ?Probe, what: Point, layer: usize, dev: u64) !void {
        const p = pb orelse return;
        if (!try p.at(p.ctx, what, layer, dev)) return error.RoundMismatch;
    }
};

/// A round's rows: token ids at positions pos. One stream's (consecutive positions) in pool slot 0 with its extent [base,
/// end) of positions; or a concurrent round's (MultiDecoder: several streams' windows, each window's rows contiguous):
/// each row's pool slot and its stream's extent, and each window's first row with its stream's ids through the window.
pub const Rows = struct {
    ids: []const i64,
    pos: []const i64,
    base: i64 = 0,
    end: i64 = 0,
    slots: ?[]const i64 = null,
    bases: ?[]const i64 = null,
    ends: ?[]const i64 = null,
    windows: ?[]const Window = null,

    fn slotOf(r: Rows, i: usize) i64 {
        return if (r.slots) |v| v[i] else 0;
    }
    fn baseOf(r: Rows, i: usize) i64 {
        return if (r.bases) |v| v[i] else r.base;
    }
    fn endOf(r: Rows, i: usize) i64 {
        return if (r.ends) |v| v[i] else r.end;
    }
};

/// A window of a concurrent round: its first row, its rows, its stream's ids through the window (Engram's n-grams).
pub const Window = struct { row: usize, n: usize, seq: []const i32 };

/// The index tensors rounds.py's _ix makes from a round's inputs, by row (int64 [R] each, gi [R, 2]).
const Glue = enum(u8) { wslot, wbase, cbase1, ctarget1, gpos1, vis1, cbase2, ctarget2, gpos2, vis2, rslot, gi };
const glue_n = @as(usize, @backingInt(Glue.gi)) + 2; // gi (the last) takes two

/// A round's buffers for up to max_rows rows (Python allocates them a round; their contents never carry over).
pub const Round = struct {
    r: usize = 0,
    bucket: usize = 0,
    glue: u64, // int64 [glue_n, max_rows]
    // RoundDecoder.inputs: one int64 [5, R] buffer (ids, positions, slots, the extent's base and end), so a round's
    // positions sit R * 8 bytes in: the alignment its kernels were specialized for (16-byte at even R only)
    inputs: u64,
    ids: u64 = 0,
    pos: u64 = 0,
    zero: u64, // int64 [1] = 0: m._zero, sparse_attn's WLO in ring mode
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
    qa: u64, // bf16 [R, q_lora]
    ykv: u64, // bf16 [R, head_dim]
    ckv: u64, // the compressor's kv projection: bf16 (ratio 1) or fp32 [R, head_dim]
    cgate: u64, // fp32 [R, head_dim]
    qr: u64, // bf16 [R, q_lora]
    kout: u64, // bf16 [R, head_dim]: q_kv_norm's KV rows
    xhq: u64, // fp16 [wq_b, idx_wq_b][R, q_lora]: their rotated input rows (q_proj's group buffers)
    q: u64, // bf16 [R, heads, head_dim]
    iq: u64, // bf16 [R, index_heads, index_head_dim]
    iq4: u64, // its fp4_qd bytes
    iw: u64, // bf16 [R, index_heads]
    keys: u64, // int64 [R, nb]
    score: u64, // fp32 [R, nb]
    cand: u64, // u8 [R, nb / candidate_block]
    has_cand: bool = false,
    // past candidate_blocks blocks (TF_DS_CAND_ONLY, served on): the candidate source's block maxima, their top blocks
    // and the pool as block ids (int32 [R, candidate_blocks], -1 none) the later layers score alone
    bmax: u64, // fp32 [R, nb / candidate_block]
    pidx: u64, // int64 [R, candidate_blocks]
    cblk: u64, // int32 [R, candidate_blocks]
    has_cblk: bool = false,
    cidx: u64, // int64 [R, index_topk]
    kk: usize = 0,
    ktop: u64, // int64 [R, index_topk]: torch's top-k values
    tmax: u64, // int64 [R, nb / 64]: each 64-key tile's largest key (past TOPK_FUSED_MAX keys)
    pbufs: tri_index.PrunedBufs, // topk_select_pruned's tpos [R, k], cand [R, k * 64] and every [R]
    kvg: u64, // fp32 [R, 2, head_dim]: the group's positional-store rows
    sg: u64,
    lat2: u64, // bf16 [R, head_dim]: the pair's weighted sum
    lat: u64, // bf16 [R, head_dim]: the compressed latent
    ikp: u64, // bf16 [R, index_head_dim]: idx_wk's projection
    ik: u64, // bf16 [R, index_head_dim]: the index key
    o: u64, // bf16 [R, heads, head_dim]
    pm_: u64, // fp32 [R, heads, 8]: the split keys' partials
    pl_: u64,
    po_: u64, // fp32 [R, heads, 8, head_dim]
    xhwo: u64, // fp16 [wo_a group][R, K]: wo_a's rotated input rows (the merge writes them)
    xb: u64, // fp16 [R, wo_b.k]: wo_b's rotated input rows (wo_a's epilogue writes them)
    u: u64, // bf16 [R, groups * o_lora]
    pa: u64, // fp32 [R, D]
    ga: u64, // fp32 [world, R, D]
    // MoE
    gl: u64, // fp32 [R, D / 256, experts]: the gate's chunk sums
    pick: u64, // int32 [R, slots]
    mw: u64, // fp32 [R, slots]
    pm: u64, // fp32 [R, D]
    gm: u64, // fp32 [world, R, D]
    // Engram
    e_in: u64, // bf16 [R, cols * engram_head_dim]: the rows read for this rank's hash columns
    ek: u64, // fp32 [R, engram_wkv.n]
    ekg: u64, // fp32 [world, R, engram_wkv.n]
    ekv: u64, // bf16 [R, engram_wkv.n]
    // the head and the taps
    xc: u64, // bf16 [R, D]
    hl: u64, // fp32 [R, columns]
    hg: u64, // fp32 [world, R, columns]
    logits: u64, // fp32 [R, world * columns]: the rows' logits, the ranks' columns in rank order
    taps: u64, // bf16 [R, taps * D]
    // wo_a's suh concatenated a layer (wo_a_rot: torch.cat of the slices' suh, cached on the layer)
    wo_suh: []u64,
    sink: tri_hc.Sink,
    // the mixes' side work (their dots and the Sinkhorn: useSide) on a stream of its own, joined at the next mix
    side: ?cuda.Stream = null,
    join: ?cuda.Event = null,
    side_out: bool = false,
    // the paced L2 prefetch (usePrefetch): the next kernels' weights touched into L2 on a stream of its own
    pf: ?cuda.Stream = null,
    pf_fork: ?cuda.Event = null,
    pf_join: ?cuda.Event = null,
    pf_out: bool = false,
    timer: ?*PhaseTimer = null, // eager rounds' GPU time by phase (a profile)
    // a profile's GPU clocks a round (Model.useClock): at the forward's start, then before and after each stretch
    // a profile's kernel launches by the phase they end in (eager rounds: in a graph only its capture launches)
    launches: [32]u64 = @splat(0),
    launch_mark: u64 = 0,
    clk_layers: bool = false, // a clock at each layer's end alone (tag 64 + the layer), TF_DS_CLOCK_MARKS=2
    clk_marks: bool = false,
    clk_ratio: ?u8 = null, // phase clocks only in the layers of this compress ratio (TF_DS_CLOCK_MARKS=3: ratio 2)
    cur_ratio: u8 = 0, // the layer a stretch is on (its compress ratio) // clocks at every phase mark too (Model.useClock(true)), not only around the stretches
    clk: u64 = 0, // [1 + 2 * max_clocks] u64: the count, then each clock and its phase (PhaseTimer.Phase or clock_*)

    pub fn init(e: *const Engine, a: *prompt.Arena, gpa: std.mem.Allocator, pool_cap: usize) !Round {
        const c = e.c;
        const w = e.w;
        const R = e.round_rows;
        if (R < default_rows or R > max_rows) return error.BadRoundRows;
        const d = c.hidden;
        const hc = c.hc;
        const hd = c.head_dim;
        const hl = e.heads();
        const l0 = w.layers[0];
        const nb_max = bucketFor(pool_cap, pool_cap);
        var rd: Round = undefined;
        rd.r = 0;
        // the profile's clocks off until Model.useClock (rd starts undefined: a field default does not apply)
        rd.launches = @splat(0);
        rd.launch_mark = 0;
        rd.clk_layers = false;
        rd.clk_marks = false;
        rd.clk_ratio = null;
        rd.cur_ratio = 0;
        rd.clk = 0;
        rd.bucket = 0;
        rd.has_cand = false;
        rd.kk = 0;
        rd.glue = try a.take(glue_n * max_rows * 8); // glue rows at max_rows apart
        rd.inputs = try a.take(5 * R * 8);
        rd.ids = rd.inputs;
        rd.pos = rd.inputs;
        rd.zero = try a.take(8);
        try prompt.fill(e, rd.zero, 0, 8);
        rd.h = try a.take(R * hc * d * 2);
        rd.h_alt = try a.take(R * hc * d * 2);
        rd.x = try a.take(R * d * 2);
        rd.part = try a.take(R * tri_basic.hc_blocks * 32 * 4);
        rd.pre = try a.take(R * hc * 4);
        rd.pre_a = try a.take(R * hc * 4);
        rd.pre_f = try a.take(R * hc * 4);
        rd.post = try a.take(R * hc * 4);
        rd.comb = try a.take(R * hc * hc * 4);
        rd.qa = try a.take(R * c.q_lora * 2);
        rd.ykv = try a.take(R * hd * 2);
        rd.ckv = try a.take(R * hd * 4);
        rd.cgate = try a.take(R * hd * 4);
        rd.qr = try a.take(R * c.q_lora * 2);
        rd.kout = try a.take(R * hd * 2);
        rd.xhq = try a.take(2 * R * c.q_lora * 2);
        rd.q = try a.take(R * hl * hd * 2);
        rd.iq = try a.take(R * c.index_heads * c.index_head_dim * 2);
        rd.iq4 = try a.take(R * c.index_heads * c.index_head_dim * 2);
        rd.iw = try a.take(R * c.index_heads * 2);
        rd.keys = try a.take(R * nb_max * 8);
        rd.score = try a.take(R * nb_max * 4);
        rd.cand = try a.take(R * (nb_max / c.candidate_block + 1));
        rd.bmax = try a.take(R * (nb_max / c.candidate_block + 1) * 4);
        rd.pidx = try a.take(R * c.candidate_blocks * 8);
        rd.cblk = try a.take(R * c.candidate_blocks * 4);
        rd.has_cblk = false;
        rd.cidx = try a.take(R * c.index_topk * 8);
        rd.ktop = try a.take(R * c.index_topk * 8);
        rd.tmax = try a.take(R * (nb_max / tri_index.tile + 1) * 8);
        rd.pbufs = .{ .tpos = try a.take(R * c.index_topk * 8), .cand = try a.take(R * c.index_topk * tri_index.tile * 8), .every = try a.take(R * 8) };
        {
            // topk_select_pruned's `every`: a visible count past every tile position (torch.full, each call)
            var ev: [max_rows]i64 = @splat(tri_index.every_vis);
            try prompt.upload(e, rd.pbufs.every, &ev, R * 8);
        }
        rd.kvg = try a.take(R * 2 * hd * 4);
        rd.sg = try a.take(R * 2 * hd * 4);
        rd.lat2 = try a.take(R * hd * 2);
        rd.lat = try a.take(R * hd * 2);
        rd.ikp = try a.take(R * c.index_head_dim * 2);
        rd.ik = try a.take(R * c.index_head_dim * 2);
        rd.o = try a.take(R * hl * hd * 2);
        rd.pm_ = try a.take(R * hl * tri_attn.attn_splits * 4);
        rd.pl_ = try a.take(R * hl * tri_attn.attn_splits * 4);
        rd.po_ = try a.take(R * hl * tri_attn.attn_splits * hd * 4);
        var wo_k: usize = 0;
        for (l0.wo_a[0..l0.groups]) |wo| wo_k += wo.k;
        rd.xhwo = try a.take(R * wo_k * 2);
        rd.xb = try a.take(R * l0.wo_b.k * 2);
        rd.u = try a.take(R * l0.groups * l0.wo_a[0].n * 2);
        rd.pa = try a.take(R * d * 4);
        rd.ga = try a.take(e.world * R * d * 4);
        rd.gl = try a.take(R * (d / 256) * c.experts * 4);
        rd.pick = try a.take(R * e.slots() * 4);
        rd.mw = try a.take(R * e.slots() * 4);
        rd.pm = try a.take(R * d * 4);
        rd.gm = try a.take(e.world * R * d * 4);
        rd.e_in = 0;
        rd.ek = 0;
        rd.ekg = 0;
        rd.ekv = 0;
        for (w.layers) |lay| {
            const ew = lay.engram_wkv orelse continue;
            rd.e_in = try a.take(R * ew.k * 2);
            const en = if (e.two) |tw| tw.engramWidth() else ew.n; // the summed projection's width (2D: both pairs')
            rd.ek = try a.take(R * ew.n * 4);
            rd.ekg = try a.take(e.world * R * en * 4);
            rd.ekv = try a.take(R * en * 2);
            break;
        }
        rd.xc = try a.take(R * d * 2);
        const hh = if (e.two) |tw| tw.hw[0] + tw.hw[1] else w.head.n; // a rank's vocabulary half (2D: both pairs' parts)
        rd.hl = try a.take(R * w.head.n * 4);
        rd.hg = try a.take(e.world * R * hh * 4);
        rd.logits = try a.take(R * e.world * hh * 4);
        rd.taps = try a.take(R * @max(c.dspark_taps.slice().len, 1) * d * 2);
        // wo_a_rot's suh: the slices' suh one after another (torch.cat's bytes)
        rd.wo_suh = try gpa.alloc(u64, w.layers.len);
        for (w.layers, 0..) |lay, li| {
            var k: usize = 0;
            for (lay.wo_a[0..lay.groups]) |wo| k += wo.k;
            const dst = try a.take(k * 2);
            var off: usize = 0;
            for (lay.wo_a[0..lay.groups]) |wo| {
                try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(dst + off * 2, wo.suh, wo.k * 2, e.s.handle), "cuMemcpyDtoDAsync");
                off += wo.k;
            }
            rd.wo_suh[li] = dst;
        }
        var ev = try cuda.Event.init(e.d, false);
        errdefer ev.deinit();
        // the Sinkhorn half's side stream is this stream (useSide gives it one of its own): its launches keep their
        // order and plain (no-PDL) launches
        rd.sink = .{ .s = e.s, .fork = ev };
        rd.side = null;
        rd.join = null;
        rd.side_out = false;
        rd.pf = null;
        rd.pf_fork = null;
        rd.pf_join = null;
        rd.pf_out = false;
        rd.timer = null;
        return rd;
    }

    /// The mixes' side work (the dots of the stored streams and the Sinkhorn) on a stream of its own, as the served
    /// build runs it: forked behind each mix's finish, joined before the next mix reads what it wrote. Its kernels and
    /// their inputs are the same, so the bytes are.
    pub fn useSide(rd: *Round, e: *const Engine) !void {
        if (rd.side != null) return;
        var s = try cuda.Stream.init(e.d, true);
        errdefer s.deinit();
        rd.join = try cuda.Event.init(e.d, false);
        rd.side = s;
        rd.sink.s = s;
    }

    pub fn dropSide(rd: *Round, e: *const Engine) void {
        if (rd.join) |*j| j.deinit();
        if (rd.side) |*s| s.deinit();
        rd.join = null;
        rd.side = null;
        rd.sink.s = e.s;
    }

    /// The paced L2 prefetch, as the served build's trace shows it: three launches a layer on a stream of its own, each
    /// touching the next kernels' weights into L2 while the current ones run (after the q/kv linears: wo_a and wo_b;
    /// after wo_b: the router; after the experts: the next layer's input linears). It writes nothing.
    pub fn usePrefetch(rd: *Round, e: *const Engine) !void {
        if (rd.pf != null) return;
        var s = try cuda.Stream.init(e.d, true);
        errdefer s.deinit();
        rd.pf_fork = try cuda.Event.init(e.d, false);
        rd.pf_join = try cuda.Event.init(e.d, false);
        rd.pf = s;
    }

    pub fn dropPrefetch(rd: *Round) void {
        if (rd.pf_fork) |*x| x.deinit();
        if (rd.pf_join) |*x| x.deinit();
        if (rd.pf) |*s| s.deinit();
        rd.pf_fork = null;
        rd.pf_join = null;
        rd.pf = null;
    }

    /// These ranges touched into L2 behind the work queued so far, on the prefetch stream.
    /// One paced L2 prefetch launch on its stream: `ranges` in order, the last one cut where the launch reaches `budget`
    /// bytes (the served launches move at most 16 MiB, 8 MiB behind wo_b).
    fn prefetch(rd: *Round, e: *const Engine, ranges: []const [2]u64, budget: u64) !void {
        const s = rd.pf orelse return;
        var a: ops_mod.Prefetch = .{};
        var left = budget;
        for (ranges) |r| {
            if (r[1] == 0 or a.n == 16 or left == 0) continue;
            const take = @min(r[1], left);
            a.ptr[@intCast(a.n)] = r[0];
            a.bytes[@intCast(a.n)] = take;
            a.n += 1;
            left -= take;
        }
        if (a.n == 0) return;
        try rd.pf_fork.?.record(e.s);
        try s.wait(rd.pf_fork.?);
        try e.ops.l2Prefetch(s, a);
        try rd.pf_join.?.record(s);
        rd.pf_out = true;
    }

    /// The main stream waits for the prefetches forked so far (a stretch ends joined: its graph's every stream).
    fn joinPrefetch(rd: *Round, e: *const Engine) !void {
        if (!rd.pf_out) return;
        try e.s.wait(rd.pf_join.?);
        rd.pf_out = false;
    }

    pub const max_clocks = 512;
    pub const clock_start = 250; // the forward's start
    pub const clock_wait = 251; // before a stretch: the GPU waiting on the host since the last clock
    pub const clock_tail = 252; // after a stretch: its work past its last phase

    /// A clock kernel on the main stream (captured into a graph like any kernel, its slot fixed by the round's order).
    fn clock(rd: *Round, e: *const Engine, ph: u8) !void {
        if (rd.clk == 0) return;
        try e.ops.gpuClock(e.s, rd.clk, ph, ph == clock_start, max_clocks);
    }

    fn mark(rd: *Round, e: *const Engine, p: PhaseTimer.Phase) !void {
        if (rd.clk != 0 and rd.clk_marks and (rd.clk_ratio == null or rd.clk_ratio.? == rd.cur_ratio)) {
            const now = cuda.launch.count;
            rd.launches[@intFromEnum(p)] += now -% rd.launch_mark;
            rd.launch_mark = now;
            try rd.clock(e, @intFromEnum(p));
        }
        if (rd.timer) |t| try t.mark(e.s, p);
    }

    /// The main stream waits for the side work forked so far.
    fn joinSide(rd: *Round, e: *const Engine) !void {
        if (!rd.side_out) return;
        try e.s.wait(rd.join.?);
        rd.side_out = false;
    }

    fn g(rd: *const Round, k: Glue) u64 {
        return rd.glue + @as(u64, @backingInt(k)) * max_rows * 8;
    }
};

/// rounds.py _ix's tensors for these rows (each row's slot and extent), made on the host and uploaded in one copy.
fn setGlue(e: *const Engine, rd: *Round, rows: Rows) !void {
    const R = rows.ids.len;
    var hg: [glue_n * max_rows]i64 = @splat(0);
    const rs: i64 = @intCast(e.c.window + ring_extra);
    const raw: i64 = raw_rows;
    for (0..R) |i| {
        const p = rows.pos[i];
        const at = struct {
            fn f(buf: []i64, k: Glue, r: usize) *i64 {
                return &buf[@as(usize, @backingInt(k)) * max_rows + r];
            }
        }.f;
        const slot = rows.slotOf(i);
        const base = rows.baseOf(i);
        // wbase = slot * RS, wslot = wbase + pos % RS
        at(&hg, .wbase, i).* = slot * rs;
        at(&hg, .wslot, i).* = slot * rs + @mod(p, rs);
        // ratio 1: cbase = base, ctarget = cbase + pos, gpos = pos, vis = pos + 1
        at(&hg, .cbase1, i).* = base;
        at(&hg, .ctarget1, i).* = base + p;
        at(&hg, .gpos1, i).* = p;
        at(&hg, .vis1, i).* = p + 1;
        // ratio 2: groups = pos // 2, target = groups if the group is complete else the extent's last row (scratch)
        const cb2 = @divFloor(base, 2);
        const groups = @divFloor(p, 2);
        const scratch = @divFloor(rows.endOf(i), 2) - 1 - cb2;
        at(&hg, .cbase2, i).* = cb2;
        at(&hg, .ctarget2, i).* = cb2 + (if (@mod(p + 1, 2) == 0) groups else scratch);
        at(&hg, .gpos2, i).* = groups * 2;
        at(&hg, .vis2, i).* = @divFloor(p + 1, 2);
        // rslot = slot * RAW + pos % RAW; gi = rbase + (groups * 2 + [0, 1]) % RAW, rbase = slot * RAW
        at(&hg, .rslot, i).* = slot * raw + @mod(p, raw);
        const gi0 = @as(usize, @backingInt(Glue.gi)) * max_rows;
        hg[gi0 + 2 * i] = slot * raw + @mod(groups * 2, raw);
        hg[gi0 + 2 * i + 1] = slot * raw + @mod(groups * 2 + 1, raw);
    }
    try prompt.upload(e, rd.glue, &hg, hg.len * 8);
    // RoundDecoder.set: [ids, pos, slots, base, end] in one [5, R] copy
    var in: [5 * max_rows]i64 = @splat(0);
    for (0..R) |i| {
        in[i] = rows.ids[i];
        in[R + i] = rows.pos[i];
        in[2 * R + i] = rows.slotOf(i);
        in[3 * R + i] = rows.baseOf(i);
        in[4 * R + i] = rows.endOf(i);
    }
    try prompt.upload(e, rd.inputs, &in, 5 * R * 8);
    rd.ids = rd.inputs;
    rd.pos = rd.inputs + R * 8;
}

/// The layer's RoPE table rows (model.py _cs: the compressed table for a compressed layer).
fn ropeOf(e: *const Engine, lay: weights.Layer) prompt.Rope {
    return if (lay.ratio != 0) e.compressed else e.plain;
}

/// _attention2 with _kv_idx (kv_done: q_kv_norm wrote the window KV), the served switches, xh none (hc_rot off): the
/// layer's attention partial pa [R, D] fp32 from its mixed rows x.
fn attention(e: *const Engine, rd: *Round, ch: *const Chunk, cs: *const Caches, rings: []const u64, sh: *prompt.Shared, li: usize) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    const t = e.t;
    const R = rd.r;
    const hd = c.head_dim;
    const rd_dim = c.rope_dim;
    const hl = e.heads();
    const ratio: usize = lay.ratio;
    const rope = ropeOf(e, lay);
    const ring = rings[li];
    const ring_rows = c.window + ring_extra;
    const pos = rd.pos;
    // attn_in: wq_a, wkv and (a kv source of a compressed layer) the compressor's, one group
    const has_comp = ratio != 0 and lay.comp_wkv != null;
    const ck_dt: exl3_linear.DType = if (ratio == 1) .bf16 else .f32;
    {
        var ls: [4]weights.Linear = undefined;
        var outs: [4]u64 = undefined;
        var lds: [4]usize = undefined;
        var dts: [4]exl3_linear.DType = undefined;
        var n: usize = 0;
        ls[n] = lay.wq_a;
        outs[n] = rd.qa;
        lds[n] = lay.wq_a.n;
        dts[n] = .bf16;
        n += 1;
        ls[n] = lay.wkv;
        outs[n] = rd.ykv;
        lds[n] = hd;
        dts[n] = .bf16;
        n += 1;
        if (has_comp) {
            ls[n] = lay.comp_wkv.?;
            outs[n] = rd.ckv;
            lds[n] = hd;
            dts[n] = ck_dt;
            n += 1;
            if (lay.comp_wgate) |cg| {
                ls[n] = cg;
                outs[n] = rd.cgate;
                lds[n] = hd;
                dts[n] = .f32;
                n += 1;
            }
        }
        const xs = [_]u64{ rd.x, rd.x, rd.x, rd.x };
        const ldxs = [_]usize{ c.hidden, c.hidden, c.hidden, c.hidden };
        try prompt.grouped(e, ch, R, ls[0..n], xs[0..n], ldxs[0..n], outs[0..n], lds[0..n], dts[0..n]);
    }
    try rd.mark(e, .a_in);
    // the indexer's top-k takes every entry it scans (short contexts): its selection needs no scores
    const has_idx = ratio != 0 and lay.idx_wq_b != null;
    const nb: usize = if (ratio != 0) rd.bucket / ratio else 0;
    const idx_all = has_idx and li != c.candidate_source and nb <= c.index_topk;
    const need_iq = has_idx and !idx_all;
    // q_proj with kv (rot_q): q's RMSNorm writes wq_b's (and the indexer wq_b's) rotated rows and the window KV into
    // the ring in one launch; wq_b's epilogue applies q's RoPE
    {
        var rot: [2]tri_norm.Rot = undefined;
        var calls: [2]exl3_linear.Call = undefined;
        var n: usize = 0;
        const qls = [_]?weights.Linear{ lay.wq_b, if (need_iq) lay.idx_wq_b else null };
        const outs = [_]u64{ rd.q, rd.iq };
        for (qls) |ol| {
            const l = ol orelse continue;
            const xh = rd.xhq + n * R * c.q_lora * 2;
            rot[n] = .{ .suh = l.suh, .h = xh };
            calls[n] = .{ .layer = l, .x = 0, .ldx = 0, .x_dtype = .bf16, .xh = xh, .y = outs[n], .ldy = @intCast(l.n), .y_dtype = .bf16, .counters = 0 };
            if (n == 0) calls[n].rope = .{ .cos = rope.cos, .sin = rope.sin, .pos = pos, .hd = @intCast(hd), .rd = @intCast(rd_dim) };
            n += 1;
        }
        try tri_norm.qKvNorm(t, rd.qa, lay.wq_a.n, lay.q_norm, c.eps, rot[0..n], rd.ykv, lay.kv_norm, rope.cos, rope.sin, pos, ring, ring_rows, rd.g(.wslot), true, rd_dim, rd.qr, rd.kout, R, lay.wq_a.n, hd);
        try prompt.groupedRotated(e, ch, R, calls[0..n]);
    }
    // the prefetch: this layer's wo_a and wo_b behind q's linears
    {
        var r: [5][2]u64 = @splat(.{ 0, 0 });
        for (lay.wo_a[0..lay.groups], 0..) |wo, j| r[j] = linearBytes(wo);
        r[4] = linearBytes(lay.wo_b);
        try rd.prefetch(e, &r, prefetch_budget);
    }
    try rd.mark(e, .a_q);
    // _kv_idx: the compressor's caches (a kv source) and the indexer's selection
    var comp: tri_attn.Comp = .none;
    var cbase: ?u64 = null;
    if (ratio != 0) {
        const cb_k: Glue = if (ratio == 1) .cbase1 else .cbase2;
        cbase = rd.g(cb_k);
        if (lay.comp_wkv != null) {
            if (ratio == 1) {
                try tri_basic.rmsnorm(t, rd.ckv, hd, lay.comp_norm, rd.lat, hd, c.eps, R, hd);
            } else if (ratio == 2) {
                // rk[rslot] = kv, rs[rslot] = score; the group's two rows back (rk[gi], rs[gi]); the pair's softmax-
                // weighted sum to bf16 (torch's ops: exact.compress2's bytes), then the RMSNorm (Triton's)
                try e.ops.scatterRows(e.s, rd.ckv, hd * 4, rd.g(.rslot), cs.raw_kv[li], hd * 4, hd * 4, R);
                try e.ops.scatterRows(e.s, rd.cgate, hd * 4, rd.g(.rslot), cs.raw_score[li], hd * 4, hd * 4, R);
                try e.d.check(e.d.api.cuMemsetD8Async(ch.invalid, 0, 4, e.s.handle), "cuMemsetD8Async");
                try e.ops.gatherRows(e.s, cs.raw_kv[li], cs.slots * raw_rows, rd.g(.gi), rd.kvg, hd * 4, 2 * R, ch.invalid);
                try e.ops.gatherRows(e.s, cs.raw_score[li], cs.slots * raw_rows, rd.g(.gi), rd.sg, hd * 4, 2 * R, ch.invalid);
                try e.exact.compress2(e.s, rd.kvg, rd.sg, rd.lat2, R, hd);
                try tri_basic.rmsnorm(t, rd.lat2, hd, lay.comp_norm, rd.lat, hd, c.eps, R, hd);
            } else return notPorted(@src());
            sh.kv_layer = li;
            const ctarget = rd.g(if (ratio == 1) .ctarget1 else .ctarget2);
            const gpos = rd.g(if (ratio == 1) .gpos1 else .gpos2);
            if (lay.idx_wk) |iwk| {
                const id = c.index_head_dim;
                try prompt.grouped(e, ch, R, &.{iwk}, &.{rd.lat}, &.{hd}, &.{rd.ikp}, &.{id}, &.{.bf16});
                try tri_basic.rmsnorm(t, rd.ikp, id, lay.idx_k_norm, rd.ik, id, c.eps, R, id);
                try tri_norm.ropeHeads(t, rd.ik, rope.cos, rope.sin, gpos, rd_dim, false, R, 1, id);
                try tri_attn.fp4Store(t, rd.ik, id, cs.idx_codes[li], cs.idx_scales[li], ctarget, R, id, 32, false);
            }
            // the latent rotated in place (it is not read again), then stored
            try tri_norm.ropeHeads(t, rd.lat, rope.cos, rope.sin, gpos, rd_dim, false, R, 1, hd);
            try tri_attn.fp4Store(t, rd.lat, hd, cs.comp_codes[li], cs.comp_scales[li], ctarget, R, hd, 16, true);
        }
        const src = sh.kv_layer orelse return error.NoKvSource;
        const vis = rd.g(if (ratio == 1) .vis1 else .vis2);
        if (idx_all) {
            // topk_indices of every scanned key is 0 .. nb - 1 whatever the scores, then the visible mask (on the
            // device: no host round trip in the round)
            if (nb > 512) return notPorted(@src());
            try e.ops.iotaVis(e.s, vis, rd.cidx, nb, R);
            rd.kk = nb;
        } else if (lay.idx_wq_b != null) {
            const ih = c.index_heads;
            const id = c.index_head_dim;
            const k: tri_index.IndexK = .{ .fp4 = .{ .codes = cs.idx_codes[src], .scales = cs.idx_scales[src] } };
            try tri_norm.ropeHeads(t, rd.iq, rope.cos, rope.sin, pos, rd_dim, false, R, ih, id);
            try tri_attn.fp4QdP2(t, rd.iq, rd.iq4, R * ih * id);
            try tri_norm.rowmmWts(t, rd.x, c.hidden, lay.idx_proj_h, exact.indexScale(c.index_head_dim, c.index_heads), rd.iw, R, c.hidden, ih);
            const kk = @min(c.index_topk, nb);
            const pow2 = kk & (kk - 1) == 0;
            const after_src = c.candidate_source < li;
            const top: tri_index.TopK = .{ .top = rd.ktop, .ctx = @ptrCast(@constCast(e)), .run = prompt.torchTopk };
            if (pow2 and after_src and rd.has_cblk) {
                // past the pool's width: the pool's blocks scored alone (the same keys there; the -inf keys elsewhere
                // left out: the same top-k), then the selection
                const nblk = c.candidate_blocks;
                try tri_index.indexKeysCand(t, rd.iq4, k, rd.iw, vis, nb, rd.cblk, nblk, nblk, c.candidate_block, cbase, rd.keys, R, ih, id);
                try tri_index.topkSelect(t, rd.keys, nblk * c.candidate_block, kk, vis, rd.cidx, top, R, nblk * c.candidate_block);
            } else if (pow2 and li != c.candidate_source) {
                // scores -> (the pool's mask) -> top-k keys in one launch, then the selection
                if (after_src and nb > c.candidate_blocks * c.candidate_block) return error.NoCandidatePool;
                const cand: ?tri_index.Cand = if (after_src) (if (rd.has_cand) .{ .mask = rd.cand, .stride = nb / c.candidate_block } else return error.NoCandidatePool) else null;
                if (nb > tri_index.topk_fused_max) {
                    // long buckets: each 64-key tile's maximum too; topk_select_pruned (the k best tiles, or torch's
                    // top-k of every key when there are no more tiles than k)
                    try tri_index.indexKeys(t, rd.iq4, k, rd.iw, vis, nb, cbase, cand, c.candidate_block, rd.tmax, kk, rd.keys, R, ih, id);
                    try tri_index.topkSelectPruned(t, rd.keys, nb, rd.tmax, kk, vis, rd.cidx, rd.pbufs, top, R, nb);
                } else {
                    try tri_index.indexKeys(t, rd.iq4, k, rd.iw, vis, nb, cbase, cand, c.candidate_block, null, kk, rd.keys, R, ih, id);
                    try tri_index.topkSelect(t, rd.keys, nb, kk, vis, rd.cidx, top, R, nb);
                }
            } else if (pow2 and li == c.candidate_source) {
                // the candidate source: the scores, the pool from them, then their keys' top-k
                try tri_index.indexScore(t, rd.iq4, k, rd.iw, vis, nb, rd.score, cbase, false, null, false, R, ih, id);
                const cb = c.candidate_block;
                if (nb > c.candidate_blocks * cb) {
                    // _candidate_blocks: the block maxima (the newest pinned), their candidate_blocks best (ties to the
                    // lower block, ascending), those with a finite maximum as block ids, the rest -1
                    const nbk = (nb + cb - 1) / cb;
                    try e.ops.blockMax(e.s, rd.score, nb, nb, cb, vis, rd.bmax, nbk, R);
                    try e.ops.topkIndices(e.s, rd.bmax, nbk, R, nbk, c.candidate_blocks, rd.pbufs.every, rd.pidx);
                    try e.ops.poolPick(e.s, rd.bmax, nbk, nbk, rd.pidx, c.candidate_blocks, 0, 0, rd.cblk, R);
                    rd.has_cblk = true;
                } else {
                    if (nb % cb != 0) return notPorted(@src()); // _candidates' padded path (buckets are whole blocks)
                    try e.ops.candFast(e.s, rd.score, nb, nb / cb, cb, vis, rd.cand, nb / cb, R);
                    rd.has_cand = true;
                }
                if (nb > tri_index.topk_fused_max) {
                    try tri_index.scoreKeys(t, rd.score, rd.tmax, rd.keys, R, nb);
                    try tri_index.topkSelectPruned(t, rd.keys, nb, rd.tmax, kk, vis, rd.cidx, rd.pbufs, top, R, nb);
                } else {
                    try tri_index.scoreKeys(t, rd.score, null, rd.keys, R, nb);
                    try tri_index.topkSelect(t, rd.keys, nb, kk, vis, rd.cidx, top, R, nb);
                }
            } else return notPorted(@src()); // the scores' topk_indices path (a top-k of no power of two)
            rd.kk = kk;
        }
        comp = .{ .fp4 = .{ .codes = cs.comp_codes[src], .scales = cs.comp_scales[src] } };
    }
    try rd.mark(e, .a_idx);
    // rot_attn: the merge applies the inverse RoPE and writes wo_a's rotated input rows; wo_ab with its fold
    const gk = lay.wo_a[0].k;
    try tri_attn.sparseAttn(t, .{
        .q = rd.q,
        .out = rd.o,
        .rows = R,
        .h = hl,
        .hd = hd,
        .sink = lay.sink,
        .wsrc = ring,
        .wsrc_rows = ring_rows,
        .wlo = rd.zero,
        .ring = true,
        .comp = comp,
        .idx = if (ratio != 0) rd.cidx else 0,
        .n_idx = if (ratio != 0) rd.kk else 0,
        .pos = pos,
        .scale = prompt.scale(hd),
        .window = c.window,
        .wbase = rd.g(.wbase),
        .cbase = cbase,
        .ring_rows = ring_rows,
        .rot = .{ .cos = rope.cos, .sin = rope.sin, .rd = rd_dim, .suh = rd.wo_suh[li], .xh = rd.xhwo, .gh = gk / hd },
        .parts = .{ .pm = rd.pm_, .pl = rd.pl_, .po = rd.po_ },
    });
    try rd.mark(e, .a_core);
    var calls: [exl3_linear.gmax]exl3_linear.Call = undefined;
    const uw = lay.groups * lay.wo_a[0].n;
    var col: usize = 0;
    var off: usize = 0;
    for (lay.wo_a[0..lay.groups], 0..) |wo, gi| {
        const rot: exl3_linear.RotOut = if (e.two) |tw| round2d.woRot(tw, lay, col) else .{ .suh = lay.wo_b.suh, .xh = rd.xb, .ldr = @intCast(lay.wo_b.k), .off = @intCast(col) };
        calls[gi] = .{ .layer = wo, .x = 0, .ldx = 0, .x_dtype = .bf16, .xh = rd.xhwo + off * 2, .y = rd.u + col * 2, .ldy = @intCast(uw), .y_dtype = .bf16, .counters = 0, .rot = rot };
        col += wo.n;
        off += R * wo.k;
    }
    try prompt.groupedRotated(e, ch, R, calls[0..lay.groups]);
    try rd.mark(e, .a_woa);
    if (e.two) |tw| try round2d.woExchange(tw, e, rd.xb, R); // 2D: the column partner's half of wo_b's input rows
    try rd.mark(e, .a_x);
    var cb_call = [_]exl3_linear.Call{.{ .layer = lay.wo_b, .x = 0, .ldx = 0, .x_dtype = .bf16, .xh = rd.xb, .y = rd.pa, .ldy = @intCast(lay.wo_b.n), .y_dtype = .f32, .counters = 0 }};
    try prompt.groupedRotated(e, ch, R, &cb_call);
    // the prefetch behind wo_b: the ffn mix's weights, the router, the shared expert's gate
    try rd.prefetch(e, &.{ .{ lay.hc_ffn[0], hcFnBytes(c) }, .{ lay.gate_w, @as(u64, c.experts) * c.hidden * 2 }, lay.experts.shared_gate, lay.experts.shared_up, lay.experts.shared_down }, prefetch_budget / 2);
}

/// Model.moe of a decode window (shared_side off: SHARED_OVERLAP unset in the served lane): the gate's chunk sums
/// (rowmm_gate), the routing with the shared expert in every row's last slot, the routed experts' decode path.
fn moe(e: *const Engine, rd: *Round, ch: *const Chunk, li: usize) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    const R = rd.r;
    const sl = e.slots();
    const kc = try tri_norm.rowmmGate(e.t, rd.x, c.hidden, lay.gate_w, rd.gl, R, c.hidden, c.experts);
    try tri_norm.route(e.t, rd.gl, kc, lay.gate_b, c.top_k, c.routed_scaling, lay.experts.count - 1, rd.pick, rd.mw, R, c.experts, sl);
    try rd.mark(e, .m_rt);
    if (e.two) |tw| {
        // 2D (round2d.experts, the profile's marks between): gate / up, the intermediate's exchange, down
        try exl3_experts2d.decodeGateUp(e.ex, e.s, lay.experts, ch.xsd, rd.x, c.hidden, rd.pick, rd.mw, rd.pm, R, c.swiglu_limit);
        try rd.mark(e, .m_gu);
        try tw.catRank(e, ch.xsd.xd, tw.xd_full, R * ch.xsd.slots, tw.gu, 2);
        try rd.mark(e, .m_x);
        try exl3_experts2d.decodeDown(e.ex, e.s, lay.experts, ch.xsd, tw.xd_full, rd.pick, rd.mw, rd.pm, R);
    } else try exl3_experts.decode(e.ex, e.s, lay.experts, ch.xsd, rd.x, c.hidden, rd.pick, rd.mw, rd.pm, R, c.swiglu_limit);
    // the prefetch behind the experts: the next layer's attention mix and input linears (an Engram layer: the first
    // 4 MiB of its wkv), after the last layer the head
    if (li + 1 < e.w.layers.len) {
        const nx = e.w.layers[li + 1];
        if (nx.engram_wkv) |ew| {
            try rd.prefetch(e, &.{linearBytes(ew)}, prefetch_budget / 4);
        } else {
            try rd.prefetch(e, &.{ .{ nx.hc_attn[0], hcFnBytes(c) }, linearBytes(nx.wq_a), linearBytes(nx.wkv), if (nx.comp_wkv) |x| linearBytes(x) else .{ 0, 0 }, if (nx.comp_wgate) |x| linearBytes(x) else .{ 0, 0 }, linearBytes(nx.wq_b) }, prefetch_budget);
        }
    } else try rd.prefetch(e, &.{linearBytes(e.w.head)}, prefetch_budget);
}

/// An EXL3 linear's words: their first byte and length (k * n values of k2 / 2 bits).
fn linearBytes(l: weights.Linear) [2]u64 {
    return .{ l.words, @as(u64, l.k) * l.n * l.k2 / 16 };
}

/// A paced prefetch launch's most bytes (the served launch trace: 16 MiB; 8 MiB behind wo_b, 4 MiB before an Engram
/// layer).
const prefetch_budget: u64 = 16 << 20;

/// An mHC mix's weights hc_*_fn: fp32 [24, hc * d].
fn hcFnBytes(c: anytype) u64 {
    return 24 * @as(u64, c.hc) * c.hidden * 4;
}

/// A round's GPU time by phase (tf-dsv41-lanes --profile 2): a timing event on the main stream at each phase's end,
/// the elapsed times read after the round. Eager rounds only (in a graph the events would be captured).
pub const PhaseTimer = struct {
    pub const Phase = enum(u8) { start, embed, engram, mix_attn, attn, gather_a, mix_ffn, moe, gather_m, post, head, a_in, a_q, a_idx, a_core, a_woa, a_x, m_rt, m_gu, m_x };
    pub const phases = @typeInfo(Phase).@"enum".field_names;
    const max_marks = 1024;
    events: []cuda.Event,
    phase: [max_marks]Phase = undefined,
    n: usize = 0,
    ms: [phases.len]f64 = @splat(0),
    rounds: u64 = 0,
    gpa: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver) !PhaseTimer {
        const ev = try gpa.alloc(cuda.Event, max_marks);
        var made: usize = 0;
        errdefer {
            for (ev[0..made]) |*x| x.deinit();
            gpa.free(ev);
        }
        for (ev) |*x| {
            x.* = try cuda.Event.init(d, true);
            made += 1;
        }
        return .{ .events = ev, .gpa = gpa };
    }

    pub fn deinit(t: *PhaseTimer) void {
        for (t.events) |*x| x.deinit();
        t.gpa.free(t.events);
    }

    pub fn mark(t: *PhaseTimer, s: cuda.Stream, p: Phase) !void {
        if (t.n == max_marks) return;
        try t.events[t.n].record(s);
        t.phase[t.n] = p;
        t.n += 1;
    }

    /// The round's marks read: each phase's time is from the mark before it to its own.
    pub fn finish(t: *PhaseTimer) !void {
        if (t.n == 0) return;
        try t.events[t.n - 1].synchronize();
        for (1..t.n) |i| t.ms[@backingInt(t.phase[i])] += try cuda.Event.elapsedMs(t.events[i - 1], t.events[i]);
        t.n = 0;
        t.rounds += 1;
    }
};

/// What a stretch of a round leaves the next on the host: the stream buffers' roles, the taps written, the latest kv
/// source and selection, the indexer's pool flags and k. Every round starts from the same roles (the buffers' contents
/// never carry over), so a stretch's launches and what it leaves depend on its rows and bucket alone.
const Carry = struct {
    h: u64,
    spare: u64,
    pre: u64,
    pre_f: u64,
    ntap: usize = 0,
    sh: prompt.Shared = .{},
    kk: usize = 0,
    has_cand: bool = false,
    has_cblk: bool = false,
};

/// The rounds' stretches as CUDA graphs (the served graph.py: one capture a rows and context bucket, the round cut
/// before each Engram layer, whose rows the host reads between the stretches): a stretch is captured the first time
/// its (rows, bucket, stretch) comes up and replayed after, with what it leaves. The same launches with the same
/// arguments as the eager round, so the same bytes.
const storage_guard = @import("graph_storage.zig");

pub const Graphs = struct {
    gpa: std.mem.Allocator,
    map: std.AutoHashMapUnmanaged(Key, Entry) = .empty,
    guard: bool = false,

    pub const Key = struct { rows: u32, bucket: u64, stretch: u32 };
    const Entry = struct { exec: cuda.graph.Exec, start: Carry, end: Carry, storage: ?[]u64 = null };

    pub fn init(gpa: std.mem.Allocator) Graphs {
        return .{ .gpa = gpa, .guard = storage_guard.enabled() };
    }

    pub fn deinit(g: *Graphs) void {
        var it = g.map.valueIterator();
        while (it.next()) |en| { en.exec.deinit(); if (en.storage) |v| g.gpa.free(v); }
        g.map.deinit(g.gpa);
        g.* = undefined;
    }

    /// The captured stretches.
    pub fn count(g: *const Graphs) usize {
        return g.map.count();
    }

    fn run(g: *Graphs, key: Key, e: *const Engine, rd: *Round, ch: *const Chunk, cs: *const Caches, rings: []const u64, cy: *Carry, li: usize, end: usize) !void {
        var signature: storage_guard.Signature = .{};
        if (g.guard) {
            try signature.add(@intFromPtr(e));
            try signature.add(@intFromPtr(e.w.layers.ptr));
            try signature.storage(rd.*, &.{"launch_mark"});
            try signature.storage(rd.pbufs, &.{});
            try signature.storage(rd.sink, &.{});
            try signature.words(rd.wo_suh);
            try signature.storage(ch.xsd, &.{});
            try signature.add(ch.invalid);
            try signature.words(&cs.comp_codes); try signature.words(&cs.comp_scales);
            try signature.words(&cs.idx_codes); try signature.words(&cs.idx_scales);
            try signature.words(&cs.raw_kv); try signature.words(&cs.raw_score);
            try signature.words(rings);
        }
        if (g.map.getPtr(key)) |en| {
            if (g.guard) try signature.check(en.storage orelse return error.GraphStorageChanged);
            if (!std.meta.eql(en.start, cy.*)) return error.GraphStateMismatch;
            try en.exec.launchOn(e.s);
            cy.* = en.end;
            rd.kk = cy.kk;
            rd.has_cand = cy.has_cand;
            rd.has_cblk = cy.has_cblk;
            return;
        }
        const start = cy.*;
        try cuda.graph.beginCapture(e.s, .thread_local);
        stretch(e, rd, ch, cs, rings, cy, li, end, null) catch |err| {
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
        try g.map.put(g.gpa, key, .{ .exec = exec, .start = start, .end = cy.*, .storage = storage });
        errdefer _ = g.map.remove(key); // launch failure must not retain a destroyed exec
        try exec.launchOn(e.s);
    }
};

/// A round of these rows through every layer and the head: rd.logits [R, vocab] and rd.taps. The caches (ring, the
/// compressed rows, the positional store) are the prompt's, extended by the rows; eh: the Engram host (the rows' n-grams
/// hashed from `seq`, the whole sequence's ids through the rows'). With `graphs` each stretch runs as its graph and a
/// probe sees only the round's end (its logits and taps).
pub fn forward(e: *const Engine, rd: *Round, ch: *const Chunk, cs: *const Caches, rings: []const u64, eh: ?*prompt.EngramHost, seq: []const i32, rows: Rows, pool_cap: usize, probe: ?Probe, graphs: ?*Graphs) !void {
    const w = e.w;
    const R = rows.ids.len;
    if (R == 0 or R > e.round_rows or rows.pos.len != R) return error.BadRound;
    rd.r = R;
    // kernels.round_rows(): up to the round's bound its rows take the row-invariant decode kernels
    const bound = tri.decode_rows;
    tri.decode_rows = e.round_rows;
    defer tri.decode_rows = bound;
    var deepest: usize = 0;
    for (rows.pos) |p| deepest = @max(deepest, @as(usize, @intCast(p)) + 1);
    rd.bucket = bucketFor(deepest, pool_cap);
    rd.has_cand = false;
    rd.has_cblk = false;
    rd.kk = 0;
    try setGlue(e, rd, rows);
    var cy: Carry = .{ .h = rd.h, .spare = rd.h_alt, .pre = rd.pre, .pre_f = rd.pre_f };
    // the round's Engram reads, every layer's at once (AIO): hashed and submitted once the first stretch is issued
    // (its GPU time covers the reads), landed on the host before each Engram layer's stretch
    var ah: Ahead = .{};
    var ahead = e.c.engram_layers.slice().len == 0;
    // the stretches: cut before each Engram layer (TF_DS_ENGRAM_SPLIT on), its rows read on the host before it
    var li: usize = 0;
    var sn: u32 = 0;
    try rd.clock(e, Round.clock_start);
    rd.launch_mark = cuda.launch.count;
    while (li < w.layers.len) : (sn += 1) {
        var end = li + 1;
        while (end < w.layers.len and w.layers[end].engram_wkv == null) end += 1;
        if (w.layers[li].engram_wkv != null) {
            if (!ahead) try engramAhead(e, eh orelse return error.NoEngramTables, seq, rows, &ah);
            ahead = true;
            try engramLand(e, rd, eh orelse return error.NoEngramTables, &ah, li);
        }
        try rd.clock(e, Round.clock_wait);
        if (graphs) |g| {
            try g.run(.{ .rows = @intCast(R), .bucket = rd.bucket, .stretch = sn }, e, rd, ch, cs, rings, &cy, li, end);
        } else try stretch(e, rd, ch, cs, rings, &cy, li, end, probe);
        try rd.clock(e, Round.clock_tail);
        if (!ahead) try engramAhead(e, eh orelse return error.NoEngramTables, seq, rows, &ah);
        ahead = true;
        li = end;
    }
    if (graphs != null) {
        try Probe.check(probe, .logits, w.layers.len, rd.logits);
        if (cy.ntap > 0) try Probe.check(probe, .taps, w.layers.len, rd.taps);
    }
}

/// Layers li .. end of a round (the first stretch starts with the rows' streams, the last ends with the head), from
/// what the stretch before left (cy), which it updates. No host step: the Engram rows are in rd.e_in already.
fn stretch(e: *const Engine, rd: *Round, ch: *const Chunk, cs: *const Caches, rings: []const u64, cy: *Carry, li: usize, end: usize, probe: ?Probe) !void {
    const c = e.c;
    const w = e.w;
    const t = e.t;
    const R = rd.r;
    const d = c.hidden;
    if (li == 0) {
        try rd.mark(e, .start);
        try tri_basic.embedInit(t, w.embed, rd.ids, cy.h, cy.pre, R, d, c.hc);
        try rd.mark(e, .embed);
    }
    var sh = cy.sh;
    var h = cy.h;
    var spare = cy.spare;
    var pre = cy.pre;
    var pre_f = cy.pre_f;
    var ntap = cy.ntap;
    var pending: ?u64 = null;
    for (li..end) |l| {
        const lay = w.layers[l];
        rd.cur_ratio = lay.ratio;
        if (lay.engram_wkv) |ew| {
            if (pending) |gp| {
                try tri_basic.hcPost(t, gp, h, rd.post, rd.comb, h, e.world, R, d);
                pending = null;
            }
            try prompt.grouped(e, ch, R, &.{ew}, &.{rd.e_in}, &.{ew.k}, &.{rd.ek}, &.{ew.n}, &.{.f32});
            try Probe.check(probe, .engram_proj, l, rd.ek);
            const en = if (e.two) |tw| tw.engramWidth() else ew.n;
            if (e.two) |tw| try tw.quarters(e, rd.ek, rd.ekg, R, tw.ew, 4) else try e.gatherF32(rd.ek, rd.ekg, R * ew.n);
            try Probe.check(probe, .engram_gather, l, rd.ekg);
            try e.ops.sumRanksBf16(e.s, rd.ekg, rd.ekv, R * en, if (e.two != null) 2 else e.world);
            try tri_basic.engramGate(t, h, rd.ekv, lay.engram_qk, spare, c.eps, R, d);
            std.mem.swap(u64, &h, &spare);
            try rd.mark(e, .engram);
        }
        // the attention mixes, the pending post fused in (pre_mix)
        h = try mix(e, rd, h, &spare, &pending, lay.hc_attn, pre, lay.attn_norm, rd.pre_a, R);
        try rd.mark(e, .mix_attn);
        if (std.mem.indexOfScalar(u16, c.dspark_taps.slice(), @intCast(l)) != null) {
            // one launch a tap, straight into its block of the taps buffer
            try tri_basic.tap(t, h, rd.taps + ntap * d * 2, c.dspark_taps.slice().len * d, R, d);
            ntap += 1;
        }
        try Probe.check(probe, .attn_in, l, rd.x);
        try attention(e, rd, ch, cs, rings, &sh, l);
        try rd.mark(e, .attn);
        try Probe.check(probe, .attn_out, l, rd.pa);
        if (e.two) |tw| try tw.quarters(e, rd.pa, rd.ga, R, tw.ow, 4) else try e.gatherF32(rd.pa, rd.ga, R * d);
        try rd.mark(e, .gather_a);
        try Probe.check(probe, .attn_gather, l, rd.ga);
        pending = rd.ga;
        h = try mix(e, rd, h, &spare, &pending, lay.hc_ffn, rd.pre_a, lay.ffn_norm, pre_f, R);
        try rd.mark(e, .mix_ffn);
        try Probe.check(probe, .moe_in, l, rd.x);
        try moe(e, rd, ch, l);
        try rd.mark(e, .moe);
        try Probe.check(probe, .moe_out, l, rd.pm);
        if (e.two) |tw| try tw.quarters(e, rd.pm, rd.gm, R, tw.dw, 4) else try e.gatherF32(rd.pm, rd.gm, R * d);
        try rd.mark(e, .gather_m);
        if (rd.clk_layers) try rd.clock(e, @intCast(64 + l));
        try Probe.check(probe, .moe_gather, l, rd.gm);
        pending = rd.gm;
        std.mem.swap(u64, &pre, &pre_f);
    }
    try rd.joinSide(e); // (a stretch ends joined: the post reads the side work's post and comb)
    try rd.joinPrefetch(e);
    if (pending) |gp| try tri_basic.hcPost(t, gp, h, rd.post, rd.comb, h, e.world, R, d);
    try rd.mark(e, .post);
    cy.* = .{ .h = h, .spare = spare, .pre = pre, .pre_f = pre_f, .ntap = ntap, .sh = sh, .kk = rd.kk, .has_cand = rd.has_cand, .has_cblk = rd.has_cblk };
    if (end < w.layers.len) return;
    // the head on every row: the logits, the ranks' columns in rank order (g.permute(1, 0, 2))
    try tri_basic.collapseNorm(t, h, pre, w.norm, rd.xc, c.eps, R, d);
    const hn: usize = w.head.n;
    try prompt.grouped(e, ch, R, &.{w.head}, &.{rd.xc}, &.{d}, &.{rd.hl}, &.{hn}, &.{.f32});
    try Probe.check(probe, .head_cols, w.layers.len, rd.hl);
    const hh = if (e.two) |tw| tw.hw[0] + tw.hw[1] else hn; // a rank's vocabulary half (2D: both pairs' quarters)
    if (e.two) |tw| try tw.quarters(e, rd.hl, rd.hg, R, tw.hw, 4) else try e.gatherF32(rd.hl, rd.hg, R * hn);
    try Probe.check(probe, .head_gather, w.layers.len, rd.hg);
    for (0..e.world) |k| try e.ops.copyRows(e.s, rd.hg + k * R * hh * 4, hh * 4, rd.logits + k * hh * 4, e.world * hh * 4, hh * 4, R);
    try rd.mark(e, .head);
    try Probe.check(probe, .logits, w.layers.len, rd.logits);
    if (ntap > 0) try Probe.check(probe, .taps, w.layers.len, rd.taps);
}

/// rounds.py pre_mix: hc_pre2 of the streams h with the pending gathered partials posted in (into the spare buffer,
/// which becomes the streams), the Sinkhorn half deferred to the side stream (this one); returns the streams.
fn mix(e: *const Engine, rd: *Round, h: u64, spare: *u64, pending: *?u64, params: [3]u64, pre_in: u64, norm: u64, pre_out: u64, R: usize) !u64 {
    const c = e.c;
    try rd.joinSide(e);
    defer if (rd.side) |s| if (rd.join.?.record(s)) {
        rd.side_out = true;
    } else |_| {};
    if (pending.*) |gp| {
        const out = try tri_hc.hcPre2(e.t, h, params[0], params[1], params[2], pre_in, norm, c.eps, c.hc_eps, c.hc_iters, rd.x, pre_out, rd.post, rd.comb, rd.part, .{ .gathered = gp, .world = e.world, .h_out = spare.* }, rd.sink, R, c.hidden);
        pending.* = null;
        spare.* = h;
        return out;
    }
    return tri_hc.hcPre2(e.t, h, params[0], params[1], params[2], pre_in, norm, c.eps, c.hc_eps, c.hc_iters, rd.x, pre_out, rd.post, rd.comb, rd.part, null, rd.sink, R, c.hidden);
}

/// A round's Engram reads, ahead of the stretches that use them: every window's n-grams hashed once (all the Engram
/// layers), each layer's ids in its own part of the host buffers (two layers in flight); with AIO every layer's reads
/// go out at the round's start, else each layer's when its stretch comes (the reader pool).
const Ahead = struct { ids: [2]u64 = .{ 0, 0 }, aio: bool = false, R: usize = 0 };

/// Layer slot j's first row in the Engram host's buffers (a round's rows of each Engram layer apart).
fn aheadRow(j: usize) usize {
    return j * max_rows;
}

fn engramAhead(e: *const Engine, eh: *prompt.EngramHost, seq: []const i32, rows: Rows, ah: *Ahead) !void {
    const c = e.c;
    const R = rows.ids.len;
    const cols = eh.hasher.cols();
    const k = eh.hi - eh.lo;
    const one = [_]Window{.{ .row = 0, .n = R, .seq = seq }};
    const windows = rows.windows orelse &one;
    if (c.engram_layers.slice().len > 2 or eh.flat.len < 2 * max_rows * k) return error.EngramShape;
    const t0 = stamp(eh);
    for (windows) |w| {
        const start: usize = @intCast(rows.pos[w.row]);
        if (w.row + w.n > R or w.seq.len < start + w.n) return error.ShortSequence;
        const per = eh.hasher.layers * cols;
        eh.hasher.hashes(w.seq, start, w.n, eh.hashes[w.row * per .. (w.row + w.n) * per]);
    }
    ah.* = .{ .aio = eh.aio != null, .R = R };
    for (c.engram_layers.slice(), 0..) |li, j| {
        const tbl = eh.tables.layers.get(li) orelse return error.MissingEngramTable;
        const f0 = aheadRow(j) * k;
        for (0..R) |r| {
            for (0..k) |q| eh.flat[f0 + r * k + q] = eh.hashes[(r * eh.hasher.layers + j) * cols + eh.lo + q];
        }
        if (eh.aio) |aio| ah.ids[j] = try aio.start(tbl, eh.flat[f0..][0 .. R * k], eh.w[f0 * tbl.row_w ..][0 .. R * k * tbl.row_w], eh.s[f0 * tbl.row_s ..][0 .. R * k * tbl.row_s]);
    }
    if (eh.io != null) eh.t_hash += stamp(eh) - t0;
}

/// Layer li's Engram rows of the round's rows (Engram.hashes of each row's stream's ids through the row, this rank's
/// columns, the tables' FP8 rows decoded to bf16 as Engram._decode does) into rd.e_in: their reads landed (or made
/// now), decoded, uploaded.
fn engramLand(e: *const Engine, rd: *Round, eh: *prompt.EngramHost, ah: *const Ahead, li: usize) !void {
    const c = e.c;
    const j = std.mem.indexOfScalar(u16, c.engram_layers.slice(), @intCast(li)) orelse return error.NotAnEngramLayer;
    const tbl = eh.tables.layers.get(@intCast(li)) orelse return error.MissingEngramTable;
    const k = eh.hi - eh.lo;
    const f0 = aheadRow(j) * k;
    const m = ah.R * k;
    const t1 = stamp(eh);
    if (ah.aio) {
        try eh.aio.?.wait(ah.ids[j]);
    } else try eh.pool.gather(tbl, eh.flat[f0..][0..m], eh.w[f0 * tbl.row_w ..][0 .. m * tbl.row_w], eh.s[f0 * tbl.row_s ..][0 .. m * tbl.row_s]);
    const t2 = stamp(eh);
    const w0 = f0 * tbl.row_w;
    const s0 = f0 * tbl.row_s;
    for (0..m) |i| engram.decodeRow(eh.w[w0 + i * tbl.row_w ..][0..tbl.row_w], eh.s[s0 + i * tbl.row_s ..][0..tbl.row_s], eh.rows[w0 + i * tbl.row_w ..][0..tbl.row_w]);
    const t3 = stamp(eh);
    try prompt.upload(e, rd.e_in, eh.rows[w0..].ptr, m * tbl.row_w * 2);
    if (eh.io != null) {
        eh.t_read += t2 - t1;
        eh.t_decode += t3 - t2;
        eh.t_upload += stamp(eh) - t3;
        eh.calls += 1;
    }
}

/// The host clock (ns) when the Engram host keeps time (0 when it does not).
fn stamp(eh: *const prompt.EngramHost) u64 {
    const io = eh.io orelse return 0;
    return @intCast(std.Io.Timestamp.now(io, .awake).nanoseconds);
}

test "context buckets as graph.py makes them" {
    try std.testing.expectEqual(@as(usize, 1024), bucketFor(1, 1 << 20));
    try std.testing.expectEqual(@as(usize, 4096), bucketFor(2230, 1 << 20));
    try std.testing.expectEqual(@as(usize, 4096), bucketFor(4096, 1 << 20));
    try std.testing.expectEqual(@as(usize, 8192), bucketFor(4097, 1 << 20));
    try std.testing.expectEqual(@as(usize, 4096), bucketFor(5000, 4096));
}

/// error.NotPortedYet with its site on stderr (a request that reaches a path the port lacks names it in the log).
fn notPorted(src: std.builtin.SourceLocation) error{NotPortedYet} {
    std.debug.print("{{\"not_ported\": \"{s}:{d}\"}}\n", .{ src.file, src.line });
    return error.NotPortedYet;
}
