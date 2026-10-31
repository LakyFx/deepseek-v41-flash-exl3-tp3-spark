//! kernels.py's sparse attention and packed FP4 rows: sparse_attn (the split keys and their merge, the merge with the
//! inverse RoPE and wo_a's input rotation, or one pass for prompt chunks), fp4_store and fp4_qd_p2, as Python launches them.
const std = @import("std");
const cuda = @import("cuda");
const tri = @import("tri.zig");
const aot = cuda.aot;
const Tri = tri.Tri;
const u = tri.u;
const p = aot.ptr;

/// kernels.DECODE_ROWS (TF_DS_DECODE_ROWS unset in the served build): calls of up to this many rows split the keys.
pub const decode_rows = 16;
/// kernels.ATTN_SPLITS: the key splits of a decode / verify window's rows.
pub const attn_splits = 8;

fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}

fn cb(name: []const u8, v: bool) aot.Const {
    return aot.ci(name, @intFromBool(v));
}

/// kernels._pow2: n is a power of two.
fn isPow2(n: usize) bool {
    return n > 0 and n & (n - 1) == 0;
}

/// sparse_attn's compressed keys: none, bf16 rows [N, HD], or a packed FP4 pair (codes u8 [N, HD/2], scales u8 [N, HD/16]).
pub const Comp = union(enum) { none, bf16: u64, fp4: struct { codes: u64, scales: u64 } };

/// sparse_attn's rot = (cos, sin, rope dim, wo_a's suh concatenated, its rotated rows' buffer, heads a slice).
pub const Rot = struct { cos: u64, sin: u64, rd: usize, suh: u64, xh: u64, gh: usize };

/// A split call's partials (rows <= decode_rows; Python allocates them): pm, pl fp32 [R * H * 8], po fp32 [R * H * 8 * HD].
pub const Parts = struct { pm: u64, pl: u64, po: u64 };

/// sparse_attn's arguments, Python's keyword defaults as field defaults.
pub const Attn = struct {
    /// q [rows, h, hd] bf16; out the same shape (Python: empty_like(q) when not given)
    q: u64,
    out: u64,
    rows: usize,
    h: usize,
    hd: usize,
    /// sink fp32 [h]
    sink: u64,
    /// the window keys bf16 [wsrc_rows, hd]: a ring (slot = position % ring size) or rows from position wlo[0] (int64 [1])
    wsrc: u64,
    wsrc_rows: usize,
    wlo: u64,
    ring: bool,
    comp: Comp = .none,
    /// int64 [rows, n_idx] (-1 = none; n_idx 0: no idx); prompt chunks read rows of n_idx rounded up to 16, -1 past n_idx
    idx: u64 = 0,
    n_idx: usize = 0,
    /// pos int64 [rows]
    pos: u64,
    /// Python's float as Triton passes it: rounded to fp32 (hd ** -0.5: 0x3d3504f3)
    scale: f32,
    window: usize,
    /// per-row ring / compressed row bases, int64 [rows] (concurrent rounds and batched drafts: rings only)
    wbase: ?u64 = null,
    cbase: ?u64 = null,
    /// a pool's ring size (0: none, wsrc's rows)
    ring_rows: usize = 0,
    rot: ?Rot = null,
    /// unused by prompt chunks (rows > decode_rows): one split writes out directly
    parts: Parts = .{ .pm = 0, .pl = 0, .po = 0 },
    /// a small prompt chunk run as a large one (one split, out directly: the prompt chunks' variant)
    one_split: bool = false,
};

/// sparse_attn: q -> out (o [R, H, HD]); window keys from wsrc, compressed keys comp[idx[r, j]].
pub var prompt_masked_heads = false;
pub var prompt_rank: u32 = 0;
var masked_checked: [2]bool = .{false,false};

pub fn maskedEligible(enabled: bool, a: Attn) bool {
    return enabled and a.rows>tri.decode_rows and a.h==24 and a.hd==512 and
        !a.ring and a.wbase==null and (a.comp==.none or a.comp==.fp4);
}

pub fn sparseAttn(t: Tri, a: Attn) !void {
    if (maskedEligible(prompt_masked_heads,a)) {
        const which: usize = @intFromBool(a.comp!=.none and a.n_idx>0);
        if (t.log==null and !masked_checked[which]) return qualifyMasked(t,a,which);
        return sparseAttnImpl(t,a,16);
    }
    return sparseAttnImpl(t,a,try tri.headBlock(a.h));
}

fn qualifyMasked(t: Tri,a: Attn,which: usize) !void {
    const bytes=a.rows*a.h*a.hd*2;
    const alloc=std.heap.page_allocator;
    const reference=try alloc.alloc(u8,bytes);defer alloc.free(reference);
    const actual=try alloc.alloc(u8,bytes);defer alloc.free(actual);
    try sparseAttnImpl(t,a,8);try t.stream.synchronize();
    try t.stream.d.check(t.stream.d.api.cuMemcpyDtoH_v2(reference.ptr,a.out,bytes),"masked attention reference");
    try sparseAttnImpl(t,a,16);try t.stream.synchronize();
    try t.stream.d.check(t.stream.d.api.cuMemcpyDtoH_v2(actual.ptr,a.out,bytes),"masked attention candidate");
    if(!std.mem.eql(u8,reference,actual)) return error.MaskedAttentionNumericalMismatch;
    masked_checked[which]=true;
    std.debug.print("prefill_masked_attn_check: rank={d} packed={d} rows={d} hb=16 exact_bf16=PASS\n",.{prompt_rank,which,a.rows});
}

fn sparseAttnImpl(t: Tri, a: Attn, hb: usize) !void {
    const bn = 32;
    const has = a.comp != .none and a.n_idx > 0;
    const pack = has and a.comp == .fp4;
    const codes: u64 = if (!has) a.wsrc else switch (a.comp) {
        .none => unreachable,
        .bf16 => |k| k,
        .fp4 => |k| k.codes,
    };
    const csc: u64 = if (pack) a.comp.fp4.scales else a.wsrc;
    const ct: []const u8 = if (pack) "*u8" else "*bf16";
    var n_idx: usize = if (has) a.n_idx else 0;
    const sp: usize = if (a.rows <= tri.decode_rows and !a.one_split) attn_splits else 1;
    const groups = (a.h+hb-1) / hb;
    const final = sp == 1;
    var picks: usize = if (has) tri.cdiv(n_idx, bn) else 0;
    if (final and picks > 0) {
        // prompt chunks: the pick list padded with -1 to a multiple of 16 entries, the pick blocks to a power of two
        n_idx = (n_idx + 15) / 16 * 16;
        picks = tri.pow2(picks);
    }
    const nblk = a.window / bn + picks;
    const ring_size: usize = if (a.ring) (if (a.ring_rows != 0) a.ring_rows else a.wsrc_rows) else 16;
    const pp: Parts = if (final) .{ .pm = a.out, .pl = a.out, .po = a.out } else a.parts;
    const pt: []const u8 = if (final) "*bf16" else "*fp32";
    const based = a.wbase != null;
    if (based and !a.ring) return error.BasesNeedRing;
    const rbase = a.wbase orelse a.pos;
    const cbase = if (based) a.cbase orelse a.wbase.? else a.pos;
    // switch "attn_pf" (on in the served build): a split's window keys into L2 before its PDL wait
    const wpf = a.ring and !final;
    try t.run("_sparse_attn_part", .{ u(a.rows), u(groups), u(sp) }, &.{
        p("Q", "*bf16", a.q),              p("WSRC", "*bf16", a.wsrc),                  p("WLO", "*i64", a.wlo),    p("COMP", ct, codes),
        p("CSC", ct, csc),                 p("IDX", "*i64", if (has) a.idx else a.pos), p("POS", "*i64", a.pos),    p("PM", pt, pp.pm),
        p("PL", pt, pp.pl),                p("PO", pt, pp.po),                          p("SINK", "*fp32", a.sink), p("OUT", "*bf16", a.out),
        aot.float("scale", a.scale),       p("RBASE", "*i64", rbase),                   p("CBASE", "*i64", cbase),  aot.int("ring_size", @intCast(ring_size)),
        aot.int("n_idx", @intCast(n_idx)),
    }, &.{
        ci("H", a.h),       ci("HD", a.hd),        ci("HB", hb),       ci("WIN", a.window), ci("BN", bn),
        cb("RING", a.ring), cb("HAS_COMP", has),   cb("PACKED", pack), ci("SPLITS", sp),    ci("NBLK", nblk),
        cb("FINAL", final), cb("HAS_BASE", based), t.pdlConst(),       cb("WPF", wpf),
    });
    if (final) return;
    const merged = [_]aot.Arg{ p("PM", "*fp32", pp.pm), p("PL", "*fp32", pp.pl), p("PO", "*fp32", pp.po), p("SINK", "*fp32", a.sink), p("OUT", "*bf16", a.out) };
    if (a.rot) |r| {
        // switch "merge_reg" (on in the served build): the head's row kept in registers when HD is a power of two
        const merge: []const u8 = if (isPow2(a.hd)) "_sparse_attn_merge_rot2" else "_sparse_attn_merge_rot";
        // TF_EXL3_L2_DISCARD unset in the served build (on): the merge drops the split outputs it read from L2
        const discard = a.hd % 32 == 0 and isPow2(sp * (a.hd / 32));
        const args = merged ++ [_]aot.Arg{ p("COS", "*fp32", r.cos), p("SIN", "*fp32", r.sin), p("POS", "*i64", a.pos), p("SUH", "*fp16", r.suh), p("XH", "*fp16", r.xh), aot.int("rows", @intCast(a.rows)) };
        try t.run(merge, .{ u(a.rows), u(a.h), 1 }, &args, &.{
            ci("H", a.h), ci("HD", a.hd), ci("HB", hb), ci("SPLITS", sp), ci("RD", r.rd), ci("GH", r.gh), cb("DISCARD", discard), t.pdlConst(),
        });
    } else {
        try t.run("_sparse_attn_merge", .{ u(a.rows), u(groups), 1 }, &merged, &.{ ci("H", a.h), ci("HD", a.hd), ci("HB", hb), ci("SPLITS", sp), t.pdlConst() });
    }
}

/// fp4_store: model.store_rows(cache, rows, x, block, e4m3) of a packed cache (codes, scales); x [n, d] bf16, row stride xs.
pub fn fp4Store(t: Tri, x: u64, xs: usize, codes: u64, scales: u64, rows: u64, n: usize, d: usize, block: usize, e4m3: bool) !void {
    try t.run("_fp4_store", .{ u(n), 1, 1 }, &.{ p("X", "*bf16", x), aot.int("xs", @intCast(xs)), p("CODES", "*u8", codes), p("SCALES", "*u8", scales), p("ROWS", "*i64", rows) }, &.{ ci("D", d), ci("BLOCK", block), cb("E4M3", e4m3), t.pdlConst() });
}

/// fp4_qd_p2: ops.fp4_qd(x, 32, e4m3_scale=False) of bf16 x (numel elements, a multiple of 32) into out, x's shape.
pub fn fp4QdP2(t: Tri, x: u64, out: u64, numel: usize) !void {
    const nblk = numel / 32;
    const blks = 32;
    try t.run("_fp4qd_p2", .{ tri.cdiv(nblk, blks), 1, 1 }, &.{ p("X", "*bf16", x), p("OUT", "*bf16", out), aot.int("nblk", @intCast(nblk)) }, &.{ ci("BLKS", blks), t.pdlConst() });
}

// -- conformance with the recorded launches ----------------------------------------------------------------------

const Case = tri.Case;

test "TP3 attention processes all 24 heads with three complete tiles" {
    for ([_]usize{ 1, 4096 }) |rows| {
        var log: tri.Log = .{ .gpa = std.testing.allocator };
        defer log.deinit();
        try sparseAttn(.{ .log = &log }, .{
            .q = 16, .out = 32, .rows = rows, .h = 24, .hd = 512,
            .sink = 48, .wsrc = 64, .wsrc_rows = 128, .wlo = 80,
            .ring = true, .pos = 96, .scale = 0.044194174, .window = 128,
            .parts = .{ .pm = 112, .pl = 128, .po = 144 },
        });
        for (log.items.items) |launch| {
            try std.testing.expectEqual(@as(u32, 3), launch.grid[1]);
            var found = false;
            for (launch.consts) |c| if (std.mem.eql(u8, c.name, "HB")) {
                try std.testing.expectEqual(@as(?i64, 8), c.int);
                found = true;
            };
            try std.testing.expect(found);
        }
        try std.testing.expectEqual(@as(usize, if (rows == 1) 2 else 1), log.items.items.len);
    }
}

fn dtype(c: Case, tensor: []const u8) []const u8 {
    return c.v.get("tensors").?.object.get(tensor).?.array.items[0].string;
}

test "sparse_attn's split keys and prompt chunks launch as recorded" {
    try tri.conform("_sparse_attn_part", struct {
        fn call(t: Tri, c: Case) !void {
            // a pick list with keys was given when IDX is [R, n]; else the kernel reads pos there
            const has = c.rank("IDX") == 2;
            const fp4 = std.mem.eql(u8, dtype(c, "COMP"), "uint8");
            const comp: Comp = if (!has) .none else if (fp4) .{ .fp4 = .{ .codes = c.ptr("COMP"), .scales = c.ptr("CSC") } } else .{ .bf16 = c.ptr("COMP") };
            const ring = c.constInt("RING") != 0;
            // wbase given or not: no tensor tells (RBASE and CBASE are pos otherwise), so HAS_BASE from the case
            const based = c.constInt("HAS_BASE") != 0;
            // the caller's ring_rows (a pool's ring size, not wsrc's rows): the recorded ring_size where it differs from them
            const rs: usize = @intCast(c.int("ring_size"));
            try sparseAttn(t, .{
                .q = c.ptr("Q"),
                .out = c.ptr("OUT"),
                .rows = c.dim("Q", 0),
                .h = c.dim("Q", 1),
                .hd = c.dim("Q", 2),
                .sink = c.ptr("SINK"),
                .wsrc = c.ptr("WSRC"),
                .wsrc_rows = c.dim("WSRC", 0),
                .wlo = c.ptr("WLO"),
                .ring = ring,
                .comp = comp,
                .idx = if (has) c.ptr("IDX") else 0,
                .n_idx = if (has) c.dim("IDX", 1) else 0,
                .pos = c.ptr("POS"),
                .scale = c.float("scale"),
                .window = @intCast(c.constInt("WIN")),
                .wbase = if (based) c.ptr("RBASE") else null,
                .cbase = if (based) c.ptr("CBASE") else null,
                .ring_rows = if (ring and rs != c.dim("WSRC", 0)) rs else 0,
                .parts = .{ .pm = c.ptr("PM"), .pl = c.ptr("PL"), .po = c.ptr("PO") },
            });
        }
    }.call);
}

/// sparse_attn's arguments for a recorded merge: the merge's from the case, the split pass's own (not recorded) fixed.
fn mergeArgs(c: Case, rot: ?Rot) Attn {
    return .{
        .q = 0x7e00_0000_0000,
        .out = c.ptr("OUT"),
        .rows = c.dim("OUT", 0),
        .h = c.dim("OUT", 1),
        .hd = c.dim("OUT", 2),
        .sink = c.ptr("SINK"),
        .wsrc = 0x7e00_0000_0100,
        .wsrc_rows = 576,
        .wlo = 0x7e00_0000_0200,
        .ring = true,
        .pos = if (c.has("POS")) c.ptr("POS") else 0x7e00_0000_0300,
        .scale = 0.044194173,
        .window = 128,
        .ring_rows = 144,
        .rot = rot,
        .parts = .{ .pm = c.ptr("PM"), .pl = c.ptr("PL"), .po = c.ptr("PO") },
    };
}

test "sparse_attn's merges launch as recorded" {
    try tri.conform("_sparse_attn_merge", struct {
        fn call(t: Tri, c: Case) !void {
            try sparseAttn(t, mergeArgs(c, null));
        }
    }.call);
    try tri.conform("_sparse_attn_merge_rot2", struct {
        fn call(t: Tri, c: Case) !void {
            const rot: Rot = .{ .cos = c.ptr("COS"), .sin = c.ptr("SIN"), .rd = @intCast(c.constInt("RD")), .suh = c.ptr("SUH"), .xh = c.ptr("XH"), .gh = @intCast(c.constInt("GH")) };
            try sparseAttn(t, mergeArgs(c, rot));
        }
    }.call);
}

test "a prompt chunk's pick list is padded to 16 entries and its pick blocks to a power of two" {
    var log: tri.Log = .{ .gpa = std.testing.allocator };
    defer log.deinit();
    const comp: Comp = .{ .fp4 = .{ .codes = 0x600, .scales = 0x700 } };
    try sparseAttn(.{ .log = &log }, .{ .q = 0x100, .out = 0x200, .rows = 17, .h = 32, .hd = 512, .sink = 0x300, .wsrc = 0x400, .wsrc_rows = 17, .wlo = 0x500, .ring = false, .comp = comp, .idx = 0x800, .n_idx = 70, .pos = 0x900, .scale = 0.044194173, .window = 128 });
    try std.testing.expectEqual(@as(usize, 1), log.items.items.len);
    const k = log.items.items[0];
    const n_idx = for (k.args) |x| {
        if (std.mem.eql(u8, x.name, "n_idx")) break x.value.i32;
    } else return error.TestUnexpectedResult;
    const nblk = for (k.consts) |x| {
        if (std.mem.eql(u8, x.name, "NBLK")) break x.int;
    } else return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(i32, 80), n_idx);
    try std.testing.expectEqual(@as(?i64, 128 / 32 + 4), nblk);
}

test "fp4_store launches as recorded" {
    try tri.conform("_fp4_store", struct {
        fn call(t: Tri, c: Case) !void {
            try fp4Store(t, c.ptr("X"), c.stride("X", 0), c.ptr("CODES"), c.ptr("SCALES"), c.ptr("ROWS"), c.dim("X", 0), c.dim("X", 1), @intCast(c.constInt("BLOCK")), c.constInt("E4M3") != 0);
        }
    }.call);
}

test "fp4_qd_p2 launches as recorded" {
    try tri.conform("_fp4qd_p2", struct {
        fn call(t: Tri, c: Case) !void {
            var numel: usize = 1;
            for (0..c.rank("X")) |i| numel *= c.dim("X", i);
            try fp4QdP2(t, c.ptr("X"), c.ptr("OUT"), numel);
        }
    }.call);
}
