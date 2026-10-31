//! CPU-only native launch matrix. No CUDA allocation, driver, model or network.
//! This covers the named wrappers, not a complete model execution trace.
const std = @import("std");
const ds = @import("dsv41");
const cuda = @import("cuda");
const tri = ds.tri;
const hc = ds.tri_hc;
const attn = ds.tri_attn;
const norm = ds.tri_norm;
const index = ds.tri_index;
const markov = ds.tri_markov;

fn fake(n: u64) u64 {
    return 0x100000 + n * 0x100;
}

// CUDA top-k is outside this Triton manifest audit. Its callback must be
// inert here; we still collect every subsequent Triton selection launch.
fn skipCudaTopk(_: ?*anyopaque, _: tri.Tri, _: u64, _: usize, _: usize, _: usize, _: usize, _: u64) !void {}

fn recordIndexer(t: tri.Tri, rows: usize, prefill: bool) !void {
    const k: index.IndexK = .{ .fp4 = .{ .codes = fake(0), .scales = fake(1) } };
    const top: index.TopK = .{ .top = fake(2), .run = skipCudaTopk };
    const base: ?u64 = if (prefill) null else fake(3);
    for ([_]usize{ 1, 64, 512, 4096, 4097, 16384, 32768, 65536, 131072, 262144, 401408, 524288 }) |n| {
        const kk = @min(n, 512);
        try index.indexScore(t, fake(4), k, fake(5), fake(6), n, fake(7), base, false, null, false, rows, 32, 128);
        try index.scoreKeys(t, fake(7), fake(8), fake(9), rows, n);
        try index.topkSelectPruned(t, fake(9), n, fake(8), kk, fake(6), fake(10), .{ .tpos = fake(11), .cand = fake(12), .every = fake(13) }, top, rows, n);
        if (!prefill) {
            for ([_]bool{ false, true }) |with_cand| {
                try index.indexKeys(t, fake(4), k, fake(5), fake(6), n, base, if (with_cand) .{ .mask = fake(14), .stride = (n + 7) / 8 } else null, 8, fake(8), kk, fake(9), rows, 32, 128);
            }
            try index.indexKeysCand(t, fake(4), k, fake(5), fake(6), n, fake(15), 2048, 2048, 8, base, fake(9), rows, 32, 128);
        }
    }
}

fn record(t: tri.Tri, rows: usize, prefill: bool) !void {
    // Actual TP3 router profile: 384 experts, TOPK=6, shared id=384.
    // C8/K5 verification can use any 1..48 rows, including 40. Prefill
    // projects by the separate dense path audited with projection-shapes.
    if (!prefill) {
        _ = try norm.rowmmGate(t, fake(0), 5120, fake(1), fake(2), rows, 5120, 384);
        try norm.rowmmWts(t, fake(0), 5120, fake(1), 0.001, fake(2), rows, 5120, 32);
        try norm.route(t, fake(0), 20, fake(1), 6, 2.5, 384, fake(2), fake(3), rows, 384, 7);
    }
    for ([_]bool{ false, true }) |posted| {
        const ps: ?hc.Posted = if (posted) .{ .gathered = fake(20), .world = 2, .h_out = fake(21) } else null;
        if (prefill) {
            _ = try hc.hcPrePf(t, fake(0), fake(1), fake(2), fake(3), fake(4), fake(5), 1e-6, 1e-6, 20, fake(6), fake(7), fake(8), fake(9), fake(10), ps, rows, 5120);
        } else for ([_]bool{ false, true }) |sink| {
            _ = try hc.hcPre2(t, fake(0), fake(1), fake(2), fake(3), fake(4), fake(5), 1e-6, 1e-6, 20, fake(6), fake(7), fake(8), fake(9), fake(10), ps, if (sink) .{ .s = undefined, .fork = undefined } else null, rows, 5120);
        }
    }
    for ([_]bool{ false, true }) |inv| try norm.ropeHeads(t, fake(0), fake(1), fake(2), fake(3), 64, inv, rows, 16, 512);
    for ([_]usize{ 0, 1, 16, 32, 128, 512, 1024 }) |nidx| {
        for ([_]bool{ false, true }) |is_packed| {
            if (nidx == 0 and is_packed) continue;
            try attn.sparseAttn(t, .{
                .q = fake(0),
                .out = fake(1),
                .rows = rows,
                .h = 16,
                .hd = 512,
                .sink = fake(2),
                .wsrc = fake(3),
                .wsrc_rows = 4096,
                .wlo = fake(4),
                .ring = !prefill,
                .comp = if (nidx == 0) .none else if (is_packed) .{ .fp4 = .{ .codes = fake(5), .scales = fake(6) } } else .{ .bf16 = fake(5) },
                .idx = fake(7),
                .n_idx = nidx,
                .pos = fake(8),
                .scale = 0.04419417382415922,
                .window = 128,
                .wbase = if (prefill) null else fake(9),
                .cbase = if (prefill) null else fake(10),
                .ring_rows = 128,
                .parts = .{ .pm = fake(11), .pl = fake(12), .po = fake(13) },
                // round.zig fuses inverse RoPE and wo_a input rotation;
                // padded H=24 / three local groups => GH=8.
                .rot = if (prefill) null else .{ .cos = fake(14), .sin = fake(15), .rd = 64, .suh = fake(16), .xh = fake(17), .gh = 8 },
                .one_split = prefill,
            });
        }
    }
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 2 and args.len != 3) return error.ExpectedAotJsonPath;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], a, .limited(1 << 26));
    var manifest = try cuda.aot.Manifest.parse(a, bytes);
    defer manifest.deinit();
    var log: tri.Log = .{ .gpa = a };
    defer log.deinit();
    const t: tri.Tri = .{ .log = &log };
    tri.decode_rows = 48; // C8, target + K5 verification. Restore in normal runtime.
    for (1..49) |rows| try record(t, rows, false);
    for ([_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 31, 32, 63, 64, 128, 256, 512, 1024, 4095, 4096 }) |rows| try record(t, rows, true);
    for ([_]usize{ 1, 2, 3, 8, 16, 24, 32, 40, 48 }) |rows| try recordIndexer(t, rows, false);
    for ([_]usize{ 1, 16, 48, 49, 128, 4096 }) |rows| try recordIndexer(t, rows, true);
    var projections: usize = 0;
    if (args.len == 3) {
        const shape_bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], a, .limited(1 << 22));
        const ShapeFile = struct { projections: []struct { k: usize, n: usize } };
        const shapes = try std.json.parseFromSlice(ShapeFile, a, shape_bytes, .{ .ignore_unknown_fields = true });
        defer shapes.deinit();
        projections = shapes.value.projections.len;
        if (projections == 0) return error.EmptyProjectionShapes;
        for (shapes.value.projections) |shape| {
            if (shape.k == 0 or shape.n == 0 or shape.k % 128 != 0 or shape.n % 128 != 0) return error.BadProjectionShape;
            for ([_]usize{ 129, 256, 1024, 4096 }) |rows| {
                for ([_]markov.OutType{ .bf16, .fp32 }) |ty| {
                    try markov.matmul(t, fake(0), fake(1), fake(2), fake(3), null, fake(4), ty, shape.n, rows, shape.k, shape.n);
                }
            }
        }
    }
    // Includes odd tails, the three 2048-row shards, and compressed-1024 shards.
    for ([_]usize{ 1, 16, 32, 64, 128, 256, 512, 1008, 1023, 1024 }) |rows| {
        for ([_][2]usize{ .{5120,1280}, .{5120,512}, .{1280,4096}, .{512,128} }) |shape| {
            for ([_]ds.tri_markov.OutType{ .bf16, .fp32 }) |ty|
                try ds.tri_markov.matmul(t, fake(0), fake(1), fake(2), fake(3), null, fake(4), ty, shape[1], rows, shape[0], shape[1]);
        }
        try ds.tri_norm.ropeHeads(t, fake(0), fake(1), fake(2), fake(3), 64, false, rows, 32, 128);
        try ds.tri_basic.rmsnorm(t, fake(0), 1280, fake(1), fake(2), 1280, 1e-20, rows, 1280);
        try ds.tri_attn.fp4QdP2(t, fake(0), fake(1), rows * 32 * 128);
        const top: ds.tri_index.TopK = .{ .top = fake(2), .run = skipCudaTopk };
        const k: ds.tri_index.IndexK = .{ .fp4 = .{ .codes = fake(0), .scales = fake(1) } };
        for ([_]usize{ 512, 513, 4096, 4097, 16384, 16385, 65536, 131072, 200704 }) |n| {
            for ([_]bool{ false, true }) |keyed| {
                try ds.tri_index.indexScore(t, fake(2), k, fake(3), fake(4), n, fake(5), null, keyed,
                    if (keyed) fake(6) else null, false, rows, 32, 128);
            }
            try ds.tri_index.topkSelectPruned(t, fake(0), n, fake(1), 512, fake(2), fake(3),
                .{ .tpos = fake(4), .cand = fake(5), .every = fake(6) }, top, rows, n);
        }
    }
    for ([_]usize{ 1, 16, 32, 64, 128, 256, 512, 1008, 1024 }) |rows| {
        try ds.tri_basic.embedInit(t, fake(0), fake(1), fake(2), fake(3), rows, 5120, 4);
        try ds.tri_basic.hcPre(t, fake(0), fake(1), fake(2), fake(3), fake(4), fake(5),
            1e-20, 1e-6, 20, fake(6), fake(7), fake(8), fake(9), fake(10), rows, 5120);
        _ = try ds.tri_hc.hcPrePf(t, fake(0), fake(1), fake(2), fake(3), fake(4), fake(5),
            1e-20, 1e-6, 20, fake(6), fake(7), fake(8), fake(9), fake(10),
            .{ .gathered = fake(11), .world = 2, .h_out = fake(12) }, rows, 5120);
        try ds.tri_basic.hcPost(t, fake(0), fake(1), fake(2), fake(3), fake(1), 3, rows, 5120);
        try ds.tri_basic.engramGate(t, fake(0), fake(1), fake(2), fake(3), 1e-20, rows, 5120);
    }

    // The TP4 head uses uneven 128-column vocabulary segments, unlike
    // the historical TP2 Markov launch (BP=64). Exercise the actual wrapper
    // for all quarters and every adaptive draft position before GPU load.
    for (0..4) |g| {
        var two: ds.prompt2d.Two = undefined;
        two.r = g % 2;
        two.p = g / 2;
        two.hw = .{ 253 * 128, 252 * 128 };
        var cfg: ds.Config = undefined;
        cfg.markov_rank = 256;
        const m = try ds.draft2d.markov(cfg, &two, fake(0), fake(1), fake(2));
        const scratch: markov.Scratch = .{ .pv = fake(3), .pi = fake(4), .send = fake(5), .stage = fake(6) };
        for ([_]usize{ 1, 3, 8, 12, 16 }) |rows| {
            for (0..5) |i| {
                _ = try m.localBest(t, fake(7), fake(8), 6, rows, 6, i, scratch);
                try m.pick(t, fake(9), 4, fake(8), 6, rows, i);
            }
        }
    }

    var missing: std.ArrayList(tri.Log.Kept) = .empty;
    var dense_launches: std.ArrayList(tri.Log.Kept) = .empty;
    for (log.items.items) |k| if (std.mem.eql(u8, k.name, "_gemm")) try dense_launches.append(a, k);
    // The scheduler takes views into i64 position arrays. Odd row offsets
    // keep natural i64 alignment but lose Triton's optional div16 guarantee.
    // Audit the resolver with actual wrapper arguments; no GPU allocation.
    var position_view_launches: usize = 0;
    for (log.items.items) |k| {
        if (!manifest.covers(k.name, k.args, k.consts)) try missing.append(a, k);
        const viewed = try a.dupe(cuda.aot.Arg, k.args);
        var has_position = false;
        for (viewed) |*arg| {
            if ((std.mem.eql(u8, arg.name, "POS") or std.mem.eql(u8, arg.name, "RBASE") or std.mem.eql(u8, arg.name, "CBASE") or std.mem.eql(u8, arg.name, "IDX") or std.mem.eql(u8, arg.name, "TOK") or std.mem.eql(u8, arg.name, "OUT")) and arg.value == .ptr and std.mem.eql(u8, arg.value.ptr.ty, "*i64")) {
                arg.value.ptr.addr += 8;
                has_position = true;
            }
        }
        if (has_position) {
            position_view_launches += 1;
            if (!manifest.covers(k.name, viewed, k.consts)) try missing.append(a, .{ .name = k.name, .grid = k.grid, .args = viewed, .consts = k.consts });
        }
    }
    var buf: [65536]u8 = undefined;
    var out = std.Io.File.stdout().writer(init.io, &buf);
    const json = try std.json.Stringify.valueAlloc(a, .{
        .status = if (missing.items.len == 0) "MATRIX_COVERED_NOT_GPU_QUALIFIED" else "MISSING_VARIANTS",
        .scope = "mHC, main RoPE, sparse attention, indexer, Triton top-k and optional native-plan dense prefill GEMM; CUDA launches excluded",
        .dense_projection_shapes = projections,
        .dense_launches = dense_launches.items,
        .launches = log.items.items.len,
        .position_view_launches = position_view_launches,
        .missing = missing.items,
    }, .{});
    try out.interface.writeAll(json);
    try out.interface.writeAll("\n");
    try out.interface.flush();
    return if (missing.items.len == 0) 0 else 2;
}
