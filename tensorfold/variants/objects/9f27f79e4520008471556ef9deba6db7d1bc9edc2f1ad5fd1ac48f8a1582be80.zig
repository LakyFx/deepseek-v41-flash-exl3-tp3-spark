//! tf-dsv41-lanes MODEL_DIR CACHE_DIR RANK WORLD MASTER_IP PORT KIT_DIR --requests FILE [--parallel N] [--pool P]
//!   [--drafts 0|1] [--arena-gib G] [--engram DIR --token-map FILE] [--rdma DEVICES --rdma-kernels FATBIN]: the lane
//! gate of the Zig port. TensorFold 1.0's lane core (core/lanes: its round loop, depth rule, window planning, copies,
//! acceptance) serves the requests through the port's Backend (lanes.zig) on every rank: rank 0 runs the core and
//! admits up to N requests at a time (the host's SuffixLookup proposer on each, min_match 4), rank 1 follows (WORLD 4:
//! the four nodes of the 2D split, ranks 1-3 following; --rdma: its decode-size exchanges over the RDMA rings, as the
//! layer gate's, their kernels from --rdma-kernels or the embedded image). Each reply's token_sha (sha256 of its comma-joined ids,
//! the served server's) against the served reply's: the drafted, concurrent reply must be the served one. One JSON
//! line a request, then the summary. FILE: zrec_lanereq.py's requests (prompt ids, max_tokens, sampling, expect_sha).
//! KIT_DIR: a gate dir's aot/, cubins/, RoPE tables (as many rows as the pool) and engram.json.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const lanes = @import("lanes");

const usage = "usage: tf-dsv41-lanes MODEL_DIR CACHE_DIR RANK WORLD MASTER_IP PORT KIT_DIR --requests FILE [--parallel N] [--pool P] [--drafts 0|1] [--arena-gib G] [--engram DIR --token-map FILE] [--rdma DEVICES --rdma-kernels FATBIN] [--fix-k K] [--round-rows R] [--streams S]\n";

const Request = struct {
    name: []const u8 = "",
    prompt: []const u32,
    max_tokens: u32,
    seed: ?u64 = null,
    temperature: f64 = 0,
    top_k: u32 = 20,
    top_p: f64 = 0.95,
    expect_sha: ?[]const u8 = null,
    serial: bool = false, // run this request without drafts, alone
    alone: bool = false, // single-stream admission while keeping drafts enabled
    expect_request: ?usize = null, // compare to an earlier completed serial request
    images: []const ImageRows = &.{}, // a vision request: its spans' rows ready (zrec_v3req.py from a recording)
};

/// An image span of a request's prompt and its rows: bf16 [tokens, hidden] from row `row` of `rows_file`.
const ImageRows = struct { at: u32, tokens: u32, rows_file: []const u8, row: u32 = 0 };

const Job = struct {
    req: *const Request,
    index: usize,
    proposer: lanes.SuffixLookup,
    stream: lanes.Stream,
    began: std.Io.Timestamp,
    done: bool = false,
};

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 8) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var o: dsv41.model.Options = .{
        .model_dir = args[1],
        .cache_dir = args[2],
        .rank = try std.fmt.parseInt(u32, args[3], 10),
        .world = try std.fmt.parseInt(u32, args[4], 10),
        .master = try dsv41.link.parseIp(args[5]),
        .port = try std.fmt.parseInt(u16, args[6], 10),
        .kit_dir = args[7],
    };
    var requests_file: ?[]const u8 = null;
    var parallel: ?usize = null; // default: every stream of the pool
    var arena_gib: ?usize = null;
    var profile = false;
    // TF_DS_CLOCK_MARKS=1 with --profile: a GPU clock at every round phase mark (in the graphs too; it splits their
    // PDL chains, so the stretch totals run long), else only around the stretches
    const clock_marks: u8 = if (std.c.getenv("TF_DS_CLOCK_MARKS")) |v| std.fmt.parseInt(u8, std.mem.span(v), 10) catch 0 else 0;
    var phases = false; // --profile 2: also the rounds' GPU time by phase (eager rounds)
    var serial = false; // --serial 1: every stream without drafts (one row a round)
    var fix_k: ?usize = null; // --fix-k K: every drafted stream verifies K drafts a round (a cost measurement)
    var ai: usize = 8;
    while (ai + 1 < args.len) : (ai += 2) {
        const key = args[ai];
        const val = args[ai + 1];
        if (std.mem.eql(u8, key, "--requests")) {
            requests_file = val;
        } else if (std.mem.eql(u8, key, "--parallel")) {
            parallel = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--round-rows")) {
            o.round_rows = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--round-ms")) {
            o.round_ms = try dsv41.lanes.parseMs(a, val);
        } else if (std.mem.eql(u8, key, "--streams")) {
            o.streams = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--pool")) {
            o.pool = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--drafts")) {
            o.drafts = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--arena-gib")) {
            arena_gib = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--aio")) {
            o.engram_aio = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--prefetch")) {
            o.prefetch = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--side")) {
            o.side = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--graphs")) {
            o.graphs = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--serial")) {
            serial = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--fix-k")) {
            fix_k = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--rdma")) {
            o.rdma_devices = val;
        } else if (std.mem.eql(u8, key, "--profile")) {
            profile = !std.mem.eql(u8, val, "0");
            phases = std.mem.eql(u8, val, "2");
        } else if (std.mem.eql(u8, key, "--engram")) {
            o.engram_dir = val;
        } else if (std.mem.eql(u8, key, "--token-map")) {
            o.token_map = val;
        } else if (std.mem.eql(u8, key, "--rdma-kernels")) {
            o.rdma_kernels = val;
        } else if (std.mem.eql(u8, key, "--vision-bias")) {
            o.vision_bias = val;
        } else if (std.mem.eql(u8, key, "--logits-sha")) {
            dsv41.lanes.logits_sha = !std.mem.eql(u8, val, "0");
        } else return error.BadArgument;
    }
    const par = parallel orelse o.streams;
    if (par == 0 or par > o.streams or o.streams > dsv41.model.max_streams) return error.BadArgument;
    o.arena_bytes = (arena_gib orelse 0) << 30; // (0: the arena grows as the buffers ask)

    // Validate paired requests before opening CUDA or loading weights on rank0.
    const request_pairs = @import("request_pairs.zig");
    const Requests = struct { requests: []Request };
    const reqs: []const Request = if (o.rank == 0) blk: {
        const text = try std.Io.Dir.cwd().readFileAlloc(io, requests_file orelse return error.NoRequests, a, .limited(1 << 30));
        break :blk (try std.json.parseFromSliceLeaky(Requests, a, text, .{ .ignore_unknown_fields = true })).requests;
    } else &.{};
    const planned_pairs = try request_pairs.validate(reqs, serial, o.drafts);

    var out_buf: [1 << 14]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &out_buf);
    const w = &out.interface;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const t0 = std.Io.Timestamp.now(io, .awake);
    const m = try dsv41.model.Model.open(gpa, io, &ctx, o);
    defer m.close();
    try w.print("{{\"rank\": {d}, \"loaded_s\": {d:.1}, \"pool\": {d}, \"drafts\": {}, \"arena_used\": {d}, \"arena_reserved\": {d}}}\n", .{ o.rank, seconds(t0, io), o.pool, m.drafting(), m.arena.used, m.arena.reserved });
    try w.flush();

    if (o.rank != 0) {
        if (profile) {
            m.prof = true;
            if (m.eh) |*eh| eh.io = io;
            if (phases) try m.usePhaseTimer();
            try m.useClock(clock_marks);
        }
        defer if (profile) {
            const fp = dsv41.lanes.follow_prof;
            const r: f64 = @floatFromInt(@max(fp.rounds, 1));
            w.print("{{\"rank\": {d}, \"follow_profile\": {{\"rounds\": {d}, \"enqueue_ms\": {d:.2}, \"forward_ms\": {d:.2}", .{ o.rank, fp.rounds, @as(f64, @floatFromInt(fp.enqueue)) / r / 1e6, @as(f64, @floatFromInt(fp.forward)) / r / 1e6 }) catch {};
            if (m.eh) |eh| if (eh.p_calls > 0) {
                const pc: f64 = @floatFromInt(eh.p_calls);
                w.print(", \"prefill_engram_ms\": {{\"calls\": {d}, \"rows\": {d:.2}, \"upload\": {d:.2}, \"mm\": {d:.2}, \"exchange\": {d:.2}, \"gate\": {d:.2}}}", .{ eh.p_calls, @as(f64, @floatFromInt(eh.p_rows)) / pc / 1e6, @as(f64, @floatFromInt(eh.p_upload)) / pc / 1e6, @as(f64, @floatFromInt(eh.p_mm)) / pc / 1e6, @as(f64, @floatFromInt(eh.p_x)) / pc / 1e6, @as(f64, @floatFromInt(eh.p_gate)) / pc / 1e6 }) catch {};
            };
            if (m.eh) |eh| w.print(", \"engram_read_ms\": {d:.3}, \"engram_hash_ms\": {d:.3}", .{ @as(f64, @floatFromInt(eh.t_read)) / r / 1e6, @as(f64, @floatFromInt(eh.t_hash)) / r / 1e6 }) catch {};
            w.print("}}}}\n", .{}) catch {};
            printClock(w, o.rank, m) catch {};
            if (m.ptimer) |tm| {
                w.print("{{\"rank\": {d}, \"prefill_phases_ms\": {{", .{o.rank}) catch {};
                for (dsv41.round.PhaseTimer.phases, 0..) |name, i| w.print("{s}\"{s}\": {d:.3}", .{ if (i == 0) "" else ", ", name, if (tm.rounds == 0) 0 else tm.ms[i] / @as(f64, @floatFromInt(tm.rounds)) }) catch {};
                w.print("}}, \"chunks\": {d}}}\n", .{tm.rounds}) catch {};
            }
            w.flush() catch {};
        };
        dsv41.lanes.follow(gpa, m) catch |err| {
            try w.print("{{\"rank\": {d}, \"follow\": \"{s}\"}}\n", .{ o.rank, @errorName(err) });
            try w.flush();
            return 1;
        };
        try w.print("{{\"rank\": {d}, \"followed\": true}}\n", .{o.rank});
        try w.flush();
        return 0;
    }

    // rank 0: the requests, the backend and the lane core
    var ln = dsv41.lanes.Lanes.init(gpa, m);
    defer ln.deinit();
    ln.fix_k = fix_k;
    if (std.c.getenv("TF_DS_ADAPTIVE_LEARNING")) |value| {
        const text = std.mem.span(value);
        if (!std.mem.eql(u8, text, "0") and !std.mem.eql(u8, text, "1")) return error.BadAdaptiveLearning;
        ln.adaptive_learning = std.mem.eql(u8, text, "1");
    }
    if (profile) {
        ln.prof = .{};
        m.prof = true;
        if (m.eh) |*eh| eh.io = io;
        if (phases) try m.usePhaseTimer();
        try m.useClock(clock_marks);
    }
    var step_ns: u64 = 0;
    const rows: u32 = @intCast(m.eng.round_rows);
    var cfg = try lanes.Config.init(gpa, ln.facts(), rows, rows - 1);
    defer cfg.deinit(gpa);
    var clock: lanes.backend.WallClock = .{ .io = io };
    var core = lanes.Engine.init(gpa, &cfg, ln.backend(), clock.clock());
    defer core.deinit();

    const eos = [_]u32{@intCast(m.cfg.eos)};
    const jobs = try a.alloc(Job, reqs.len);
    const completed = try a.alloc(bool, reqs.len);
    @memset(completed, false);
    const reply_hashes = try a.alloc(?[]const u8, reqs.len);
    @memset(reply_hashes, null);
    const reply_reasons = try a.alloc(?[]const u8, reqs.len);
    @memset(reply_reasons, null);
    var pairs_checked: usize = 0;
    var pairs_with_drafts: usize = 0;
    var serial_running: ?usize = null;
    var admitted: usize = 0;
    var finished: usize = 0;
    var all_equal = true;
    var tokens_out: usize = 0;
    const started = std.Io.Timestamp.now(io, .awake);
    while (finished < reqs.len) {
        // admit while a lane is free (the host's admission: one prompt pass at a time, before the rounds)
        while (admitted < reqs.len and core.activeCount() < par) {
            if (serial_running) |ref| if (!completed[ref]) break;
            const r = &reqs[admitted];
            if (!request_pairs.ready(r.serial or r.alone, r.expect_request, completed, core.activeCount())) break;
            const j = &jobs[admitted];
            j.* = .{ .req = r, .index = admitted, .proposer = try lanes.SuffixLookup.init(gpa, .{ .min_match = 4 }), .stream = undefined, .began = std.Io.Timestamp.now(io, .awake) };
            j.stream = try lanes.Stream.init(gpa, .{
                .id = r.name,
                .prompt = r.prompt,
                .images = try imageRows(a, io, r.images, m.cfg.hidden),
                .max_new = r.max_tokens,
                .eos = &eos,
                .sampling = if (r.temperature > 0) .{ .seed = r.seed orelse 0, .temperature = r.temperature, .top_k = r.top_k, .top_p = r.top_p } else null,
                .drafts = !serial and !r.serial,
                .proposer = j.proposer.proposer(),
            });
            admitted += 1;
            if (r.serial or r.alone) serial_running = j.index;
            core.addStream(&j.stream) catch |err| {
                try w.print("{{\"rank\": 0, \"request\": {d}, \"name\": \"{s}\", \"error\": \"{s}\"}}\n", .{ j.index, r.name, @errorName(err) });
                try w.flush();
                return 1;
            };
            try w.print("{{\"rank\": 0, \"request\": {d}, \"name\": \"{s}\", \"prompt_tokens\": {d}, \"prefill_s\": {d:.2}}}\n", .{ j.index, r.name, r.prompt.len, seconds(j.began, io) });
            try w.flush();
            if (r.serial or r.alone) break; // preserve a truly single-stream request
        }
        const s0 = m.now();
        if (core.live.items.len > 0) core.step() catch |err| {
            try w.print("{{\"rank\": 0, \"step\": {d}, \"error\": \"{s}\"}}\n", .{ core.steps, @errorName(err) });
            try w.flush();
            return 1;
        };
        step_ns += m.now() - s0;
        // the finished streams' replies
        for (jobs[0..admitted]) |*j| {
            if (j.done or !j.stream.finished) continue;
            j.done = true;
            finished += 1;
            const reply = j.stream.emitted();
            tokens_out += reply.len;
            const sha = try tokenSha(a, reply);
            reply_hashes[j.index] = sha; // arena-owned; survives stream.deinit
            reply_reasons[j.index] = j.stream.reason.name();
            completed[j.index] = true;
            var equal = if (j.req.expect_sha) |e| std.mem.eql(u8, e, sha) else true;
            if (j.req.expect_request) |ref| {
                const baseline_sha = reply_hashes[ref] orelse return error.MissingReferenceOutput;
                const baseline_reason = reply_reasons[ref] orelse return error.MissingReferenceOutput;
                equal = equal and std.mem.eql(u8, baseline_sha, sha) and std.mem.eql(u8, baseline_reason, j.stream.reason.name());
                pairs_checked += 1;
                if (j.stream.drafted > 0) pairs_with_drafts += 1;
                try w.print("{{\"pair_request\":{d},\"reference_request\":{d},\"equal\":{},\"drafted\":{d}}}\n", .{ j.index, ref, equal, j.stream.drafted });
            }
            if (!equal) all_equal = false;
            try w.print("{{\"rank\": 0, \"request\": {d}, \"name\": \"{s}\", \"tokens\": {d}, \"reason\": \"{s}\", \"token_sha\": \"{s}\", \"expect_sha\": \"{s}\", \"equal\": {}, \"rounds\": {d}, \"drafted\": {d}, \"accepted\": {d}, \"min_rows\": {d}, \"seconds\": {d:.2}, \"reply\": [", .{ j.index, j.req.name, reply.len, j.stream.reason.name(), sha, j.req.expect_sha orelse "", equal, j.stream.rounds, j.stream.drafted, j.stream.accepted, j.stream.min_rows, seconds(j.began, io) });
            for (reply, 0..) |t, k| try w.print("{s}{d}", .{ if (k == 0) "" else ", ", t });
            try w.print("]}}\n", .{});
            try w.flush();
            j.stream.deinit(gpa);
            j.proposer.deinit();
        }
    }
    const total_s = seconds(started, io);
    if (ln.prof) |p| {
        const ms = struct {
            fn f(ns: u64, n: u64) f64 {
                return if (n == 0) 0 else @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(n)) / 1e6;
            }
        }.f;
        try w.print("{{\"rank\": 0, \"profile\": {{\"rounds\": {d}, \"rows_a_round\": {d:.2}, \"step_ms\": {d:.2}, \"send_ms\": {d:.3}, \"enqueue_ms\": {d:.2}, \"forward_ms\": {d:.2}, \"absorb_ms\": {d:.2}, \"logits_ms\": {d:.2}, \"sample_ms\": {d:.2}, \"passes\": {d}, \"streams_a_pass\": {d:.2}, \"pass_ms\": {d:.2}, \"prefills\": {d}, \"prefill_ms\": {d:.1}}}}}\n", .{ p.rounds, ms(p.rows * 1000000, p.rounds), ms(step_ns, core.steps), ms(p.send, p.rounds), ms(p.enqueue, p.rounds), ms(p.forward, p.rounds), ms(p.absorb, p.rounds), ms(p.logits, p.rounds), ms(p.sample, p.rounds), p.passes, ms(p.pass_streams * 1000000, p.passes), ms(p.pass, p.passes), p.prefills, ms(p.prefill, p.prefills) });
        try w.flush();
        try printClock(w, 0, m);
        if (m.eh) |eh| if (eh.p_calls > 0) {
            const pc: f64 = @floatFromInt(eh.p_calls);
            try w.print("{{\"rank\": 0, \"prefill_engram_ms\": {{\"calls\": {d}, \"rows\": {d:.2}, \"upload\": {d:.2}, \"mm\": {d:.2}, \"exchange\": {d:.2}, \"gate\": {d:.2}}}}}\n", .{ eh.p_calls, @as(f64, @floatFromInt(eh.p_rows)) / pc / 1e6, @as(f64, @floatFromInt(eh.p_upload)) / pc / 1e6, @as(f64, @floatFromInt(eh.p_mm)) / pc / 1e6, @as(f64, @floatFromInt(eh.p_x)) / pc / 1e6, @as(f64, @floatFromInt(eh.p_gate)) / pc / 1e6 });
        };
        if (m.ptimer) |tm| {
            try w.print("{{\"rank\": 0, \"prefill_phases_ms\": {{", .{});
            for (dsv41.round.PhaseTimer.phases, 0..) |name, i| try w.print("{s}\"{s}\": {d:.3}", .{ if (i == 0) "" else ", ", name, if (tm.rounds == 0) 0 else tm.ms[i] / @as(f64, @floatFromInt(tm.rounds)) });
            try w.print("}}, \"chunks\": {d}}}\n", .{tm.rounds});
        }
        if (m.timer) |tm| {
            try w.print("{{\"rank\": 0, \"phases_ms\": {{", .{});
            for (dsv41.round.PhaseTimer.phases, 0..) |name, i| try w.print("{s}\"{s}\": {d:.3}", .{ if (i == 0) "" else ", ", name, if (tm.rounds == 0) 0 else tm.ms[i] / @as(f64, @floatFromInt(tm.rounds)) });
            try w.print("}}}}\n", .{});
            try w.flush();
        }
        if (m.eh) |eh| {
            try w.print("{{\"rank\": 0, \"engram\": {{\"calls_a_round\": {d:.2}, \"hash_ms\": {d:.3}, \"read_ms\": {d:.3}, \"decode_ms\": {d:.3}, \"upload_ms\": {d:.3}}}}}\n", .{ ms(eh.calls * 1000000, p.rounds), ms(eh.t_hash, p.rounds), ms(eh.t_read, p.rounds), ms(eh.t_decode, p.rounds), ms(eh.t_upload, p.rounds) });
            try w.flush();
        }
    }
    try w.print("{{\"rank\": 0, \"requests\": {d}, \"tokens\": {d}, \"steps\": {d}, \"shared_rounds\": {d}, \"drafted\": {d}, \"accepted\": {d}, \"seconds\": {d:.2}, \"all_equal\": {}}}\n", .{ reqs.len, tokens_out, core.steps, core.shared_rounds, core.drafted, core.accepted, total_s, all_equal });
    try w.flush();
    if (planned_pairs != 0) {
        const passed = all_equal and pairs_checked == planned_pairs and pairs_with_drafts == planned_pairs;
        try w.print("{{\"paired_status\":\"{s}\",\"planned\":{d},\"checked\":{d},\"with_drafts\":{d},\"model_quality_qualified\":false}}\n", .{ if (passed) "GREEDY_SERIAL_SPEC_MATCH" else "GREEDY_SERIAL_SPEC_FAILED", planned_pairs, pairs_checked, pairs_with_drafts });
        try w.flush();
        if (!passed) return 1;
    }
    return if (all_equal) 0 else 1;
}

fn seconds(t: std.Io.Timestamp, io: std.Io) f64 {
    return @as(f64, @floatFromInt(t.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e9;
}

/// A request's image spans with their rows read (the files kept for the run).
fn imageRows(a: std.mem.Allocator, io: std.Io, imgs: []const ImageRows, hidden: usize) ![]const lanes.stream.Image {
    const out = try a.alloc(lanes.stream.Image, imgs.len);
    for (imgs, out) |img, *o| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, img.rows_file, a, .limited(1 << 32));
        const row = hidden * 2;
        if ((@as(usize, img.row) + img.tokens) * row > bytes.len) return error.ShortImageRows;
        o.* = .{ .at = img.at, .tokens = img.tokens, .bytes = bytes[img.row * row ..][0 .. img.tokens * row], .rows = true };
    }
    return out;
}

/// The served server's token_sha: sha256 of the reply's ids joined by commas, its first 12 hex digits.
fn tokenSha(a: std.mem.Allocator, ids: []const u32) ![]const u8 {
    var text: std.ArrayList(u8) = .empty;
    var buf: [16]u8 = undefined;
    for (ids, 0..) |t, k| {
        if (k > 0) try text.append(a, ',');
        try text.appendSlice(a, try std.fmt.bufPrint(&buf, "{d}", .{t}));
    }
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text.items, &h, .{});
    const hex = try std.fmt.allocPrint(a, "{x}", .{&h});
    return hex[0..12];
}

/// A profile's round GPU clock gaps (us a round): start to the first stretch, then each stretch and the wait after it.
fn printClock(w: *std.Io.Writer, rank: u32, m: *const dsv41.model.Model) !void {
    if (m.clk_rounds == 0) return;
    const r: f64 = @floatFromInt(m.clk_rounds);
    try w.print("{{\"rank\": {d}, \"round_clock_us\": {{\"rounds\": {d}", .{ rank, m.clk_rounds });
    const names = dsv41.round.PhaseTimer.phases;
    for (0..256) |i| {
        if (m.clk_cnt[i] == 0) continue;
        const name = if (i < names.len) names[i] else switch (i) {
            dsv41.round.Round.clock_wait => "wait",
            dsv41.round.Round.clock_tail => "tail",
            else => "other",
        };
        if (i >= 64 and i < 128) {
            try w.print(", \"L{d}\": [{d:.1}, {d:.1}]", .{ i - 64, @as(f64, @floatFromInt(m.clk_ph[i])) / r / 1e3, @as(f64, @floatFromInt(m.clk_cnt[i])) / r });
            continue;
        }
        try w.print(", \"{s}\": [{d:.1}, {d:.1}]", .{ name, @as(f64, @floatFromInt(m.clk_ph[i])) / r / 1e3, @as(f64, @floatFromInt(m.clk_cnt[i])) / r });
    }
    try w.print("}}}}\n", .{});
    try w.print("{{\"rank\": {d}, \"round_launches\": {{", .{rank});
    var first = true;
    for (m.rd.launches, 0..) |n, i| {
        if (n == 0 or i >= names.len) continue;
        try w.print("{s}\"{s}\": {d:.1}", .{ if (first) "" else ", ", names[i], @as(f64, @floatFromInt(n)) / r });
        first = false;
    }
    try w.print("}}}}\n", .{});
    try w.flush();
}
