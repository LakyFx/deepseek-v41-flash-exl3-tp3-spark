//! DeepSeek-V4.1-Flash behind the native CUDA family interface, tensor parallel over two GPUs: the server's process is
//! rank 0 (the lane core over lanes.zig's Backend on model.zig), rank 1 is `tf-dsv41-lanes` following it on the other
//! node (the same model directory, rank cache, kit, pool and port). The ranks link over TCP and NCCL; the inputs
//! beyond the checkpoint come from the environment:
//!   TF_TP_WORLD (2), TF_TP_MASTER, TF_TP_PORT   the ranks: rank 0's fabric address and port (rank 1 connects)
//!   TF_DS_RANK_CACHE                            the lane's per-rank weight files (the Python lane's TF_DS_RANK_CACHE)
//!   TF_DS_KIT                                   cubins/ (the served extension cubins), the served RoPE tables
//!                                               (rope-*.f32, as many rows as the context) and engram.json
//!   TF_DS_ENGRAM, TF_DS_TOKEN_MAP               the Engram tables and the compressed token map
//!   TF_DS_ARENA_GIB                             a first block of device memory for the buffers (unset: they take what
//!                                               they need, the pool's caches by the context)
//!   TF_DS_POOL_TOKENS                          optional aggregate pool, independent of --context per request;
//!                                               followers must use the same value via --pool
//!   TF_DS_ROUND_MS                              the lane core's round costs by rows (ms, comma separated; the
//!                                               served ROUND_MS unless set: a four-node lane's rounds cost less)
//!   TF_DS_ROUND_ROWS, TF_DS_STREAMS             a round's rows at most (16; the four-node lane's 48) and the pool's
//!                                               streams (4; its 16): --parallel at most the streams
//!   TF_RDMA_DEVICES                             the decode gathers over the RDMA ring on these devices (else NCCL)
//!   TF_DS_GRAPHS=1                              the rounds' stretches as CUDA graphs
//!   TF_DS_HC_SIDE=1                             the mixes' side work on a stream of its own
//!   TF_DS_L2_PREFETCH=1                         the paced L2 prefetch of the next kernels' weights
//!   TF_DS_ENGRAM_AIO=1                          a round's Engram reads by Linux AIO on O_DIRECT
//!   TF_DS_FIX_K=K                               a measurement: every drafted stream verifies K drafts a round
//! The kernel set (TENSORFOLD_CUDA_KERNELS) is the recorded Triton set (aot_pack.py). The pool holds TF_DS_POOL_TOKENS
//! (default --context)
//! positions for every stream together (each takes an extent: its prompt, its max_tokens and a round's rows).
//! TF_TP_WORLD=4: the four nodes of the exact 2D split (prompt2d.zig), ranks 1-3 `tf-dsv41-lanes` following on the
//! other nodes; TF_RDMA_DEVICES (and TF_DS_RDMA_KERNELS: rdma_gather.cu's fatbin, unset: the embedded one) put its
//! decode-size exchanges on the RDMA rings, as the four-node Python lane's TF_RDMA_DEVICES does.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const model = @import("model.zig");
const lanes_mod = @import("lanes.zig");
const round = @import("round.zig");
const link = @import("link.zig");
const rank_cache = @import("rank_cache.zig");
const vision_mod = @import("vision.zig");

pub const model_type = "deepseek_v41";
pub const formats: []const []const u8 = &.{"exl3-mul1"};
/// The Python lane's default window; the server's --context moves it up to the model's 1,048,576.
pub const default_context: i64 = 262144;
pub const max_segments: u32 = 1;
/// The served build's prompt chunk (TF_DS_PREFILL_CHUNK).
pub const prompt_rows: u32 = 2048;
/// The served lane's reasoning effort numbers (text.zig): the server passes each name and the template hears its number.
pub const efforts = @import("text.zig").efforts;

pub const Options = struct { context: usize, drafts: bool, segments: usize = 1 };

/// Separate the HTTP request window from the aggregate placement plane. The
/// legacy default is unchanged; an explicit pool leaves room for the aligned
/// placement's verification tail even for a request filling its whole window.
/// RoPE uses FP32 positions, exact only below 2^24; this is an arithmetic bound,
/// not a memory-fit promise. Every rank still needs matching pool and RoPE files.
pub fn poolTokens(context: usize, value: ?[]const u8) !usize {
    const text = value orelse return context;
    if (text.len == 0) return error.BadPool;
    for (text) |c| if (c < '0' or c > '9') return error.BadPool;
    const count = std.fmt.parseInt(usize, text, 10) catch return error.BadPool;
    const limit = (1 << 24) - round.max_rows - 1;
    if (context > limit or count > limit) return error.BadPool;
    const need = std.mem.alignForward(usize, context + round.max_rows + 2, model.chunk_rows);
    if (count < need) return error.BadPool;
    return count;
}

test "aggregate pool is independent of request context and keeps legacy default" {
    try std.testing.expectEqual(@as(usize, 401408), try poolTokens(401408, null));
    try std.testing.expectEqual(@as(usize, 1048576), try poolTokens(401408, "1048576"));
    try std.testing.expectEqual(@as(usize, 3300000), try poolTokens(401408, "3300000"));
    try std.testing.expectEqual(@as(usize, 403456), try poolTokens(401408, "403456"));
    for ([_][]const u8{ "", "0", "401408", "403455", "-1", "+1048576", "1_048_576", " 1048576", "1048576 ", "16777216", "999999999999999999999999" }) |value|
        try std.testing.expectError(error.BadPool, poolTokens(401408, value));
}

/// A lone drafted stream's own driver: decodes it until it finishes (false) or `yield` hands it over (true).
pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool) anyerror!bool;

/// What the native server drives: the lane backend, the facts its round loop reads, and how to free it.
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    /// Device bytes each admitted stream allocates for its own sequence.
    stream_bytes: usize,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
    /// Kept prompt states for the host's prompt cache (core/prompt_cache Snapshots: `ptr` and its functions).
    snaps: ?Snaps = null,
    /// Images in requests (the server renders their placeholders; the tower runs on this rank).
    vision: ?lanes.stream.Vision = null,
};

pub const Snaps = struct {
    ptr: *anyopaque,
    bytes: *const fn (*anyopaque, u32) u64,
    save: *const fn (*anyopaque, ?*anyopaque, u32) anyerror!*anyopaque,
    restore: *const fn (*anyopaque, ?*anyopaque, *anyopaque) anyerror!void,
    drop: *const fn (*anyopaque, *anyopaque) void,
    live: *const fn (*anyopaque, *anyopaque) bool,
    dead_reason: *const fn (*anyopaque, *anyopaque) u8,
    compatible: *const fn (*anyopaque, *anyopaque, usize) bool,
    exact_end: bool = true,
    map_mark: *const fn (*anyopaque, ?*anyopaque, usize, u32) u32,
    automatic_marks: *const fn (*anyopaque, ?*anyopaque, usize, *[2]u32) usize,
    tail_window: u32,
    continuation_hint: *const fn (*anyopaque, *anyopaque, u64) void,
    adaptive_hint: *const fn (*anyopaque, *anyopaque, u32) void,
    ghost_range: *const fn (*anyopaque, *anyopaque) [2]u32,
};

/// The rank's model and the backend over it (what `open` hands the server, freed by `deinit`).
const State = struct {
    gpa: std.mem.Allocator,
    m: *model.Model,
    lanes: lanes_mod.Lanes,
    vision: ?*vision_mod.Vision = null,
};

fn getenv(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// An empty variable as none (a container passes TF_RDMA_DEVICES empty when it is unset).
fn nonEmpty(v: ?[]const u8) ?[]const u8 {
    const s = v orelse return null;
    return if (s.len == 0) null else s;
}

/// The device bytes this rank's weights take, for the host's memory check: its file in TF_DS_RANK_CACHE (the
/// checkpoint holds both ranks'); 0 when there is none to read (open then says what to set).
pub fn weightBytes(io: std.Io, dir: []const u8) u64 {
    _ = dir;
    var d = std.Io.Dir.cwd().openDir(io, getenv("TF_DS_RANK_CACHE") orelse return 0, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    var buf: [512]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const world = std.fmt.parseInt(u32, getenv("TF_TP_WORLD") orelse "2", 10) catch return 0;
    const name = rank_cache.find(fba.allocator(), io, d, 0, world) catch return 0;
    const st = d.statFile(io, name, .{}) catch return 0;
    return st.size;
}

/// Loads rank 0 and links followers against TF_TP_MASTER:TF_TP_PORT; the pool holds TF_DS_POOL_TOKENS or `o.context`
/// positions. Needs `ctx` current.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options) !Loaded {
    const aggregate_pool = try poolTokens(o.context, getenv("TF_DS_POOL_TOKENS"));
    const world = std.fmt.parseInt(u32, getenv("TF_TP_WORLD") orelse "2", 10) catch return error.BadWorld;
    try model.checkWorldMode(world, getenv("TF_DS_EXPERIMENTAL_TP3"));
    const master = link.parseIp(getenv("TF_TP_MASTER") orelse return error.NoMaster) catch return error.BadMaster;
    const port = std.fmt.parseInt(u16, getenv("TF_TP_PORT") orelse "29620", 10) catch return error.BadPort;
    const arena_gib = if (getenv("TF_DS_ARENA_GIB")) |t| std.fmt.parseInt(usize, t, 10) catch return error.BadArena else 0;
    const st = try gpa.create(State);
    errdefer gpa.destroy(st);
    st.gpa = gpa;
    st.m = try model.Model.open(gpa, io, ctx, .{
        .model_dir = dir,
        .cache_dir = getenv("TF_DS_RANK_CACHE") orelse return error.NoRankCache,
        .kit_dir = getenv("TF_DS_KIT") orelse return error.NoKit,
        .aot_dir = kernels,
        .rank = 0,
        .world = world,
        .master = master,
        .port = port,
        .engram_dir = getenv("TF_DS_ENGRAM"),
        .token_map = getenv("TF_DS_TOKEN_MAP"),
        .rdma_devices = nonEmpty(getenv("TF_RDMA_DEVICES")),
        .rdma_kernels = getenv("TF_DS_RDMA_KERNELS"),
        .graphs = if (getenv("TF_DS_GRAPHS")) |v| !std.mem.eql(u8, v, "0") else false,
        .side = if (getenv("TF_DS_HC_SIDE")) |v| !std.mem.eql(u8, v, "0") else false,
        .prefetch = if (getenv("TF_DS_L2_PREFETCH")) |v| !std.mem.eql(u8, v, "0") else false,
        .engram_aio = if (getenv("TF_DS_ENGRAM_AIO")) |v| !std.mem.eql(u8, v, "0") else false,
        .pool = aggregate_pool,
        .drafts = o.drafts,
        .arena_bytes = arena_gib << 30,
        .round_rows = if (getenv("TF_DS_ROUND_ROWS")) |v| std.fmt.parseInt(usize, v, 10) catch return error.BadRoundRows else round.default_rows,
        .round_ms = if (getenv("TF_DS_ROUND_MS")) |v| lanes_mod.parseMs(std.heap.page_allocator, v) catch return error.BadRoundMs else null,
        .streams = if (getenv("TF_DS_STREAMS")) |v| std.fmt.parseInt(usize, v, 10) catch return error.BadStreams else model.max_streams_default,
    });
    errdefer st.m.close();
    st.lanes = lanes_mod.Lanes.init(gpa, st.m);
    if (getenv("TF_DS_ADAPTIVE_LEARNING")) |value| {
        if (!std.mem.eql(u8, value, "0") and !std.mem.eql(u8, value, "1")) return error.BadAdaptiveLearning;
        st.lanes.adaptive_learning = std.mem.eql(u8, value, "1");
    }
    st.lanes.logit_dump_dir = getenv("TF_DS_CACHE_LOGITS_DIR");
    st.lanes.short_continuation = if (getenv("TF_DS_CACHE_SHORT_CONTINUATION")) |v| std.mem.eql(u8, v, "1") else false;
    if (getenv("TF_DS_FIX_K")) |t| st.lanes.fix_k = std.fmt.parseInt(usize, t, 10) catch return error.BadFixK;
    // a vision checkpoint's tower on this rank, from the kit's vision/ (torch's attention cubin, the routing bias)
    st.vision = null;
    const diagnostic_no_vision = if (getenv("TF_DS_DIAGNOSTIC_NO_VISION")) |v| std.mem.eql(u8, v, "1") else false;
    if (diagnostic_no_vision) std.log.warn("DIAGNOSTIC ONLY: vision initialization omitted, model not qualified", .{});
    if (st.m.cfg.vision and !diagnostic_no_vision) st.vision = try vision_mod.Vision.open(gpa, io, st.m, dir, getenv("TF_DS_KIT").?);
    if (st.m.ring) |r| r.dumpRegistrations("after HTTP family init");
    errdefer if (st.vision) |v| v.close();
    st.lanes.vision = st.vision;
    // the pool is the model's (preallocated): a stream takes no device memory of its own
    const L = lanes_mod.Lanes;
    const snaps: Snaps = .{ .ptr = &st.lanes, .bytes = L.snapBytesFn, .save = L.snapSaveFn, .restore = L.snapRestoreFn, .drop = L.snapDropFn, .live = L.snapLiveFn, .dead_reason = L.snapDeadReasonFn, .compatible = L.snapCompatibleFn, .map_mark = L.snapMapMarkFn, .automatic_marks = L.snapAutomaticMarksFn, .tail_window = @intCast(st.m.cfg.window), .continuation_hint = L.snapContinuationHintFn, .adaptive_hint = L.snapAdaptiveHintFn, .ghost_range = L.snapGhostRangeFn };
    return .{ .backend = st.lanes.backend(), .facts = st.lanes.facts(), .rows = @intCast(st.m.eng.round_rows), .stream_bytes = 0, .ctx = st, .deinit = deinitFn, .snaps = snaps, .vision = if (st.vision) |v| v.hook() else null };
}

fn deinitFn(ptr: *anyopaque) void {
    const st: *State = @ptrCast(@alignCast(ptr));
    st.lanes.deinit(); // rank 1 hears done
    if (st.vision) |v| v.close();
    st.m.close();
    st.gpa.destroy(st);
}

/// A request or a start this engine refuses, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.NotPortedYet => "the native DeepSeek-V4.1 engine supports world2/4 and this fork's experimental world3",
        error.ExperimentalTP3Required => "TP3 requires TF_DS_EXPERIMENTAL_TP3=1 on every rank and the padded TP3 checkpoint; GPU/numerical qualification is still required",
        error.NoMaster, error.BadMaster => "set TF_TP_MASTER to rank 0's fabric address (rank 1: tf-dsv41-lanes MODEL CACHE 1 2 MASTER PORT KIT; ranks 1-3 of four: ... G 4 ...)",
        error.NoRdmaKernels => "TF_RDMA_DEVICES at world 4 needs TF_DS_RDMA_KERNELS (rdma_gather.cu's fatbin) in a build without the kernel images",
        error.NoRankCache => "set TF_DS_RANK_CACHE to the lane's per-rank weight files",
        error.NoKit => "set TF_DS_KIT to the kernel kit (cubins/, the RoPE tables, engram.json)",
        error.NoFreeSlot => "this engine serves TF_DS_STREAMS streams at once (default 4): --parallel at most that",
        error.BadRoundRows => "TF_DS_ROUND_ROWS: a round's rows at most, 16 .. 48",
        error.BadStreams => "TF_DS_STREAMS: the pool's streams, 1 .. 16",
        error.BadPool => "TF_DS_POOL_TOKENS: aggregate positions, at least alignUp(--context + 50, 2048), below 2^24 - 48; use the same --pool on followers and sufficient RoPE tables",
        error.ContextFull => "the streams' prompts and budgets fill the context's pool: lower max_tokens, or retry when a stream ends",
        error.PromptTooLong => "the prompt and its max_tokens do not fit the context window",
        error.NoVisionKit => "this checkpoint reads images: its kit (TF_DS_KIT) needs vision/torch_fmha_sm120.cubin and vision/gate_bias_vl.safetensors",
        else => null,
    };
}

test "refusals name what to set" {
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.NoMaster).?, "TF_TP_MASTER") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.NoFreeSlot).?, "TF_DS_STREAMS") != null);
    try std.testing.expectEqual(@as(?[]const u8, null), explain(null, error.OutOfMemory));
}
