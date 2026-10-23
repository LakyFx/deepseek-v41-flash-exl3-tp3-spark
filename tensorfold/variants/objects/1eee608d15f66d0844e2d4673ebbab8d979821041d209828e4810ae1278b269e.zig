//! One stream (Python LaneStream) and the round loop's state for it (Python's dicts keyed by stream id).
const std = @import("std");
const Novelty = @import("novelty.zig").Novelty;
const shape = @import("shape.zig");
const plan_lanes = @import("plan_lanes.zig");
const Allocator = std.mem.Allocator;
const Proposer = @import("proposer.zig").Proposer;
const Sampling = @import("sampling.zig").Sampling;
const State = @import("depth.zig").State;

pub const Reason = enum {
    none,
    stop,
    length,
    @"error",
    cancelled,

    pub fn name(r: Reason) []const u8 {
        return if (r == .none) "" else @tagName(r);
    }
};

pub const Mode = enum { pipe, drain, verify, exit };

/// Verified head chains only. Reached includes the first rejected draft, not untested descendants.
pub const Depth = struct { drafted: u32 = 0, reached: u32 = 0, kept: u32 = 0 };

pub const loop_max_period: usize = 8;
pub const loop_min_run: usize = 256;
pub const loop_warm_in: usize = 64;

/// Return the shortest exact cycle at the end of a combined token prefix.
fn cycleAt(head: []const u32, tail: []const u32, n: usize) ?u32 {
    if (n < loop_min_run + 1) return null;
    for (1..loop_max_period + 1) |period| {
        if (n < loop_min_run + period) break;
        if (n - loop_min_run - period < loop_warm_in) break;
        var matches = true;
        for (1..loop_min_run + 1) |k| {
            const right = n - k;
            const left = right - period;
            const right_token = if (right < head.len) head[right] else tail[right - head.len];
            const left_token = if (left < head.len) head[left] else tail[left - head.len];
            if (right_token != left_token) {
                matches = false;
                break;
            }
        }
        if (matches) return @intCast(period);
    }
    return null;
}

/// Return the first token index in a candidate batch that would fire the guard.
pub fn loopCut(s: *const Stream, tokens: []const u32) ?usize {
    if (!s.guarding()) return null;
    const dry = if (s.novelty) |*n| n.firesAt(s.emitted(), tokens) else null;
    for (tokens, 0..) |token, i| {
        if (@as(i64, token) == s.think_end) return null;
        if (cycleAt(s.emitted(), tokens, s.emitted().len + i + 1) != null) return i;
        if (dry) |at| if (at == i) return i;
    }
    return null;
}

/// Return the detected period at the end of committed tokens, or null.
pub fn detectLoop(emitted: []const u32) ?u32 {
    return cycleAt(emitted, &.{}, emitted.len);
}

/// A stop-string check over the emitted tokens (the server's StopPolicy decodes their tail).
pub const StopCheck = struct {
    ptr: *anyopaque,
    check: *const fn (ptr: *anyopaque, emitted: []const u32) bool,
};

/// The host's answer to whether the stream's request was cancelled, asked between prompt chunks.
pub const CancelCheck = struct {
    ptr: *anyopaque,
    check: *const fn (ptr: *anyopaque) bool,
};

/// A kept prompt state the backend restores before its prompt pass (core/prompt_cache.zig), and where it keeps new ones.
pub const Reuse = struct {
    saved: ?*anyopaque = null, // the backend's copy of the state after `at` prompt tokens (null: start at 0)
    at: u32 = 0,
    marks: []const u32 = &.{}, // ascending: the pass cuts a chunk at each and calls `hook` there
    hook: ?MarkHook = null,
};

/// Called between prompt chunks, the pass standing at `at` (the GPU done with the chunk that ends there).
pub const MarkHook = struct {
    ptr: *anyopaque,
    at: *const fn (ptr: *anyopaque, s: *Stream, at: u32) void,
};

/// Drafts the backend holds for the stream's next round; a tree keeps its tokens and parents on the host.
pub const Held = struct {
    count: u32,
    tokens: ?[]u32 = null,
    parents: ?[]i32 = null,
};

/// An image in the prompt (a vision family's): the span of `tokens` positions from `at` it fills (each position carries
/// the image token id), and the payload its family's Vision.prepare made, or with `rows` its span's embedding rows
/// ready (a gate's input).
pub const Image = struct {
    at: u32,
    tokens: u32,
    bytes: []const u8,
    rows: bool = false,
};

/// A vision family's image input as a server renders it (engine_api.Info.vision): each image part becomes
/// `placeholder` (one token: `token`) in the template's text, and `prepare` turns the image's bytes into its span's
/// length and the payload the backend reads (Image.bytes), on the request's thread; on BadImage `problem` says why.
pub const Vision = struct {
    ctx: *anyopaque,
    placeholder: []const u8,
    token: u32,
    prepare: *const fn (ctx: *anyopaque, a: Allocator, bytes: []const u8, problem: *[]const u8) PrepareError!Prepared,

    pub const Prepared = struct { tokens: u32, payload: []const u8 };
    pub const PrepareError = error{ BadImage, OutOfMemory };
};

pub const Spec = struct {
    id: []const u8,
    prompt: []const u32,
    max_new: u32,
    eos: []const u32 = &.{},
    sampling: ?Sampling = null,
    drafts: bool = true,
    proposer: ?Proposer = null,
    stop_check: ?StopCheck = null,
    cancel_check: ?CancelCheck = null,
    think_budget: u32 = 0,
    think_close: []const u32 = &.{},
    think_end: i64 = -1,
    think_open: ?bool = null, // null: open when a budget is set (the server's rule)
    loop_guard: bool = false,
    loop_dry_windows: u32 = 10, // patient token-novelty threshold; zero disables novelty, retaining exact cycles
    chunks: []const u32 = &.{}, // where prefill chunks start after 0 (Python's PrefillPlan); empty: the backend's step
    reuse: Reuse = .{},
    prefill_progress: ?MarkHook = null, // confirmed GPU chunk boundary, observational only
    images: []const Image = &.{}, // in prompt order; the caller's until the stream is released
};

pub const Stream = struct {
    id: []const u8,
    prompt_len: usize,
    max_new: u32,
    eos: []const u32,
    sampling: ?Sampling,
    drafts: bool,
    proposer: ?Proposer,
    stop_check: ?StopCheck,
    cancel_check: ?CancelCheck,
    think_budget: u32,
    think_close: []const u32,
    think_end: i64,
    think_open: bool,
    loop_guard: bool,
    loop_period: ?u32 = null,
    loop_novelty: ?f64 = null,
    novelty: ?Novelty = null,
    chunks: []const u32,
    reuse: Reuse = .{},
    prefill_progress: ?MarkHook = null,
    images: []const Image = &.{},
    cached: u32 = 0, // prompt tokens the backend restored from `reuse` (its prompt pass started there)
    reuse_failed: bool = false, // the backend's restore of `reuse` failed: it prefilled from 0
    context: std.ArrayList(u32) = .empty,
    pending: ?u32 = null,
    force: std.ArrayList(u32) = .empty,
    cache_len: u64 = 0,
    finished: bool = false,
    reason: Reason = .none,
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    depths: [shape.max_depth]Depth = @splat(.{}),
    branch_rows: u64 = 0, // suffix-match branch rows verified (lanes/fill.zig)
    branch_accepted: u64 = 0, // of those, rows on the accepted path
    graft_hit: f64 = 1, // recent share of grafted rows kept; a read of the chain that grafted none counts as a miss
    odds: ?shape.Odds = null, // how often the target took the head's rank-r lane at each depth (head trees)
    lanes: ?shape.Shape = null, // the shape of the held tree drafts
    landed: [shape.max_depth][shape.ranks][2]u32 = @splat(@splat(.{ 0, 0 })), // tree lanes verified and taken
    picks: [plan_lanes.sizes.len + 1]u32 = @splat(0), // the planner's picks: each tree size, then chain + grafts
    graft_room: ?u32 = null, // grafted rows the planner left this round's window (null: no planner)
    held_levels: u32 = 0, // head levels drafted for the held window
    round_over: ?f64 = null, // the stream's round time beyond its window and head levels (ms, recent): chains
    tree_over: ?f64 = null, // and trees
    measured: [plan_lanes.sizes.len + 1]plan_lanes.Measured = @splat(.{}), // each pick's round times past its table price
    priced: ?struct { choice: usize, ms: f64 } = null, // the pick that shaped the held window, at its table price
    min_rows: u32 = 0,
    paused: bool = false,
    depth: ?State = null,
    mode: ?Mode = null,
    next: ?Held = null,
    inflight: ?u64 = null,
    served: ?u64 = null,
    granted: ?u32 = null,
    copy_width: ?u32 = null,

    pub fn init(gpa: Allocator, spec: Spec) !Stream {
        var s: Stream = .{
            .id = spec.id,
            .prompt_len = spec.prompt.len,
            .max_new = spec.max_new,
            .eos = spec.eos,
            .sampling = spec.sampling,
            .drafts = spec.drafts,
            .proposer = if (spec.drafts) spec.proposer else null,
            .stop_check = spec.stop_check,
            .cancel_check = spec.cancel_check,
            .think_budget = spec.think_budget,
            .think_close = spec.think_close,
            .think_end = spec.think_end,
            .think_open = spec.think_open orelse (spec.think_budget > 0 or (spec.loop_guard and spec.think_end >= 0)),
            .loop_guard = spec.loop_guard,
            .novelty = if (spec.loop_guard and spec.loop_dry_windows > 0) .{ .windows = spec.loop_dry_windows } else null,
            .chunks = spec.chunks,
            .reuse = spec.reuse,
            .prefill_progress = spec.prefill_progress,
            .images = spec.images,
        };
        try s.context.appendSlice(gpa, spec.prompt);
        return s;
    }

    pub fn deinit(s: *Stream, gpa: Allocator) void {
        s.context.deinit(gpa);
        s.force.deinit(gpa);
        if (s.novelty) |*n| n.deinit(gpa);
        s.dropRoundState(gpa);
    }

    /// Forget the round loop's state (Python `_release_stream_state`).
    pub fn dropRoundState(s: *Stream, gpa: Allocator) void {
        if (s.depth) |*d| d.deinit(gpa);
        s.dropHeld(gpa);
        s.depth = null;
        s.mode = null;
        s.inflight = null;
        s.served = null;
        s.granted = null;
        s.copy_width = null;
    }

    pub fn dropHeld(s: *Stream, gpa: Allocator) void {
        if (s.next) |h| {
            if (h.tokens) |t| gpa.free(t);
            if (h.parents) |p| gpa.free(p);
        }
        s.next = null;
        if (s.lanes) |l| l.deinit(gpa);
        s.lanes = null;
    }

    pub fn prompt(s: *const Stream) []const u32 {
        return s.context.items[0..s.prompt_len];
    }

    pub fn emitted(s: *const Stream) []const u32 {
        return s.context.items[s.prompt_len..];
    }

    pub fn budgetLeft(s: *const Stream) i64 {
        return @as(i64, s.max_new) - @as(i64, @intCast(s.emitted().len));
    }

    /// Whether the request was cancelled (false with no hook); a prompt pass asks before each chunk.
    pub fn isCancelled(s: *const Stream) bool {
        const c = s.cancel_check orelse return false;
        return c.check(c.ptr);
    }

    fn budgetActive(s: *const Stream) bool {
        return s.think_open and s.think_budget > 0 and s.think_close.len > 0;
    }

    /// The index the thinking budget replaces with `think_close[0]`, or null if the block closed first.
    pub fn thinkCut(s: *const Stream, tokens: anytype) ?usize {
        if (!s.budgetActive()) return null;
        const done = s.emitted().len;
        for (tokens, 0..) |t, i| {
            if (done + i + 1 >= s.think_budget) return i;
            if (@as(i64, t) == s.think_end) return null;
        }
        return null;
    }

    /// Python `think_cut([-1]) == 0`: the next position is the budget's cut.
    pub fn cutsNext(s: *const Stream) bool {
        return s.budgetActive() and s.emitted().len + 1 >= s.think_budget;
    }

    /// Begin the thinking budget's close: its first token now, the rest forced after it.
    pub fn startClose(s: *Stream, gpa: Allocator) !u32 {
        s.think_open = false;
        s.force.clearRetainingCapacity();
        try s.force.appendSlice(gpa, s.think_close[1..]);
        return s.think_close[0];
    }

    /// Tokens a round may commit before the length limit or the thinking budget's cut.
    pub fn draftRoom(s: *const Stream) i64 {
        var room = s.budgetLeft();
        if (s.budgetActive()) room = @min(room, @as(i64, s.think_budget) - @as(i64, @intCast(s.emitted().len)));
        return room;
    }

    pub fn isEos(s: *const Stream, token: u32) bool {
        return std.mem.indexOfScalar(u32, s.eos, token) != null;
    }

    /// Exact counts for the first sixteen head depths; deeper positions remain in overall draft totals only.
    pub fn noteDepths(s: *Stream, drafted: usize, kept: usize) void {
        std.debug.assert(kept <= drafted);
        for (s.depths[0..@min(drafted, s.depths.len)], 0..) |*d, j| {
            d.drafted += 1;
            d.reached += @intFromBool(j <= kept);
            d.kept += @intFromBool(j < kept);
        }
    }

    fn guarding(s: *const Stream) bool {
        return s.loop_guard and s.think_open and s.think_end >= 0 and s.loop_period == null and s.loop_novelty == null;
    }

    /// Append tokens until the stream finishes; how many landed (a prefix of `tokens`).
    pub fn commit(s: *Stream, gpa: Allocator, tokens: []const u32) !usize {
        var landed: usize = 0;
        for (tokens) |t| {
            if (s.finished) break;
            try s.context.append(gpa, t);
            landed += 1;
            if (@as(i64, t) == s.think_end) s.think_open = false;
            const watch = s.guarding();
            const period = if (watch) detectLoop(s.emitted()) else null;
            const dry: ?f64 = if (watch and period == null and s.novelty != null) blk: {
                const n = &s.novelty.?;
                const share = try n.observe(gpa, s.emitted()) orelse break :blk null;
                break :blk if (n.fires()) share else null;
            } else null;
            if (s.isEos(t) or s.stopped()) {
                s.finished = true;
                s.reason = .stop;
            } else if (period != null or dry != null) {
                s.loop_period = period;
                s.loop_novelty = dry;
                if (s.emitted().len >= s.max_new) {
                    s.finished = true;
                    s.reason = .length;
                } else if (s.think_close.len == 0) {
                    s.finished = true;
                    s.reason = .stop;
                } else {
                    s.think_open = false;
                    s.force.clearRetainingCapacity();
                    try s.force.appendSlice(gpa, s.think_close);
                }
                break;
            } else if (s.emitted().len >= s.max_new) {
                s.finished = true;
                s.reason = .length;
            }
        }
        return landed;
    }

    fn stopped(s: *const Stream) bool {
        const c = s.stop_check orelse return false;
        return c.check(c.ptr, s.emitted());
    }

    pub fn popForce(s: *Stream) ?u32 {
        if (s.force.items.len == 0) return null;
        return s.force.orderedRemove(0);
    }
};

test "commit stops at eos and length" {
    const gpa = std.testing.allocator;
    var s = try Stream.init(gpa, .{ .id = "a", .prompt = &.{ 1, 2 }, .max_new = 3, .eos = &.{9} });
    defer s.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 2), try s.commit(gpa, &.{ 5, 9, 7 }));
    try std.testing.expectEqualSlices(u32, &.{ 5, 9 }, s.emitted());
    try std.testing.expect(s.finished and s.reason == .stop);
}

test "loop guard stops at the cap even when no cycle fires" {
    const gpa = std.testing.allocator;
    var tokens: [20]u32 = undefined;
    for (&tokens, 0..) |*token, i| token.* = @intCast(100 + i);
    var guarded = try Stream.init(gpa, .{ .id = "guarded", .prompt = &.{1}, .max_new = 10, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_guard = true });
    defer guarded.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 10), try guarded.commit(gpa, &tokens));
    try std.testing.expect(guarded.finished and guarded.reason == .length);
}

test "loop guard fires, closes thinking, and latches once" {
    const gpa = std.testing.allocator;
    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(gpa);
    for (0..loop_warm_in) |i| try tokens.append(gpa, @intCast(1000 + i));
    for (0..257) |_| try tokens.append(gpa, 7);
    var s = try Stream.init(gpa, .{ .id = "loop", .prompt = &.{1}, .max_new = 400, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_guard = true });
    defer s.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 321), try s.commit(gpa, tokens.items));
    try std.testing.expectEqual(@as(u32, 1), s.loop_period.?);
    try std.testing.expect(!s.finished and !s.think_open and s.force.items.len == 3);
    while (s.popForce()) |token| _ = try s.commit(gpa, &.{token});
    try std.testing.expect(!s.finished and s.force.items.len == 0);
    _ = try s.commit(gpa, &.{8});
    try std.testing.expect(!s.finished and s.loop_period.? == 1);
}

const reference = @import("novelty.zig").reference;

/// Commit `tokens` `batch` at a time as the lane core's rounds do, cut where loopCut cuts, until the guard fires or the
/// stream finishes; how many landed. With rounds of 7 or 50 a window ends inside a round: only loopCut ends it there.
fn rounds(gpa: Allocator, s: *Stream, tokens: []const u32, batch: usize) !usize {
    var at: usize = 0;
    while (at < tokens.len and !s.finished and s.force.items.len == 0) {
        var next = tokens[at..@min(tokens.len, at + batch)];
        if (loopCut(s, next)) |cut| next = next[0 .. cut + 1];
        const landed = try s.commit(gpa, next);
        at += landed;
        if (landed < next.len) {
            try std.testing.expect(s.finished); // a guard that fires mid-batch fires where loopCut cut
            break;
        }
    }
    return at;
}

const Fire = struct { at: usize, share: f64 };

/// Where the novelty signal fires on `tokens` (the reply index of its last token) and the window's share of new grams.
fn novelFire(gpa: Allocator, tokens: []const u32, windows: u32, batch: usize) !?Fire {
    var s = try Stream.init(gpa, .{ .id = "novel", .prompt = &.{1}, .max_new = 1 << 20, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_guard = true, .loop_dry_windows = windows });
    defer s.deinit(gpa);
    const landed = try rounds(gpa, &s, tokens, batch);
    try std.testing.expect(s.loop_period == null and !s.finished);
    const share = s.loop_novelty orelse {
        try std.testing.expectEqual(tokens.len, landed);
        return null;
    };
    try std.testing.expect(!s.think_open and std.mem.eql(u32, &.{ 90, 91, 92 }, s.force.items));
    return .{ .at = landed - 1, .share = share };
}

test "the novelty signal fires where the Python guard does, a token or a round at a time" {
    const gpa = std.testing.allocator;
    var a: std.ArrayList(u32) = .empty;
    defer a.deinit(gpa);
    try reference.fresh(gpa, &a, 0, 2500);
    try reference.loop13(gpa, &a, 20000);
    var b: std.ArrayList(u32) = .empty;
    defer b.deinit(gpa);
    try reference.fresh(gpa, &b, 0, 2048);
    try reference.loop13(gpa, &b, 5 * 1024);
    try reference.fresh(gpa, &b, 10000, 20 * 1024);
    var c: std.ArrayList(u32) = .empty; // the think block closes before the loop's fifth window ends
    defer c.deinit(gpa);
    try reference.fresh(gpa, &c, 0, 2500);
    try reference.loop13(gpa, &c, 3000);
    try c.append(gpa, 91);
    try reference.loop13(gpa, &c, 20000);
    var d: std.ArrayList(u32) = .empty;
    defer d.deinit(gpa);
    try reference.lcg(gpa, &d, 60 * 1024);
    const cases = [_]struct { tokens: []const u32, windows: u32, want: ?Fire }{
        .{ .tokens = a.items, .windows = 3, .want = .{ .at = 6143, .share = 0 } },
        .{ .tokens = a.items, .windows = 10, .want = .{ .at = 13311, .share = 0 } },
        .{ .tokens = b.items, .windows = 3, .want = .{ .at = 6143, .share = 0 } },
        .{ .tokens = b.items, .windows = 10, .want = null },
        .{ .tokens = c.items, .windows = 3, .want = null },
        .{ .tokens = c.items, .windows = 10, .want = null },
        .{ .tokens = d.items, .windows = 3, .want = .{ .at = 30719, .share = 5.0 / 1024.0 } },
        .{ .tokens = d.items, .windows = 10, .want = .{ .at = 37887, .share = 2.0 / 1024.0 } },
        .{ .tokens = a.items, .windows = 0, .want = null }, // the exact-cycle check alone: a loop of 13 is no cycle
    };
    for (cases) |case| for ([_]usize{ 1, 7, 50 }) |batch| {
        try std.testing.expectEqual(case.want, try novelFire(gpa, case.tokens, case.windows, batch));
    };
}

test "the novelty signal closes thinking, reports the window's share, and latches once" {
    const gpa = std.testing.allocator;
    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(gpa);
    try reference.fresh(gpa, &tokens, 0, 1024);
    try reference.loop13(gpa, &tokens, 4 * 1024);
    const spec: Spec = .{ .id = "templated", .prompt = &.{1}, .max_new = 8000, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_guard = true, .loop_dry_windows = 1 };
    var s = try Stream.init(gpa, spec);
    defer s.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 3 * 1024), try rounds(gpa, &s, tokens.items, 50));
    try std.testing.expect(s.loop_period == null and s.loop_novelty.? == 0);
    try std.testing.expect(!s.finished and !s.think_open and s.force.items.len == 3);
    while (s.popForce()) |token| _ = try s.commit(gpa, &.{token});
    try std.testing.expectEqual(tokens.items.len - 3 * 1024, try rounds(gpa, &s, tokens.items[3 * 1024 ..], 50));
    try std.testing.expect(!s.finished and s.loop_novelty.? == 0 and s.force.items.len == 0);

    // A driver handing in more than one window also cuts at the same token.
    var whole = try Stream.init(gpa, spec);
    defer whole.deinit(gpa);
    try std.testing.expectEqual(@as(?usize, 3 * 1024 - 1), loopCut(&whole, tokens.items));
    try std.testing.expectEqual(@as(usize, 3 * 1024), try whole.commit(gpa, tokens.items));
    try std.testing.expect(whole.loop_novelty.? == 0 and !whole.think_open);
}
