//! Pressure-only host planner. No CUDA, tokens, allocation mutations or wall-clock TTL.
const std = @import("std");
pub const quantum = 2048;
pub const hard_copy_bytes: usize = 320 * 1024 * 1024;
pub const max_states = 512;
pub const State = struct {
    key: u64, base: usize, len: usize,
    restores: u64 = 0, used: u64 = 0, protected: bool = false,
};
pub const Move = struct { from: usize, to: usize, len: usize };
pub const Plan = struct {
    base: usize,
    lost: usize,
    reuse_loss: u64 = 0,
    recency_loss: u64 = 0,
    victims: usize = 0,
    move: ?Move = null,
    evaluations: usize = 0,
    truncated: bool = false,
};
pub const Options = struct {
    max_move_rows: usize = 0,
    min_saved_rows: usize = 16384,
    // A bounded CPU search; callers may additionally give an absolute monotonic deadline.
    deadline_ns: u64 = std.math.maxInt(u64),
    max_evaluations: usize = 2048,
};
/// The only commit order used by the real native mover and its CPU fault-injection tests.
pub fn transferAndCommit(runtime: anytype) !usize {
    try runtime.dispatch();
    const bytes = try runtime.copy();
    try runtime.receipts(bytes);
    runtime.commit();
    return bytes;
}
fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
fn aligned(n: usize) usize { return std.mem.alignForward(usize, n, quantum); }
pub fn overlap(a: usize, b: usize, c: usize, d: usize) bool { return a < b and c < d and a < d and c < b; }
fn clash(live: []const [2]usize, lo: usize, hi: usize) bool {
    for (live) |r| if (overlap(lo, hi, r[0], r[1])) return true;
    return false;
}
pub fn victim(s: State, base: usize, size: usize) bool { return overlap(s.base, s.base + s.len, base, base + size); }
fn less(_: void, a: State, b: State) bool { return a.base < b.base or (a.base == b.base and a.len < b.len); }
fn unionRows(states: []const State, cut: ?[2]usize) usize {
    var end: usize = 0;
    var total: usize = 0;
    for (states) |s| {
        if (cut) |c| if (victim(s, c[0], c[1])) { continue; };
        total += (s.base + s.len) -| @max(end, s.base);
        end = @max(end, s.base + s.len);
    }
    return total;
}
fn better(a: Plan, b: Plan) bool {
    if (a.lost != b.lost) return a.lost < b.lost;
    if (a.reuse_loss != b.reuse_loss) return a.reuse_loss < b.reuse_loss;
    if (a.recency_loss != b.recency_loss) return a.recency_loss < b.recency_loss;
    if (a.victims != b.victims) return a.victims < b.victims;
    return a.base < b.base;
}
fn evaluate(states: []const State, live: []const [2]usize, cap: usize, size: usize, base: usize, before: usize) ?Plan {
    if (size > cap or base > cap - size or clash(live, base, base + size)) return null;
    var p: Plan = .{ .base = base, .lost = 0 };
    for (states) |s| if (victim(s, base, size)) {
        // Do not invalidate even a longer snapshot whose prefix still belongs to a live stream.
        if (s.protected or clash(live, s.base, s.base + s.len)) return null;
        p.victims += 1;
        p.reuse_loss +|= @as(u64, @intCast(s.len)) *| @min(s.restores, 8);
        p.recency_loss +|= s.used;
    };
    p.lost = before - unionRows(states, .{ base, size });
    return p;
}
fn firstFit(states: []const State, live: []const [2]usize, size: usize, cap: usize, move: ?Move) ?usize {
    if (size > cap) return null;
    var base: usize = 0;
    while (base <= cap - size) {
        var end = base;
        for (live) |r| if (overlap(base, base + size, r[0], r[1])) { end = @max(end, r[1]); };
        for (states) |s| {
            const sb = if (move != null and s.base == move.?.from) move.?.to else s.base;
            if (overlap(base, base + size, sb, sb + s.len)) end = @max(end, sb + s.len);
        }
        if (end == base) return base;
        base = aligned(end);
    }
    return null;
}
fn budget(evals: usize, o: Options) bool {
    return evals < o.max_evaluations and (evals % 16 != 0 or now() < o.deadline_ns);
}
/// Complete victim set is selected before the caller changes a single state.
/// Boundary candidates enumerate every distinct aligned overlap set, without scanning the pool.
pub fn plan(gpa: std.mem.Allocator, input: []const State, live: []const [2]usize, size: usize, cap: usize, o: Options) !?Plan {
    if (size == 0 or size > cap or input.len > max_states or live.len > 32) return null;
    for (input) |s| if (s.base > cap or s.len > cap - s.base) return error.BadGeometry;
    for (live) |r| if (r[0] > r[1] or r[1] > cap) return error.BadGeometry;
    const states = try gpa.dupe(State, input);
    defer gpa.free(states);
    std.mem.sort(State, states, {}, less);
    const before = unionRows(states, null);
    var best: ?Plan = null;
    var evals: usize = 0;
    // Just to the right of every interval end and just to the left of its start.
    var starts: std.ArrayList(usize) = .empty;
    defer starts.deinit(gpa);
    try starts.append(gpa, 0);
    for (states) |s| {
        try starts.append(gpa, aligned(s.base + s.len));
        if (s.base >= size) try starts.append(gpa, std.mem.alignBackward(usize, s.base - size, quantum));
    }
    for (live) |r| {
        try starts.append(gpa, aligned(r[1]));
        if (r[0] >= size) try starts.append(gpa, std.mem.alignBackward(usize, r[0] - size, quantum));
    }
    std.mem.sort(usize, starts.items, {}, std.sort.asc(usize));
    var previous: ?usize = null;
    for (starts.items) |b| {
        if (previous != null and previous.? == b) continue;
        previous = b;
        if (!budget(evals, o)) break;
        evals += 1;
        if (evaluate(states, live, cap, size, b, before)) |p| if (best == null or better(p, best.?)) { best = p; };
    }
    if (best == null) return null;
    // Copy only when a nontrivial real loss is avoided; known reuse is required on the donor.
    if (o.max_move_rows > 0 and best.?.lost >= o.min_saved_rows) {
        var i: usize = 0;
        var best_move: ?Plan = null;
        // Current free gaps do not depend on the chosen donor. Build them once, not once per group.
        var occupied: std.ArrayList([2]usize) = .empty;
        defer occupied.deinit(gpa);
        for (states) |s| try occupied.append(gpa, .{ s.base, s.base + s.len });
        try occupied.appendSlice(gpa, live);
        std.mem.sort([2]usize, occupied.items, {}, struct { fn lt(_: void, a: [2]usize, b: [2]usize) bool { return a[0] < b[0]; } }.lt);
        try occupied.append(gpa, .{ cap, cap });
        while (i < states.len and budget(evals, o)) {
            const src = states[i].base;
            var end_i = i;
            var length: usize = 0;
            var reused = false;
            var protected = false;
            while (end_i < states.len and states[end_i].base == src) : (end_i += 1) {
                length = @max(length, states[end_i].len);
                reused = reused or states[end_i].restores > 0;
                protected = protected or states[end_i].protected;
            }
            const start_i = i;
            i = end_i;
            if (!reused or protected or length == 0 or length > o.max_move_rows or clash(live, src, src + length)) continue;
            var independent = true;
            for (states, 0..) |s, j| if ((j < start_i or j >= end_i) and overlap(src, src + length, s.base, s.base + s.len)) { independent = false; break; };
            if (!independent) continue;
            // Moving this group can create space only between its immediate retained/live neighbours.
            // Reject impossible donors before trying any destination or doing a first-fit rescan.
            var left: usize = 0;
            var right = cap;
            for (states, 0..) |s, j| {
                if (j >= start_i and j < end_i) continue;
                if (s.base + s.len <= src) left = @max(left, s.base + s.len);
                if (s.base >= src + length) right = @min(right, s.base);
            }
            for (live) |r| {
                if (r[1] <= r[0]) continue;
                if (r[1] <= src) left = @max(left, r[1]);
                if (r[0] >= src + length) right = @min(right, r[0]);
            }
            if (aligned(left) > right or size > right - aligned(left)) continue;
            // Find already free gaps by merging sorted retained intervals plus live reservations.
            var cursor: usize = 0;
            for (occupied.items) |r| {
                if (r[0] > cursor and r[0] - cursor >= length) {
                    const targets = [_]usize{ aligned(cursor), std.mem.alignBackward(usize, r[0] - length, quantum) };
                    for (targets) |dst| {
                        if (!budget(evals, o)) break;
                        evals += 1;
                        if (dst < cursor or dst > r[0] or length > r[0] - dst or overlap(src, src + length, dst, dst + length)) continue;
                        const mv: Move = .{ .from = src, .to = dst, .len = length };
                        const b = firstFit(states, live, size, cap, mv) orelse continue;
                        const p: Plan = .{ .base = b, .lost = 0, .move = mv };
                        if (best_move == null or length < best_move.?.move.?.len or (length == best_move.?.move.?.len and b < best_move.?.base)) best_move = p;
                    }
                }
                cursor = @max(cursor, r[1]);
            }
        }
        if (best_move) |p| best = p;
    }
    best.?.evaluations = evals;
    best.?.truncated = !budget(evals, o);
    return best;
}

test "whole victim planning keeps borrowed and live-owned snapshots, with no input mutation" {
    const s = [_]State{ .{ .key = 1, .base = 0, .len = 4096, .protected = true }, .{ .key = 2, .base = 4096, .len = 4096 }, .{ .key = 3, .base = 8192, .len = 4096 } };
    const p = (try plan(std.testing.allocator, &s, &.{}, 8192, 12288, .{})).?;
    try std.testing.expectEqual(@as(usize, 4096), p.base);
    try std.testing.expectEqual(@as(usize, 8192), p.lost);
    try std.testing.expectEqual(@as(u64, 2), s[1].key);
    try std.testing.expectEqual(@as(?Plan, null), try plan(std.testing.allocator, &s, &.{.{ 8192, 12288 }}, 8192, 12288, .{}));
}

test "one donor copies shared rows once, but rejects unproven reuse, borrowed, large and multi-hop moves" {
    var s = [_]State{ .{ .key = 1, .base = 0, .len = 2048, .restores = 1 }, .{ .key = 2, .base = 0, .len = 4096 }, .{ .key = 3, .base = 8192, .len = 4096, .protected = true }, .{ .key = 4, .base = 16384, .len = 4096, .protected = true } };
    const opts: Options = .{ .max_move_rows = 4096, .min_saved_rows = 1 };
    const p = (try plan(std.testing.allocator, &s, &.{}, 8192, 24576, opts)).?;
    try std.testing.expect(p.move != null);
    try std.testing.expectEqual(@as(usize, 0), p.move.?.from);
    try std.testing.expectEqual(@as(usize, 4096), p.move.?.len);
    try std.testing.expectEqual(@as(usize, 0), p.lost);
    s[0].restores = 0;
    try std.testing.expect((try plan(std.testing.allocator, &s, &.{}, 8192, 24576, opts)).?.move == null);
    s[0].restores = 1;
    s[0].protected = true;
    try std.testing.expectEqual(@as(?Plan, null), try plan(std.testing.allocator, &s, &.{}, 8192, 24576, opts));
    s[0].protected = false;
    try std.testing.expect((try plan(std.testing.allocator, &s, &.{}, 8192, 24576, .{ .max_move_rows = 2048 })).?.move == null);
}

test "unique loss is not sum of overlapping checkpoint lengths; reuse breaks equal-loss ties" {
    const s = [_]State{ .{ .key = 1, .base = 0, .len = 4096, .restores = 1 }, .{ .key = 2, .base = 0, .len = 2048 }, .{ .key = 3, .base = 6144, .len = 4096 } };
    const p = (try plan(std.testing.allocator, &s, &.{}, 4096, 10240, .{})).?;
    try std.testing.expectEqual(@as(usize, 2048), p.base);
    try std.testing.expectEqual(@as(usize, 2048), p.lost);
    try std.testing.expectEqual(@as(usize, 1), p.victims);
    const tie = [_]State{ s[0], s[2] };
    const t = (try plan(std.testing.allocator, &tie, &.{}, 4096, 10240, .{})).?;
    try std.testing.expectEqual(@as(usize, 4096), t.base);
    try std.testing.expectEqual(@as(u64, 0), t.reuse_loss);
}

test "small randomized geometries agree with exhaustive aligned placement oracle" {
    var prng = std.Random.DefaultPrng.init(0x125b24);
    const rng = prng.random();
    for (0..3000) |_| {
        var s: [6]State = undefined;
        for (&s, 0..) |*v, i| {
            const b = rng.uintLessThan(usize, 14) * quantum;
            v.* = .{ .key = i, .base = b, .len = rng.uintLessThan(usize, (16 * quantum - b) / quantum + 1) * quantum, .protected = rng.uintLessThan(u8, 8) == 0 };
        }
        const size = (1 + rng.uintLessThan(usize, 6)) * quantum;
        const live = [_][2]usize{.{ 7 * quantum, 8 * quantum }};
        const sorted = try std.testing.allocator.dupe(State, &s);
        defer std.testing.allocator.free(sorted);
        std.mem.sort(State, sorted, {}, less);
        const before = unionRows(sorted, null);
        var oracle: ?Plan = null;
        var b: usize = 0;
        while (b + size <= 16 * quantum) : (b += quantum) {
            if (evaluate(sorted, &live, 16 * quantum, size, b, before)) |p| if (oracle == null or better(p, oracle.?)) { oracle = p; };
        }
        const actual = try plan(std.testing.allocator, &s, &live, size, 16 * quantum, .{});
        try std.testing.expectEqual(oracle == null, actual == null);
        if (oracle) |p| {
            try std.testing.expectEqual(p.lost, actual.?.lost);
            try std.testing.expectEqual(p.base, actual.?.base);
        }
    }
}

test "bounded search, capacity checks and malformed geometry fail without changes" {
    const s = [_]State{.{ .key = 1, .base = 0, .len = quantum }};
    try std.testing.expectEqual(@as(?Plan, null), try plan(std.testing.allocator, &s, &.{}, quantum, quantum, .{ .max_evaluations = 0 }));
    try std.testing.expectError(error.BadGeometry, plan(std.testing.allocator, &s, &.{.{ 0, quantum + 1 }}, quantum, quantum, .{}));
    try std.testing.expectEqual(@as(?Plan, null), try plan(std.testing.allocator, &s, &.{}, quantum + 1, quantum, .{}));
}

test "transaction never publishes before every rank receipt, including each injected failure" {
    const Fake = struct {
        stage: usize = 0, fail: usize = 0, published: bool = false,
        pub fn dispatch(f: *@This()) !void { f.stage = 1; if (f.fail == 1) return error.DispatchFailed; }
        pub fn copy(f: *@This()) !usize { std.debug.assert(!f.published and f.stage == 1); f.stage = 2; if (f.fail == 2) return error.CopyFailed; return 32; }
        pub fn receipts(f: *@This(), bytes: usize) !void {
            std.debug.assert(!f.published and bytes == 32 and f.stage == 2);
            // Two follower receipts; a missing, mismatched or failed receipt must stop commit.
            f.stage = 3; if (f.fail == 3) return error.FirstRankFailed;
            f.stage = 4; if (f.fail == 4) return error.SecondRankFailed;
        }
        pub fn commit(f: *@This()) void { std.debug.assert(f.stage == 4 and !f.published); f.stage = 5; f.published = true; }
    };
    for (1..5) |stage| {
        var f: Fake = .{ .fail = stage };
        if (transferAndCommit(&f)) |_| return error.ExpectedFailure else |_| {}
        try std.testing.expect(!f.published);
    }
    var ok: Fake = .{};
    try std.testing.expectEqual(@as(usize, 32), try transferAndCommit(&ok));
    try std.testing.expect(ok.published);
}

test "320MiB permits a reused 330K donor that old 96MiB cannot preserve" {
    const donor_rows: usize = 330000;
    const per_chunk: usize = 890 * quantum;
    const states = [_]State{.{ .key = 1, .base = 0, .len = donor_rows, .restores = 1 }};
    const live = [_][2]usize{.{ 512 * 1024, 1152 * 1024 }};
    // There is room in total, but no 500K free span. Moving this idle donor to
    // the tail opens the front without touching the active middle reservation.
    const old = (try plan(std.testing.allocator, &states, &live, 500000, 1536 * 1024,
        .{ .max_move_rows = (96 * 1024 * 1024 / per_chunk) * quantum })).?;
    try std.testing.expect(old.move == null);
    try std.testing.expectEqual(donor_rows, old.lost);
    const larger = (try plan(std.testing.allocator, &states, &live, 500000, 1536 * 1024,
        .{ .max_move_rows = (hard_copy_bytes / per_chunk) * quantum })).?;
    try std.testing.expect(larger.move != null);
    try std.testing.expectEqual(donor_rows, larger.move.?.len);
    try std.testing.expectEqual(@as(usize, 0), larger.lost);
    try std.testing.expect(!overlap(larger.move.?.to, larger.move.?.to + donor_rows,
                                 live[0][0], live[0][1]));
}
