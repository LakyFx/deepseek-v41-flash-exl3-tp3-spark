//! Pressure-only host planner. No CUDA, tokens, allocation mutations or wall-clock TTL.
const std = @import("std");
pub const quantum = 2048;
pub const hard_copy_bytes: usize = 320 * 1024 * 1024;
pub const max_states = 512;
/// Soft evidence ages with intervening saves/restores, not elapsed idle time.
/// Only a verified continuing tail can receive it. Never a real restore count.
pub const continuation_recent_events: u64 = 128;
pub const State = struct {
    key: u64, base: usize, len: usize,
    restores: u64 = 0, used: u64 = 0, protected: bool = false,
    continuation_at: u64 = 0,
    adaptive_density: u32 = 0, adaptive_at: u64 = 0,
};
pub fn recentContinuation(s: State, clock: u64) bool {
    return s.continuation_at != 0 and clock >= s.continuation_at and
        clock - s.continuation_at < continuation_recent_events;
}
/// Counts of donor groups visited in the EXISTING bounded search. Rejections
/// are first-failing gates, not requests, unique sessions or cache misses.
pub const Donors = struct {
    groups: usize = 0, direct: usize = 0, hinted: usize = 0,
    unproven: usize = 0, protected: usize = 0, credit: usize = 0,
    overlap: usize = 0, opening: usize = 0, target: usize = 0, fit: usize = 0,
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
    short_applied: bool = false,
    short_saved_rows: usize = 0,
    short_extra_rows: usize = 0,
    short_evaluations: usize = 0,
    baseline_base: usize = 0,
    baseline_lost: usize = 0,
    donors: Donors = .{},
    continuation_move: bool = false,
    adaptive_applied: bool = false,
    adaptive_before_us: u64 = 0, adaptive_after_us: u64 = 0,
    adaptive_extra_rows: usize = 0, adaptive_evaluations: usize = 0,
    adaptive_geometry_skip: bool = false,
};
pub const Options = struct {
    max_move_rows: usize = 0,
    min_saved_rows: usize = 16384,
    // A bounded CPU search; callers may additionally give an absolute monotonic deadline.
    deadline_ns: u64 = std.math.maxInt(u64),
    max_evaluations: usize = 2048,
    short_reuse: bool = false,
    short_clock: u64 = 0,
    continuation_donors: bool = false,
    continuation_clock: u64 = 0,
    adaptive_prefix: bool = false,
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
    const baseline = best.?;
    var donors: Donors = .{};
    var short_done = false;
    // Copy only when a nontrivial real loss is avoided; known reuse is required on the donor.
    if (o.max_move_rows > 0 and best.?.lost >= o.min_saved_rows) {
        var best_move: ?Plan = null;
        // Current free gaps do not depend on the chosen donor. Build them once, not once per group.
        var occupied: std.ArrayList([2]usize) = .empty;
        defer occupied.deinit(gpa);
        for (states) |s| try occupied.append(gpa, .{ s.base, s.base + s.len });
        try occupied.appendSlice(gpa, live);
        std.mem.sort([2]usize, occupied.items, {}, struct { fn lt(_: void, a: [2]usize, b: [2]usize) bool { return a[0] < b[0]; } }.lt);
        try occupied.append(gpa, .{ cap, cap });
        // First keep the existing proven-donor search exactly in priority. Weak
        // continuation evidence gets only leftover budget, if no such move fits.
        const passes: usize = if (o.continuation_donors) 2 else 1;
        for (0..passes) |pass| {
          if (pass == 1) {
            if (best_move != null) break;
            // D's original no-copy refinement also gets its original priority.
            // If the new rescue finds nothing, that exact decision survives.
            best = shortRefine(states, live, starts.items, before, size, cap, best.?, &evals, o);
            short_done = true;
            if (best.?.lost < o.min_saved_rows or now() >= o.deadline_ns) break;
          }
          var i: usize = 0;
          while (i < states.len and budget(evals, o) and (pass == 0 or now() < o.deadline_ns)) {
            const src = states[i].base;
            var end_i = i;
            var length: usize = 0;
            var reused = false;
            var hinted = false;
            var protected = false;
            while (end_i < states.len and states[end_i].base == src) : (end_i += 1) {
                length = @max(length, states[end_i].len);
                reused = reused or states[end_i].restores > 0;
                hinted = hinted or (o.continuation_donors and recentContinuation(states[end_i], o.continuation_clock));
                protected = protected or states[end_i].protected;
            }
            const start_i = i;
            i = end_i;
            if (pass == 0) {
                donors.groups += 1;
                if (reused) donors.direct += 1 else if (hinted) donors.hinted += 1 else donors.unproven += 1;
            }
            if (pass == 0 and !reused) continue;
            if (pass == 1 and (reused or !hinted)) continue;
            if (protected or clash(live, src, src + length)) { donors.protected += 1; continue; }
            if (length == 0 or length > o.max_move_rows) { donors.credit += 1; continue; }
            var independent = true;
            for (states, 0..) |s, j| if ((j < start_i or j >= end_i) and overlap(src, src + length, s.base, s.base + s.len)) { independent = false; break; };
            if (!independent) { donors.overlap += 1; continue; }
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
            if (aligned(left) > right or size > right - aligned(left)) { donors.opening += 1; continue; }
            // Find already free gaps by merging sorted retained intervals plus live reservations.
            var cursor: usize = 0;
            var target_found = false;
            var fit_found = false;
            for (occupied.items) |r| {
                if (r[0] > cursor and r[0] - cursor >= length) {
                    const targets = [_]usize{ aligned(cursor), std.mem.alignBackward(usize, r[0] - length, quantum) };
                    for (targets) |dst| {
                        if (!budget(evals, o)) break;
                        evals += 1;
                        if (dst < cursor or dst > r[0] or length > r[0] - dst or overlap(src, src + length, dst, dst + length)) continue;
                        target_found = true;
                        const mv: Move = .{ .from = src, .to = dst, .len = length };
                        const b = firstFit(states, live, size, cap, mv) orelse continue;
                        fit_found = true;
                        const p: Plan = .{ .base = b, .lost = 0, .move = mv, .continuation_move = !reused and hinted };
                        if (best_move == null or length < best_move.?.move.?.len or (length == best_move.?.move.?.len and b < best_move.?.base)) best_move = p;
                    }
                }
                cursor = @max(cursor, r[1]);
            }
            // Do not misreport a deadline/evaluation cutoff as a geometry result.
            if (budget(evals, o)) {
                if (!target_found) donors.target += 1 else if (!fit_found) donors.fit += 1;
            }
          }
        }
        if (best_move) |p| best = p;
    }
    if (!short_done) best = shortRefine(states, live, starts.items, before, size, cap, best.?, &evals, o);
    best = adaptiveRefine(states, live, starts.items, before, size, cap, best.?, &evals, o);
    best.?.evaluations = evals;
    best.?.truncated = !budget(evals, o);
    best.?.baseline_base = baseline.base;
    best.?.baseline_lost = baseline.lost;
    best.?.donors = donors;
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

// D: pressure-only refinement; never runs during prefill/decode or with free placement.
const short_rows: usize = 32768;
const short_extra_cap: usize = 32768;
const short_eval_cap: usize = 128;
const short_recent_events: u64 = 512;

const ShortGain = struct { saved: usize = 0, extra: usize = 0, value: u64 = 0 };

/// Sorted aliases share a physical base. Count proven savings once per base.
/// An alternative cannot newly invalidate ANY long-group or reused-group state.
fn shortGain(states: []const State, baseline: Plan, candidate: Plan, size: usize, clock: u64) ?ShortGain {
    const extra = candidate.lost -| baseline.lost;
    if (extra > short_extra_cap) return null;
    var gain: ShortGain = .{ .extra = extra };
    var i: usize = 0;
    while (i < states.len) {
        const first = i;
        var length: usize = 0;
        var reused = false;
        while (i < states.len and states[i].base == states[first].base) : (i += 1) {
            length = @max(length, states[i].len);
            reused = reused or states[i].restores > 0;
        }
        var saved: usize = 0;
        var value: u64 = 0;
        for (states[first..i]) |s| {
            const was = victim(s, baseline.base, size);
            const becomes = victim(s, candidate.base, size);
            if (becomes and !was and (length > short_rows or reused)) return null;
            if (length <= short_rows and was and !becomes and s.restores > 0 and clock -| s.used <= short_recent_events) {
                saved = @max(saved, s.len);
                value = @max(value, @as(u64, @intCast(s.len)) *| @min(s.restores, 8));
            }
        }
        gain.saved +|= saved;
        gain.value +|= value;
    }
    // A byte/row cap alone is insufficient: the proven reuse must justify the loss.
    return if (gain.saved > 0 and gain.value > extra) gain else null;
}

fn shortRefine(states: []const State, live: []const [2]usize, starts: []const usize, before: usize, size: usize, cap: usize, baseline: Plan, evals: *usize, o: Options) Plan {
    if (!o.short_reuse or baseline.move != null or baseline.lost == 0 or !budget(evals.*, o) or now() >= o.deadline_ns) return baseline;
    var clock = o.short_clock;
    for (states) |s| clock = @max(clock, s.used);
    // Skip the scan if no recently restored short state is at risk.
    const trigger = shortGain(states, baseline, .{ .base = cap, .lost = baseline.lost }, size, clock) orelse return baseline;
    _ = trigger;
    var best = baseline;
    var best_value: u64 = 0;
    var scanned: usize = 0;
    var previous: ?usize = null;
    for (starts) |base| {
        if (previous != null and previous.? == base) continue;
        previous = base;
        if (scanned >= short_eval_cap or !budget(evals.*, o) or now() >= o.deadline_ns) break;
        evals.* += 1;
        scanned += 1;
        const p = evaluate(states, live, cap, size, base, before) orelse continue;
        const gain = shortGain(states, baseline, p, size, clock) orelse continue;
        const value = gain.value - gain.extra;
        if (value > best_value or (value == best_value and better(p, best))) {
            best = p;
            best.short_applied = true;
            best.short_saved_rows = gain.saved;
            best.short_extra_rows = gain.extra;
            best_value = value;
        }
    }
    best.short_evaluations = scanned;
    return best;
}

test "D saves proven short reuse by sacrificing bounded cold short rows" {
    const q = quantum;
    const s = [_]State{
        .{ .key = 1, .base = 0, .len = q, .restores = 4, .used = 10 },
        .{ .key = 2, .base = 4*q, .len = 2*q },
        .{ .key = 3, .base = 8*q, .len = 4*q, .protected = true },
    };
    const b = (try plan(std.testing.allocator, &s, &.{}, 4*q, 12*q, .{})).?;
    const d = (try plan(std.testing.allocator, &s, &.{}, 4*q, 12*q, .{ .short_reuse = true, .short_clock = 10 })).?;
    try std.testing.expect(victim(s[0], b.base, 4*q));
    try std.testing.expect(d.short_applied and !victim(s[0], d.base, 4*q));
    try std.testing.expectEqual(q, d.short_extra_rows);
    try std.testing.expectEqual(q, d.short_saved_rows);
    try std.testing.expectEqual(@as(usize, 2*q), d.lost);
    try std.testing.expect(d.move == null);
}

test "D leaves copies, expired reuse, unproven short states and exhausted searches unchanged" {
    const q = quantum;
    var s = [_]State{
        .{ .key = 1, .base = 0, .len = q, .restores = 4, .used = 10 },
        .{ .key = 2, .base = 4*q, .len = 2*q },
        .{ .key = 3, .base = 8*q, .len = 4*q, .protected = true },
    };
    const b = (try plan(std.testing.allocator, &s, &.{}, 4*q, 12*q, .{})).?;
    const expired = (try plan(std.testing.allocator, &s, &.{}, 4*q, 12*q, .{ .short_reuse = true, .short_clock = 523 })).?;
    try std.testing.expectEqual(b.base, expired.base);
    const exhausted = (try plan(std.testing.allocator, &s, &.{}, 4*q, 12*q, .{ .short_reuse = true, .max_evaluations = b.evaluations })).?;
    try std.testing.expectEqual(b.base, exhausted.base);
    try std.testing.expect(!exhausted.short_applied);
    s[0].restores = 0;
    const cold = (try plan(std.testing.allocator, &s, &.{}, 4*q, 12*q, .{ .short_reuse = true })).?;
    try std.testing.expect(!cold.short_applied);
    var e: usize = 0;
    const copied: Plan = .{ .base = 0, .lost = q, .move = .{ .from = 0, .to = q, .len = q } };
    try std.testing.expectEqual(copied, shortRefine(&s, &.{}, &.{0}, q, q, 12*q, copied, &e, .{ .short_reuse = true }));
}

test "D aliases do not multiply savings or disguise a long/reused victim" {
    const q = quantum;
    const b: Plan = .{ .base = 0, .lost = q };
    const c: Plan = .{ .base = 4*q, .lost = 2*q };
    var s = [_]State{
        .{ .key = 1, .base = 0, .len = q, .restores = 4, .used = 10 },
        .{ .key = 2, .base = 0, .len = q, .restores = 4, .used = 10 },
        .{ .key = 3, .base = 4*q, .len = 2*q },
        .{ .key = 4, .base = 4*q, .len = q },
    };
    try std.testing.expectEqual(q, shortGain(&s, b, c, q, 10).?.saved);
    try std.testing.expectEqual(@as(u64, 4*q), shortGain(&s, b, c, q, 10).?.value);
    s[3].len = short_rows + q;
    try std.testing.expectEqual(@as(?ShortGain, null), shortGain(&s, b, c, q, 10));
    s[3].len = q; s[3].restores = 1;
    try std.testing.expectEqual(@as(?ShortGain, null), shortGain(&s, b, c, q, 10));
    s[3].restores = 0;
    var costly = c; costly.lost = short_extra_cap + 2*q;
    try std.testing.expectEqual(@as(?ShortGain, null), shortGain(&s, b, costly, q, 10));
}

test "D randomized contract: no new long/reused victims, no extra copies, immutable input" {
    var prng = std.Random.DefaultPrng.init(0xd1261025);
    const rng = prng.random();
    for (0..5000) |_| {
        var s: [12]State = undefined;
        for (&s, 0..) |*v, i| {
            const b = rng.uintLessThan(usize, 60) * quantum;
            v.* = .{ .key = i, .base = b, .len = rng.uintLessThan(usize, (64*quantum-b)/quantum+1)*quantum,
                .restores = rng.uintLessThan(u64, 5), .used = rng.uintLessThan(u64, 1000), .protected = rng.uintLessThan(u8, 12) == 0 };
        }
        const original = s;
        const live = [_][2]usize{.{ 30*quantum, 31*quantum }};
        const size = (1+rng.uintLessThan(usize, 10))*quantum;
        const b = try plan(std.testing.allocator, &s, &live, size, 64*quantum, .{});
        const d = try plan(std.testing.allocator, &s, &live, size, 64*quantum, .{ .short_reuse = true, .short_clock = 1000 });
        try std.testing.expectEqualDeep(original, s);
        try std.testing.expectEqual(b == null, d == null);
        if (b == null) continue;
        try std.testing.expect(d.?.move == null and d.?.short_evaluations <= short_eval_cap and d.?.evaluations <= 2048);
        if (!d.?.short_applied) { try std.testing.expectEqual(b.?.base, d.?.base); continue; }
        try std.testing.expect(d.?.lost <= b.?.lost + short_extra_cap);
        try std.testing.expect(!clash(&live, d.?.base, d.?.base+size));
        for (s) |v| {
            var group_long = false; var group_reused = false;
            for (s) |a| if (a.base == v.base) { group_long = group_long or a.len > short_rows; group_reused = group_reused or a.restores > 0; };
            if (victim(v, d.?.base, size)) {
                try std.testing.expect(!v.protected and !clash(&live, v.base, v.base+v.len));
                if (group_long or group_reused) try std.testing.expect(victim(v, b.?.base, size));
            }
        }
    }
}

test "X2 rescues a verified copied 48K continuation without sacrificing another prefix" {
    var states = [_]State{.{ .key = 3742, .base = 0, .len = 47518, .continuation_at = 10 }};
    const live = [_][2]usize{.{ 98304, 393216 }};
    const cold = (try plan(std.testing.allocator, &states, &live, 88064, 458752, .{ .max_move_rows = 330000 })).?;
    try std.testing.expect(cold.move == null and cold.lost == 47518);
    const opts: Options = .{ .max_move_rows = 330000, .continuation_donors = true, .continuation_clock = 20 };
    const rescued = (try plan(std.testing.allocator, &states, &live, 88064, 458752, opts)).?;
    try std.testing.expect(rescued.move != null and rescued.continuation_move);
    try std.testing.expectEqual(@as(usize, 0), rescued.lost);
    try std.testing.expectEqual(@as(usize, 47518), rescued.baseline_lost);
    try std.testing.expectEqual(@as(usize, 1), rescued.donors.hinted);
    try std.testing.expectEqual(@as(u64, 0), states[0].restores);
    var expired = opts;
    expired.continuation_clock = 10 + continuation_recent_events;
    const aged = (try plan(std.testing.allocator, &states, &live, 88064, 458752, expired)).?;
    try std.testing.expect(aged.move == null and aged.lost == cold.lost);
    states[0].continuation_at = 0;
    try std.testing.expect((try plan(std.testing.allocator, &states, &live, 88064, 458752, opts)).?.move == null);
}

test "X2 continuing 301K session can move; cold cron, copy credit and small-loss gate stay bounded" {
    var states = [_]State{.{ .key = 1, .base = 0, .len = 301000, .continuation_at = 1 }};
    const live = [_][2]usize{.{ 512*1024, 1152*1024 }};
    const opts: Options = .{ .max_move_rows = 330000, .continuation_donors = true, .continuation_clock = 2 };
    const p = (try plan(std.testing.allocator, &states, &live, 500000, 1536*1024, opts)).?;
    try std.testing.expect(p.continuation_move and p.move.?.len == 301000 and p.lost == 0);
    states[0].continuation_at = 0;
    const cron = (try plan(std.testing.allocator, &states, &live, 500000, 1536*1024, opts)).?;
    try std.testing.expect(cron.move == null and cron.donors.unproven == 1);
    states[0].continuation_at = 1;
    var bounded = opts;
    bounded.max_move_rows = 300000;
    const too_big = (try plan(std.testing.allocator, &states, &live, 500000, 1536*1024, bounded)).?;
    try std.testing.expect(too_big.move == null and too_big.donors.credit == 1);
    var small = [_]State{.{ .key = 1, .base = 0, .len = 1485, .continuation_at = 1 }};
    const tiny = (try plan(std.testing.allocator, &small, &.{.{8192, 32768}}, 8192, 34816, opts)).?;
    try std.testing.expect(tiny.move == null and tiny.lost == 1485 and tiny.donors.groups == 0);
}

test "X2 hints do not bypass live ownership, overlapping groups or missing free destination" {
    const opts: Options = .{ .max_move_rows = 330000, .continuation_donors = true, .continuation_clock = 2 };
    var states = [_]State{.{ .key = 1, .base = 0, .len = 47518, .continuation_at = 1, .protected = true }};
    const live = [_][2]usize{.{ 98304, 393216 }};
    try std.testing.expectEqual(@as(?Plan, null), try plan(std.testing.allocator, &states, &live, 88064, 458752, opts));
    states[0].protected = false;
    const no_target = (try plan(std.testing.allocator, &states, &live, 88064, 430080, opts)).?;
    // A side gap can hold the donor, but moving there cannot open the requested
    // 88K span. Distinguish "free target exists" from "whole placement fits".
    try std.testing.expect(no_target.move == null and no_target.donors.fit == 1);
    const aliases = [_]State{ states[0], .{ .key = 2, .base = 2048, .len = 16384 } };
    const overlap_rejected = (try plan(std.testing.allocator, &aliases, &live, 88064, 458752, opts)).?;
    try std.testing.expect(overlap_rejected.move == null and overlap_rejected.donors.overlap == 1);
    try std.testing.expect(!recentContinuation(.{ .key = 1, .base = 0, .len = 1, .continuation_at = 3 }, 2));
}

test "X2 randomized hinted donor contract: no new victims, unchanged no-copy decision, bounded work" {
    var prng = std.Random.DefaultPrng.init(0x126e2);
    const rng = prng.random();
    for (0..3000) |_| {
        var states: [10]State = undefined;
        for (&states, 0..) |*s, i| {
            const base = rng.uintLessThan(usize, 64) * quantum;
            s.* = .{ .key = i, .base = base, .len = rng.uintLessThan(usize, (64*quantum-base)/quantum+1)*quantum,
                .restores = rng.uintLessThan(u64, 3), .used = rng.uintLessThan(u64, 200),
                .continuation_at = if (rng.boolean()) rng.uintLessThan(u64, 200) else 0, .protected = rng.uintLessThan(u8, 12) == 0 };
        }
        const original = states;
        const live = [_][2]usize{.{ 30*quantum, 31*quantum }};
        const size = (1+rng.uintLessThan(usize, 10))*quantum;
        const off: Options = .{ .max_move_rows = 16*quantum, .short_reuse = true, .short_clock = 200 };
        var on = off;
        on.continuation_donors = true; on.continuation_clock = 200;
        const baseline = try plan(std.testing.allocator, &states, &live, size, 64*quantum, off);
        const actual = try plan(std.testing.allocator, &states, &live, size, 64*quantum, on);
        try std.testing.expectEqualDeep(original, states);
        try std.testing.expectEqual(baseline == null, actual == null);
        if (baseline == null) continue;
        const p = actual.?;
        try std.testing.expect(p.evaluations <= 2048 and p.short_evaluations <= short_eval_cap);
        if (!p.continuation_move) {
            try std.testing.expectEqual(baseline.?.base, p.base);
            try std.testing.expectEqualDeep(baseline.?.move, p.move);
            continue;
        }
        try std.testing.expect(baseline.?.move == null and p.move != null and p.lost == 0 and p.victims == 0);
        const mv = p.move.?;
        try std.testing.expect(mv.len <= off.max_move_rows and !clash(&live, mv.from, mv.from+mv.len) and !clash(&live, mv.to, mv.to+mv.len));
        try std.testing.expect(firstFit(&states, &live, size, 64*quantum, mv) != null);
        var proven_hint = false;
        for (states) |s| {
            try std.testing.expect(!overlap(mv.to, mv.to+mv.len, s.base, s.base+s.len));
            if (s.base == mv.from) {
                try std.testing.expect(!s.protected and s.restores == 0);
                proven_hint = proven_hint or recentContinuation(s, on.continuation_clock);
            }
        }
        try std.testing.expect(proven_hint);
    }
}

// X3: refine a complete X2 no-copy plan with learned recomputation cost.
// Never introduces a new copy, live victim, unbounded search or GPU operation.
const adaptive_eval_cap: usize = 128;
const adaptive_extra_cap: usize = 32768;

fn adaptiveDensity(s: State, clock: u64) u64 {
    const shift: u5 = @intCast(@min((clock -| s.adaptive_at) / 128, 16));
    // Neutral floor for unobserved/forgotten work: not an admission rejection.
    return @max(16, @as(u64, s.adaptive_density) >> shift);
}

/// Native aliases have the same physical base; count them once. If a future
/// backend supplies intersecting DIFFERENT-base groups, keep the X2 decision
/// rather than double-charge their shared rows using an invalid approximation.
fn adaptiveGeometry(states: []const State) bool {
    var end: usize = 0;
    var i: usize = 0;
    while (i < states.len) {
        const b = states[i].base;
        if (b < end) return false;
        var n: usize = 0;
        while (i < states.len and states[i].base == b) : (i += 1) n = @max(n, states[i].len);
        end = b + n;
    }
    return true;
}

fn adaptiveLoss(states: []const State, base: usize, size: usize, clock: u64) u64 {
    var cost: u64 = 0;
    var i: usize = 0;
    while (i < states.len) {
        const b = states[i].base;
        var total: usize = 0;
        var kept: usize = 0;
        var density: u64 = 16;
        while (i < states.len and states[i].base == b) : (i += 1) {
            const s = states[i];
            total = @max(total, s.len);
            if (!victim(s, base, size)) kept = @max(kept, s.len);
            density = @max(density, adaptiveDensity(s, clock));
        }
        cost +|= @as(u64, @intCast(total - kept)) *| density;
    }
    return cost;
}

fn adaptiveRefine(states: []const State, live: []const [2]usize, starts: []const usize, before: usize, size: usize, cap: usize, baseline: Plan, evals: *usize, o: Options) Plan {
    if (!o.adaptive_prefix or baseline.move != null or baseline.lost == 0 or !budget(evals.*, o) or now() >= o.deadline_ns) return baseline;
    var best = baseline;
    if (!adaptiveGeometry(states)) { best.adaptive_geometry_skip = true; return best; }
    var clock = o.continuation_clock;
    var learned = false;
    for (states) |s| { clock = @max(clock, s.used); learned = learned or s.adaptive_density > 16; }
    if (!learned) return baseline;
    const original = adaptiveLoss(states, baseline.base, size, clock);
    var best_cost = original;
    const extra = @min(adaptive_extra_cap, @max(quantum, baseline.lost / 8));
    // Deduplicate aliases' boundary candidates, then cover the whole address
    // range instead of spending all 128 evaluations on its lowest addresses.
    var unique: [2 * max_states + 65]usize = undefined;
    var count: usize = 0;
    for (starts) |b| {
        if (count != 0 and unique[count-1] == b) continue;
        if (count == unique.len) return baseline;
        unique[count]=b; count+=1;
    }
    const limit=@min(adaptive_eval_cap,count);
    var scanned: usize = 0;
    for (0..limit) |j| {
        const idx=if (limit < count and limit > 1) j*(count-1)/(limit-1) else j;
        const b=unique[idx];
        if (b == baseline.base) continue;
        if (scanned >= adaptive_eval_cap or !budget(evals.*, o) or now() >= o.deadline_ns) break;
        evals.* += 1;
        scanned += 1;
        const p = evaluate(states, live, cap, size, b, before) orelse continue;
        if (p.lost > baseline.lost +| extra) continue;
        const cost = adaptiveLoss(states, p.base, size, clock);
        // Hysteresis: do not swap placements for a tiny noisy score difference.
        if (cost +| original / 20 < best_cost) { best = p; best_cost = cost; }
    }
    best.adaptive_applied = best.base != baseline.base;
    best.adaptive_before_us = original;
    best.adaptive_after_us = best_cost;
    best.adaptive_extra_rows = best.lost -| baseline.lost;
    best.adaptive_evaluations = scanned;
    return best;
}

test "X3 protects a fresh returning medium prefix within bounded extra loss" {
    const q = quantum;
    const s = [_]State{
        .{ .key=1, .base=0, .len=40*q, .adaptive_density=400, .adaptive_at=100, .used=100 },
        .{ .key=2, .base=40*q, .len=42*q, .adaptive_density=20, .adaptive_at=100, .used=90 },
    };
    const off = (try plan(std.testing.allocator, &s, &.{}, 40*q, 82*q, .{})).?;
    const on = (try plan(std.testing.allocator, &s, &.{}, 40*q, 82*q, .{ .adaptive_prefix=true, .continuation_clock=100 })).?;
    try std.testing.expectEqual(@as(usize, 0), off.base);
    try std.testing.expect(on.adaptive_applied and on.base >= 40*q);
    try std.testing.expect(on.adaptive_after_us < on.adaptive_before_us);
    try std.testing.expect(on.adaptive_extra_rows <= 32768 and on.move == null);
}
test "X3 old confidence decays and cannot sacrifice a much larger hot group" {
    const q = quantum;
    const s = [_]State{
        .{ .key=1, .base=0, .len=40*q, .adaptive_density=400, .adaptive_at=1, .used=1 },
        .{ .key=2, .base=40*q, .len=80*q, .adaptive_density=500, .adaptive_at=1, .used=1 },
    };
    const p = (try plan(std.testing.allocator, &s, &.{}, 40*q, 120*q, .{ .adaptive_prefix=true, .continuation_clock=4096 })).?;
    try std.testing.expect(!p.adaptive_applied);
    try std.testing.expectEqual(@as(usize, 0), p.base);
}
test "X3 aliases are charged once and unsupported overlap falls back" {
    const q = quantum;
    const s = [_]State{
        .{ .key=1, .base=0, .len=8*q, .adaptive_density=100 },
        .{ .key=2, .base=0, .len=8*q, .adaptive_density=100 },
        .{ .key=3, .base=8*q, .len=8*q, .adaptive_density=20 },
    };
    try std.testing.expectEqual(@as(u64, 8*q*100), adaptiveLoss(&s, 0, q, 0));
    var cross = s;
    cross[2].base = 4*q;
    try std.testing.expect(!adaptiveGeometry(&cross));
    const p = (try plan(std.testing.allocator, &cross, &.{}, 4*q, 12*q, .{ .adaptive_prefix=true })).?;
    try std.testing.expect(p.adaptive_geometry_skip and !p.adaptive_applied);
}
test "X3 live protection and exhausted search budget survive adaptive scoring" {
    const q = quantum;
    const s = [_]State{
        .{ .key=1, .base=0, .len=40*q, .adaptive_density=400 },
        .{ .key=2, .base=40*q, .len=42*q, .adaptive_density=20, .protected=true },
    };
    const p = (try plan(std.testing.allocator, &s, &.{}, 40*q, 82*q, .{ .adaptive_prefix=true })).?;
    try std.testing.expectEqual(@as(usize, 0), p.base);
    const off = (try plan(std.testing.allocator, &s, &.{}, 40*q, 82*q, .{})).?;
    const bounded = (try plan(std.testing.allocator, &s, &.{}, 40*q, 82*q, .{ .adaptive_prefix=true, .max_evaluations=off.evaluations })).?;
    try std.testing.expectEqual(off.base, bounded.base);
}

test "X3 randomized geometry keeps bounds, complete victim sets and alias conservation" {
    var rng = std.Random.DefaultPrng.init(0x5833);
    const r = rng.random();
    for (0..3000) |_| {
        var s: [12]State = undefined;
        for (&s,0..) |*v,i| v.* = .{ .key=i+1, .base=i*8*quantum,
            .len=r.intRangeAtMost(usize,1,8)*quantum,
            .adaptive_density=r.intRangeAtMost(u32,0,1000), .adaptive_at=90,
            .used=90, .protected=r.uintLessThan(u8,16)==0 };
        const size = r.intRangeAtMost(usize,1,12)*quantum;
        const off = try plan(std.testing.allocator,&s,&.{},size,96*quantum,.{});
        const on = try plan(std.testing.allocator,&s,&.{},size,96*quantum,.{ .adaptive_prefix=true, .continuation_clock=100 });
        if (off == null) { try std.testing.expect(on == null); continue; }
        const p = on.?;
        try std.testing.expect(p.evaluations <= 2048 and p.adaptive_evaluations <= 128);
        if (p.adaptive_applied) {
            try std.testing.expect(p.move == null);
            try std.testing.expect(p.lost <= off.?.lost + @min(adaptive_extra_cap,@max(quantum,off.?.lost/8)));
            try std.testing.expect(p.adaptive_after_us < p.adaptive_before_us);
        }
        for (s) |v| if (v.protected) try std.testing.expect(!victim(v,p.base,size));
    }
}
