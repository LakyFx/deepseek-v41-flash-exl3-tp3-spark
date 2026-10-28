//! X3: bounded CPU-only retention evidence. Not a restore authority.
//! Updates happen on request/restore/completion, never in a token loop.
const std = @import("std");
pub const capacity = 2048;
const ways = 4;
const model_capacity = 64;
const age_quantum = 128;
pub const Record = struct {
    id: u64 = 0,
    namespace: u64 = 0,
    last_request: u64 = 0,
    last_return: u64 = 0,
    completed: u64 = 0,
    last_eviction: u64 = 0,
    interval: u32 = 16,
    heat: u16 = 0,
    returns: u16 = 0,
    ghost_returns: u16 = 0,
};
const Model = struct {
    namespace: u64 = 0,
    valid: bool = false,
    used: u64 = 0,
    recency_weight: u16 = 128,
    // Robust EWMA of observed microseconds/new input token, by length bucket.
    cost: [8]u32 = @splat(500),
    observations: [8]u32 = @splat(0),
};
pub const Counters = struct {
    requests: u64 = 0, returns: u64 = 0, forks: u64 = 0,
    ghosts: u64 = 0, replaced: u64 = 0, cost_samples: u64 = 0,
};
pub const History = struct {
    records: [capacity]Record = @splat(.{}),
    models: [model_capacity]Model = @splat(.{}),
    counters: Counters = .{},

    fn index(namespace: u64, id: u64, n: usize) usize {
        // Fixed probes; no token hashing, allocation or open-ended chains.
        const h = (namespace ^ (id *% 0x9e3779b97f4a7c15)) *% 0xbf58476d1ce4e5b9;
        return @as(usize, @truncate(h ^ (h >> 32))) % (n / ways) * ways;
    }
    pub fn get(h: *History, namespace: u64, id: u64) ?*Record {
        if (id == 0) return null;
        const b = index(namespace, id, capacity);
        for (h.records[b..][0..ways]) |*r| if (r.id == id and r.namespace == namespace) return r;
        return null;
    }
    fn put(h: *History, namespace: u64, id: u64, now: u64) *Record {
        if (h.get(namespace, id)) |r| return r;
        const b = index(namespace, id, capacity);
        var slot = &h.records[b];
        for (h.records[b..][0..ways]) |*r| {
            if (r.id == 0) { slot = r; break; }
            if (r.last_request < slot.last_request) slot = r;
        }
        if (slot.id != 0) h.counters.replaced +|= 1;
        slot.* = .{ .id = id, .namespace = namespace, .last_request = now, .last_return = now };
        return slot;
    }
    fn model(h: *History, namespace: u64, now: u64) *Model {
        const b = index(namespace, 1, model_capacity);
        var slot = &h.models[b];
        for (h.models[b..][0..ways]) |*m| {
            if (m.valid and m.namespace == namespace) { m.used = now; return m; }
            if (!m.valid or m.used < slot.used) slot = m;
        }
        slot.* = .{ .valid = true, .namespace = namespace, .used = now };
        return slot;
    }
    fn faded(value: u16, age: u64) u16 {
        return value >> @as(u4, @intCast(@min(age / age_quantum, 15)));
    }
    pub fn begin(h: *History, namespace: u64, request: u64) u64 {
        _ = h.put(namespace, request, request);
        h.counters.requests +|= 1;
        return request;
    }
    /// Identity must come from an exact successful old-tail check (or a
    /// diagnostic full-span ghost check). A shared header is not a lineage.
    pub fn returned(h: *History, namespace: u64, request: u64, parent: u64, parent_request: u64, ghost: bool, exact_replay: bool) u64 {
        const old = if (h.get(namespace, parent)) |p| p.* else return h.begin(namespace, request);
        var inherited = old;
        const age = request -| old.last_return;
        inherited.heat = @min(256, faded(old.heat, age) +| 48);
        inherited.returns = @min(255, faded(old.returns, age) +| 1);
        inherited.interval = @intCast((@as(u64, old.interval) * 3 + std.math.clamp(age, 1, 1024)) / 4);
        inherited.last_return = request;
        inherited.last_request = request;
        inherited.last_eviction = 0;
        // A second child of the same historical head is a fork. It receives
        // bounded prior confidence but cannot boost the first child's history.
        const same = exact_replay or old.last_request == parent_request;
        const id = if (same) parent else request;
        if (!same) { inherited.heat /= 2; h.counters.forks +|= 1; }
        inherited.id = id;
        if (ghost) {
            inherited.ghost_returns +|= 1;
            h.counters.ghosts +|= 1;
            const m = h.model(namespace, request);
            // Adjust from observed misses, not from requests that have not
            // returned yet (their future is censored, not a failed prediction).
            if (old.returns < 3 and age <= 32) m.recency_weight = @min(224, m.recency_weight + 4)
            else if (old.returns >= 3) m.recency_weight = @max(32, m.recency_weight -| 4);
        }
        h.put(namespace, id, request).* = inherited;
        h.counters.returns +|= 1;
        return id;
    }
    pub fn evicted(h: *History, namespace: u64, id: u64, request: u64) void {
        if (h.get(namespace, id)) |r| r.last_eviction = request;
    }
    fn bucket(tokens: u32) usize {
        var n = tokens / 4096;
        var b: usize = 0;
        while (n > 1 and b < 7) : (b += 1) n >>= 1;
        return b;
    }
    pub fn completed(h: *History, namespace: u64, id: u64, request: u64, input: u32, cached: u32, prefill_ns: u64, ok: bool) void {
        if (!ok) return; // failed/cancelled partial work is not a useful sample
        if (h.get(namespace, id)) |r| r.completed = request;
        const fresh = input -| cached;
        if (fresh < 128 or prefill_ns == 0) return;
        const m = h.model(namespace, request);
        const b = bucket(input);
        const sample = std.math.clamp(prefill_ns / fresh / 1000, 10, 10000);
        // Bound scheduler/load noise before the EWMA; API queue time is absent.
        const clipped = std.math.clamp(sample, @max(10, m.cost[b] / 4), @min(10000, m.cost[b] * 4));
        m.cost[b] = @intCast((@as(u64, m.cost[b]) * 7 + clipped) / 8);
        m.observations[b] +|= 1;
        h.counters.cost_samples +|= 1;
    }
    /// Estimated retained microseconds/token. Bounded, aged and numeric only.
    pub fn density(h: *History, namespace: u64, id: u64, now: u64, tokens: u32) u32 {
        const r = h.get(namespace, id) orelse return 0;
        const m = h.model(namespace, now);
        const age = now -| r.last_return;
        const heat: u64 = faded(r.heat, age);
        const frequency: u64 = @min(224, 16 + heat * 3 / 4);
        const interval: u64 = @max(4, r.interval);
        const recent: u64 = (32 + heat * 3 / 4) * interval / (interval + age);
        const probability = (@as(u64, m.recency_weight) * recent + (256 - @as(u64, m.recency_weight)) * frequency) / 256;
        return @intCast(@min(10000, probability * m.cost[bucket(tokens)] / 256));
    }
};

test "X3 bounded metadata and four-probe namespace isolation" {
    var h: History = .{};
    try std.testing.expect(@sizeOf(History) < 256 * 1024);
    _ = h.begin(1, 1);
    try std.testing.expect(h.get(2, 1) == null);
    for (2..10000) |i| _ = h.begin(1, i);
    try std.testing.expect(h.counters.replaced > 0);
}
test "X3 learns sequential continuation and separates forks" {
    var h: History = .{};
    var id = h.begin(1, 1);
    const cold = h.density(1, id, 1, 90000);
    for (2..12) |i| id = h.returned(1, i, id, i-1, false, false);
    try std.testing.expectEqual(@as(u64, 1), id);
    try std.testing.expect(h.density(1, id, 11, 90000) > cold);
    const before = h.get(1, id).?.*;
    const fork = h.returned(1, 12, id, 5, false, false);
    try std.testing.expectEqual(@as(u64, 12), fork);
    try std.testing.expectEqualDeep(before, h.get(1, id).?.*);
    try std.testing.expect(h.density(1, fork, 12, 90000) < h.density(1, id, 12, 90000));
}
test "X3 old history fades with competing requests, not wall clock TTL" {
    var h: History = .{};
    const id = h.begin(1, 1);
    _ = h.returned(1, 2, id, 1, false, false);
    const hot = h.density(1, id, 2, 301000);
    try std.testing.expectEqual(hot, h.density(1, id, 2, 301000));
    try std.testing.expect(h.density(1, id, 2000, 301000) < hot);
    try std.testing.expect(h.get(1, id) != null);
}
test "X3 verified ghost feedback adapts to recent and frequent returns" {
    var h: History = .{};
    _ = h.begin(1, 1);
    h.evicted(1, 1, 2);
    _ = h.returned(1, 3, 1, 1, true, false);
    try std.testing.expect(h.model(1, 3).recency_weight > 128);
    var id: u64 = 1;
    for (4..9) |i| id = h.returned(1, i, id, i-1, false, false);
    const before = h.model(1, 8).recency_weight;
    _ = h.returned(1, 9, id, 8, true, false);
    try std.testing.expect(h.model(1, 9).recency_weight < before);
    try std.testing.expectEqual(@as(u64, 2), h.counters.ghosts);
}
test "X3 cost learning excludes failed work and clips noisy observations" {
    var h: History = .{};
    _ = h.begin(1, 1);
    h.completed(1, 1, 1, 90000, 0, 90 * std.time.ns_per_s, false);
    try std.testing.expectEqual(@as(u64, 0), h.counters.cost_samples);
    h.completed(1, 1, 1, 90000, 0, 90 * std.time.ns_per_s, true);
    try std.testing.expectEqual(@as(u64, 1), h.counters.cost_samples);
    try std.testing.expect(h.model(1, 1).cost[History.bucket(90000)] > 500);
    try std.testing.expectEqual(@as(u64, 1), h.get(1, 1).?.completed);
}
