//! Bounded position-keyed RNG preparation, overlapping a verify GPU forward.
//! Workers own only copied scalar keys and disjoint CPU buffers. No GPU/stream
//! pointers, mutable sequences or arenas. Ready lookup never waits for workers.
const std = @import("std");
const sampling = @import("sampling.zig");

pub const max_jobs = 32;
pub const workers = 4;
pub const max_bytes = 32 * 1024 * 1024;
pub const Request = struct { seed: u64, position: u64 };

pub const Pool = struct {
    io: std.Io,
    vocab: usize,
    buffer: []f64,
    threads: [workers]std.Thread = undefined,
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    done: std.Io.Condition = .init,
    generation: u64 = 0,
    stop: bool = false,
    remaining: usize = 0,
    count: usize = 0,
    keys: [max_jobs]Request = undefined,
    ready: [max_jobs]std.atomic.Value(bool) = @splat(.init(false)),
    next: std.atomic.Value(usize) = .init(0),
    cancel: std.atomic.Value(bool) = .init(false),

    pub fn create(a: std.mem.Allocator, io: std.Io, vocab: usize) !*Pool {
        if (vocab == 0 or vocab > max_bytes / max_jobs / @sizeOf(f64)) return error.PenaltyMemoryCap;
        const p = try a.create(Pool);
        errdefer a.destroy(p);
        const buffer = try a.alloc(f64, max_jobs * vocab);
        errdefer a.free(buffer);
        p.* = .{ .io = io, .vocab = vocab, .buffer = buffer };
        var started: usize = 0;
        errdefer p.shutdown(started);
        for (&p.threads) |*th| {
            th.* = try std.Thread.spawn(.{ .stack_size = 2 * 1024 * 1024 }, loop, .{p});
            started += 1;
        }
        return p;
    }

    fn shutdown(p: *Pool, n: usize) void {
        p.cancel.store(true, .release);
        p.mutex.lockUncancelable(p.io);
        p.stop = true;
        p.cond.broadcast(p.io);
        p.mutex.unlock(p.io);
        for (p.threads[0..n]) |th| th.join();
    }

    pub fn destroy(p: *Pool, a: std.mem.Allocator) void {
        p.shutdown(workers);
        a.free(p.buffer);
        a.destroy(p);
    }

    /// One engine-thread producer; start/finish delimit buffer lifetime.
    pub fn start(p: *Pool, requests: []const Request) !void {
        if (requests.len > max_jobs) return error.PenaltyJobCap;
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        if (p.remaining != 0 or p.stop) return error.PenaltyPoolBusy;
        @memcpy(p.keys[0..requests.len], requests);
        p.count = requests.len;
        for (&p.ready) |*r| r.store(false, .release);
        p.next.store(0, .monotonic);
        p.cancel.store(false, .release);
        if (requests.len == 0) return;
        p.remaining = workers;
        p.generation += 1;
        p.cond.broadcast(p.io);
    }

    /// Cancel unneeded work and drain bounded libm loops before reuse/free.
    /// Never used to wait for a cache hit: callers use their original sampler.
    pub fn finish(p: *Pool) void {
        p.cancel.store(true, .release);
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        while (p.remaining != 0) p.done.waitUncancelable(p.io, &p.mutex);
    }

    pub fn waitForTest(p: *Pool) void {
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        while (p.remaining != 0) p.done.waitUncancelable(p.io, &p.mutex);
    }

    pub fn cache(p: *Pool) sampling.PenaltyCache {
        return .{ .ptr = p, .lookup = lookup };
    }

    fn lookup(ptr: *anyopaque, seed: u64, position: u64, vocab: usize) ?[]const f64 {
        const p: *Pool = @ptrCast(@alignCast(ptr));
        if (vocab != p.vocab) return null;
        for (p.keys[0..p.count], 0..) |key, i| {
            if (key.seed == seed and key.position == position and p.ready[i].load(.acquire))
                return p.buffer[i * vocab ..][0..vocab];
        }
        return null;
    }

    fn loop(p: *Pool) void {
        var seen: u64 = 0;
        while (true) {
            p.mutex.lockUncancelable(p.io);
            while (!p.stop and seen == p.generation) p.cond.waitUncancelable(p.io, &p.mutex);
            if (p.stop) { p.mutex.unlock(p.io); return; }
            seen = p.generation;
            p.mutex.unlock(p.io);
            p.work();
            p.mutex.lockUncancelable(p.io);
            p.remaining -= 1;
            p.done.broadcast(p.io);
            p.mutex.unlock(p.io);
        }
    }

    fn work(p: *Pool) void {
        while (!p.cancel.load(.acquire)) {
            const job = p.next.fetchAdd(1, .monotonic);
            if (job >= p.count) return;
            const key = p.keys[job];
            const values = p.buffer[job * p.vocab ..][0..p.vocab];
            for (values, 0..) |*v, id| {
                if (id % 256 == 0 and p.cancel.load(.acquire)) return;
                v.* = sampling.keyedPenalty(key.seed, key.position, id);
            }
            p.ready[job].store(true, .release);
        }
    }
};

test "bounded overlap preserves exact RNG and target draws with identity fallback" {
    const a = std.testing.allocator;
    const p = try Pool.create(a, std.testing.io, 1024);
    defer p.destroy(a);
    var workspace: sampling.Workspace = .{};
    defer workspace.deinit(a);
    var row: [1024]f32 = undefined;
    for (&row, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 37)) / 8;
    for (0..12) |generation| {
        const keys = [_]Request{ .{ .seed = generation, .position = 401000 }, .{ .seed = generation, .position = 401001 } };
        try p.start(&keys);
        p.waitForTest();
        const cache = p.cache();
        try std.testing.expect(cache.lookup(cache.ptr, generation + 1, 401000, 1024) == null);
        try std.testing.expect(cache.lookup(cache.ptr, generation, 400999, 1024) == null);
        try std.testing.expect(cache.lookup(cache.ptr, generation, 401000, 1023) == null);
        const penalties = cache.lookup(cache.ptr, generation, 401000, 1024).?;
        for (penalties, 0..) |v, id| try std.testing.expectEqual(@as(u64, @bitCast(sampling.keyedPenalty(generation, 401000, id))), @as(u64, @bitCast(v)));
        for ([_]sampling.Sampling{
            .{ .seed = generation, .temperature = 0.6, .top_k = 0, .top_p = 1 },
            .{ .seed = generation, .temperature = 1.1, .top_k = 0, .top_p = 1, .min_p = 0.2 },
            .{ .seed = generation, .temperature = 0.6, .top_k = 20, .top_p = 0.95 },
            .{ .seed = generation, .temperature = 0.6, .top_k = 0, .top_p = 0.95 },
        }) |sm| {
            workspace.penalty_cache = null;
            const want = try sampling.sampleRowWithWorkspace(&workspace, a, &row, 401000, sm);
            workspace.penalty_cache = cache;
            try std.testing.expectEqual(want, try sampling.sampleRowWithWorkspace(&workspace, a, &row, 401000, sm));
        }
        workspace.penalty_cache = null;
        p.finish();
    }
}

test "cancel drain reuse and hard caps" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.PenaltyMemoryCap, Pool.create(a, std.testing.io, max_bytes));
    const p = try Pool.create(a, std.testing.io, 4096);
    defer p.destroy(a);
    var keys: [max_jobs + 1]Request = @splat(.{ .seed = 42, .position = 99 });
    try std.testing.expectError(error.PenaltyJobCap, p.start(&keys));
    for (0..40) |_| {
        try p.start(keys[0..max_jobs]);
        p.finish();
        try p.start(&.{});
        p.finish();
    }
    try p.start(keys[0..2]);
    // Destruction while work is outstanding owns the cancellation and joins.
}
