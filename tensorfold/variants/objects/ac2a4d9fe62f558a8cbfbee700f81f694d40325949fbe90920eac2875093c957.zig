//! Host-only aggregate credit for NEW idle relocations, never copy-on-resume.
const std = @import("std");
pub const capacity: u64 = 320 * 1024 * 1024;
pub const refill_per_second: u64 = 8 * 1024 * 1024;
pub const refill_seconds: u64 = (capacity + refill_per_second - 1) / refill_per_second;
pub const Budget = struct {
    credit: u64 = capacity,
    last_ns: ?u64 = null,
    fraction: u64 = 0,
    charged: u64 = 0,
    moves: u64 = 0,

    pub fn available(b: *Budget, now: u64) u64 {
        const previous = b.last_ns orelse {
            b.last_ns = now;
            return b.credit;
        };
        if (now <= previous) return b.credit;
        b.last_ns = now;
        // One full refill interval fills the bucket from zero. Bound arithmetic and
        // discard excess time/credit: long idle must never buy a multi-GiB burst.
        const elapsed = @min(now - previous, refill_seconds * std.time.ns_per_s);
        const scaled = elapsed * refill_per_second + b.fraction;
        b.credit = @min(capacity, b.credit + scaled / std.time.ns_per_s);
        b.fraction = if (b.credit == capacity) 0 else scaled % std.time.ns_per_s;
        return b.credit;
    }

    pub fn consume(b: *Budget, now: u64, bytes: u64) bool {
        const credit = b.available(now);
        if (bytes == 0 or bytes > credit or bytes > capacity) return false;
        b.credit -= bytes;
        b.charged +|= bytes;
        b.moves +|= 1;
        return true;
    }
};

test "aggregate move burst cannot exceed one capacity at the same time" {
    var b: Budget = .{};
    for (0..12) |i| try std.testing.expect(b.consume(0, if (i == 11) b.credit else capacity / 12));
    try std.testing.expect(!b.consume(0, 1));
    try std.testing.expectEqual(capacity, b.charged);
    try std.testing.expectEqual(@as(u64, 12), b.moves);
}

test "refill is monotonic fractional bounded and idle does not bank bursts" {
    var b: Budget = .{};
    try std.testing.expect(b.consume(100, capacity));
    try std.testing.expectEqual(@as(u64, 0), b.available(99));
    try std.testing.expectEqual(@as(u64, 0), b.available(101));
    try std.testing.expectEqual(refill_per_second, b.available(std.time.ns_per_s + 100));
    try std.testing.expectEqual(capacity, b.available(std.math.maxInt(u64)));
    try std.testing.expect(b.consume(std.math.maxInt(u64), capacity));
    try std.testing.expect(!b.consume(std.math.maxInt(u64), 1));
}

test "invalid debit has no effect and many subsecond refills conserve bytes" {
    var b: Budget = .{};
    try std.testing.expect(!b.consume(0, 0));
    try std.testing.expect(!b.consume(0, capacity + 1));
    try std.testing.expect(b.consume(0, capacity));
    for (1..1001) |i| _ = b.available(i * std.time.ns_per_ms);
    try std.testing.expectEqual(refill_per_second, b.credit);
    try std.testing.expectEqual(capacity, b.charged);
    try std.testing.expect(b.consume(std.time.ns_per_s, 123));
    try std.testing.expectEqual(refill_per_second - 123, b.credit);
}

test "one long idle refills the entire larger burst without banking more" {
    var b: Budget = .{};
    try std.testing.expect(b.consume(0, capacity));
    try std.testing.expectEqual(capacity, b.available(40 * std.time.ns_per_s));
    try std.testing.expect(b.consume(40 * std.time.ns_per_s, capacity));
    try std.testing.expect(!b.consume(40 * std.time.ns_per_s, 1));
    try std.testing.expectEqual(capacity, b.available(std.math.maxInt(u64)));
    try std.testing.expectEqual(@as(u64, 2), b.moves);
}

test "330K real rows fit once while insufficient remaining credit cannot debit" {
    // The sealed Flash layout has three ratio-2 sources and one ratio-1 source.
    const bytes: u64 = 356 * 330000 + 1068 * (330000 / 2);
    var b: Budget = .{};
    try std.testing.expect(bytes > 256 * 1024 * 1024 and bytes < capacity);
    try std.testing.expect(b.consume(0, bytes));
    const remaining = b.credit;
    try std.testing.expect(!b.consume(0, bytes));
    try std.testing.expectEqual(remaining, b.credit);
    try std.testing.expectEqual(bytes, b.charged);
    try std.testing.expectEqual(@as(u64, 1), b.moves);
    try std.testing.expectEqual(remaining + refill_per_second, b.available(std.time.ns_per_s));
}
