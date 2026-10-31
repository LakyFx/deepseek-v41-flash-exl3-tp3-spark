const std = @import("std");

pub const Span = struct { first: usize, count: usize };

// The two column partners own disjoint token rows of the same hash columns.
// Decode, short tails, and every unmodified topology retain the original path.
pub fn rows(n: usize, pair: ?usize) Span {
    const p = pair orelse return .{ .first = 0, .count = n };
    std.debug.assert(p < 2);
    if (n < 64) return .{ .first = 0, .count = n };
    const cut = (n + 1) / 2;
    return if (p == 0) .{ .first = 0, .count = cut } else .{ .first = cut, .count = n - cut };
}

test "Engram token halves cover even and odd chunks exactly without changing decode" {
    for (0..8193) |n| {
        const old = rows(n, null);
        try std.testing.expectEqual(@as(usize, 0), old.first);
        try std.testing.expectEqual(n, old.count);
        const a = rows(n, 0);
        const b = rows(n, 1);
        if (n < 64) {
            try std.testing.expectEqual(old, a);
            try std.testing.expectEqual(old, b);
        } else {
            try std.testing.expectEqual(@as(usize, 0), a.first);
            try std.testing.expectEqual(a.count, b.first);
            try std.testing.expectEqual(n, a.count + b.count);
            try std.testing.expect(a.count == b.count or a.count == b.count + 1);
        }
    }
}
